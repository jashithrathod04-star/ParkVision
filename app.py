"""
ParkVision AI - Intelligent parking occupancy analytics (Streamlit + YOLO).

Run (Windows):   streamlit run app.py
Model location:  models/best.pt   (or upload one from the sidebar)

What makes detection more reliable than a single predict() call:
  * Colour order fixed (Ultralytics expects BGR numpy arrays, the old code sent RGB).
  * Inference at the resolution the weights were trained at (read from the checkpoint),
    plus an optional larger full-image pass.
  * Overlapping tile inference (slicing) so small / distant bays are seen at a useful scale.
  * Optional mirror test-time augmentation.
  * Detections from every pass are fused: duplicates merged, class decided by weighted vote,
    tile-edge cut boxes down-weighted, weak single-pass detections filtered, size outliers removed.
  * Quality-aware preprocessing (measured sharpness / brightness / contrast / resolution).
Nothing in this file invents detections, metrics or history; every number comes from the model
output or from files you provide.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html as _html
import io
import json
import os
import random
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

try:  # keep the UI alive (with a clear message) if ultralytics is missing
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover
    YOLO = None
    YOLO_IMPORT_ERROR = _exc

st.set_page_config(
    page_title="ParkVision AI | Urban Parking Intelligence",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================================
# CONSTANTS
# =====================================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "models" / "best.pt",
    APP_DIR / "best.pt",
    APP_DIR / "model" / "best.pt",
]
MAX_SIDE = 3200  # larger uploads are downscaled (keeps Streamlit Cloud memory safe)

C_CYAN, C_VIOLET, C_GREEN, C_RED, C_AMBER = "#38D9FF", "#9B7BFF", "#35E69B", "#FF526B", "#FFB84D"
RGB_STATUS = {"empty": (53, 230, 155), "occupied": (255, 82, 107), "unknown": (255, 184, 77)}
HEX_STATUS = {"empty": C_GREEN, "occupied": C_RED, "unknown": C_AMBER}
LEVEL_COLOR = {"Low": C_CYAN, "Moderate": C_AMBER, "High": C_RED, "None": "#8FA3BC"}

NAV = [
    ("Dashboard", "▦   Dashboard"),
    ("Vision Lab", "◉   Vision Lab"),
    ("Analytics", "▤   Analytics"),
    ("Model Lab", "⬡   Model Lab"),
    ("Export Centre", "⇩   Export Centre"),
]
STAGES = ["Preparing image", "Analysing parking spaces", "Merging detections", "Generating insights"]
_INFER_LOCK = threading.Lock()

LOGO_SVG = (
    '<svg viewBox="0 0 48 48" width="40" height="40" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#38D9FF"/><stop offset="1" stop-color="#9B7BFF"/></linearGradient></defs>'
    '<path d="M24 3 43 14v20L24 45 5 34V14z" fill="rgba(56,217,255,.08)" stroke="url(#lg)" stroke-width="2.2"/>'
    '<path d="M18 33V15h8.5a6 6 0 0 1 0 12H18" fill="none" stroke="url(#lg)" stroke-width="3" stroke-linecap="round"/>'
    '<circle cx="35" cy="14" r="2.2" fill="#35E69B"/></svg>'
)

# =====================================================================================
# SMALL HELPERS
# =====================================================================================
def _clean(s: str) -> str:
    """Markdown treats indented lines / blank lines as code / block breaks: flatten the HTML."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def H(html: str) -> None:
    st.markdown(_clean(html), unsafe_allow_html=True)


def _stretch(fn, *args, **kwargs):
    """Full-width widgets across Streamlit versions (width='stretch' vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_image(img, caption=None):
    return _stretch(st.image, img, caption=caption)


def show_df(df):
    return _stretch(st.dataframe, df, hide_index=True)


def show_plot(fig, key: str):
    try:
        import inspect
        has_width = "width" in inspect.signature(st.plotly_chart).parameters
    except Exception:
        has_width = False
    kw = {"width": "stretch"} if has_width else {"use_container_width": True}
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **kw)


def b64_jpeg(rgb: np.ndarray, max_side: int = 1100, quality: int = 82) -> str:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        rgb = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def esc(x) -> str:
    return _html.escape(str(x), quote=True)


def fmt_int(n) -> str:
    return f"{int(n):,}"


# =====================================================================================
# STATE
# =====================================================================================
def init_state() -> None:
    defaults = {
        "stage": "welcome", "authed": False, "user": "", "page": "Dashboard",
        "history": [], "analysis": None, "failures": [], "fails": 0, "lock_until": 0.0,
        "custom_model_path": None, "custom_model_hash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def nav(page: str) -> None:
    st.session_state.page = page


def go_stage(stage: str) -> None:
    st.session_state.stage = stage


def logout() -> None:
    for k in ("authed", "user", "analysis", "history", "failures"):
        st.session_state.pop(k, None)
    st.session_state.stage = "welcome"
    st.session_state.page = "Dashboard"


# =====================================================================================
# AUTHENTICATION (demonstration-grade, stated honestly in the UI)
# =====================================================================================
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(("parkvision::" + pw).encode("utf-8")).hexdigest()


def known_users() -> dict:
    users = {"assessor": _hash_pw("parkvision2026"), "demo": _hash_pw("demo")}
    try:  # optional: add users in .streamlit/secrets.toml  ->  [users]  name = "<sha256 of parkvision::password>"
        extra = st.secrets.get("users", {})
        for k in extra:
            users[str(k).lower()] = str(extra[k])
    except Exception:
        pass
    return users


def try_login(username: str, password: str) -> bool:
    users = known_users()
    expected = users.get(username.strip().lower(), _hash_pw("x" * 32))
    ok = hmac.compare_digest(_hash_pw(password), expected) and username.strip().lower() in users
    return bool(ok)


# =====================================================================================
# MODEL
# =====================================================================================
def _first_int(v, default=640):
    try:
        if isinstance(v, (list, tuple)):
            v = v[0]
        return int(v)
    except Exception:
        return default


@st.cache_resource(show_spinner="Loading detection model…")
def load_model(path: str, mtime: float):
    if YOLO is None:
        raise RuntimeError(f"ultralytics could not be imported: {YOLO_IMPORT_ERROR}")
    model = YOLO(path)
    ck = getattr(model, "ckpt", None)
    ck = ck if isinstance(ck, dict) else {}
    targs = ck.get("train_args") if isinstance(ck.get("train_args"), dict) else {}
    over = getattr(model, "overrides", None) or {}
    try:
        params = int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        params = 0
    info = {
        "names": dict(getattr(model, "names", {}) or {}),
        "task": getattr(model, "task", "detect"),
        "train_imgsz": _first_int(targs.get("imgsz") or over.get("imgsz"), 640),
        "params": params,
        "train_args": targs,
        "train_metrics": ck.get("train_metrics") if isinstance(ck.get("train_metrics"), dict) else None,
        "epoch": ck.get("epoch"),
        "date": ck.get("date"),
        "version": ck.get("version"),
        "file": Path(path).name,
        "size_mb": Path(path).stat().st_size / 1e6,
    }
    return model, info


def resolve_model_path() -> Path | None:
    custom = st.session_state.get("custom_model_path")
    if custom and Path(custom).is_file():
        return Path(custom)
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return p
    models_dir = APP_DIR / "models"
    if models_dir.is_dir():
        found = sorted(models_dir.glob("*.pt"))
        if found:
            return found[0]
    return None


def class_type(name, class_id: int, n_classes: int) -> str:
    n = str(name).lower().replace("_", "-").replace(" ", "-")
    if any(k in n for k in ("occupied", "busy", "taken", "full")):
        return "occupied"
    if any(k in n for k in ("empty", "free", "vacant", "available")):
        return "empty"
    if n_classes == 2:  # unrecognised names: fall back to the project's known ordering
        return "empty" if class_id == 0 else "occupied"
    return "unknown"


# =====================================================================================
# IMAGE QUALITY + ENHANCEMENT
# =====================================================================================
def assess_quality(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = 1000.0 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return {
        "width": w, "height": h, "megapixels": round(w * h / 1e6, 2),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def _gamma(rgb, g):
    lut = (np.power(np.arange(256) / 255.0, g) * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(rgb, lut)


def _clahe(rgb, clip=2.5):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def _unsharp(rgb, amount):
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.6)
    return cv2.addWeighted(rgb, 1 + amount, blur, -amount, 0)


def _upscale(rgb, factor):
    h, w = rgb.shape[:2]
    factor = min(factor, 3600 / max(h, w))
    if factor <= 1.02:
        return rgb, 1.0
    out = cv2.resize(rgb, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
    return out, out.shape[1] / w


def enhance_image(rgb: np.ndarray, q: dict, d: dict):
    """Returns (processed_rgb, scale_factor_applied, human_readable_steps)."""
    mode = d["enh_mode"]
    steps: list[str] = []
    out, scale = rgb, 1.0
    if mode == "Off":
        return out, scale, steps
    if mode == "Auto":
        side = max(q["height"], q["width"])
        if side < 800:
            out, scale = _upscale(out, min(2.5, 1280 / side))
            steps.append(f"Upscaled x{scale:.2f} (small image, {q['width']}x{q['height']})")
        if q["brightness"] < 75:
            g = float(np.clip(np.log(0.45) / np.log(max(q["brightness"], 20) / 255.0), 0.5, 1.0))
            out = _gamma(out, g)
            steps.append(f"Brightened (gamma {g:.2f}, mean brightness {q['brightness']:.0f})")
        elif q["brightness"] > 205:
            out = _gamma(out, 1.25)
            steps.append(f"Toned down highlights (gamma 1.25, mean brightness {q['brightness']:.0f})")
        if q["contrast"] < 42 or q["brightness"] < 75:
            out = _clahe(out, 2.5)
            steps.append(f"Local contrast (CLAHE), contrast {q['contrast']:.0f}")
        if q["sharpness"] < 80:
            out = _unsharp(out, 0.8)
            steps.append(f"Sharpened (blur score {q['sharpness']:.0f})")
        return out, scale, steps
    # Manual
    if d["m_up"] > 1.02:
        out, scale = _upscale(out, d["m_up"])
        steps.append(f"Upscaled x{scale:.2f}")
    if d["m_denoise"]:
        out = cv2.bilateralFilter(out, 5, 40, 40)
        steps.append("Denoised (bilateral)")
    if abs(d["m_gamma"] - 1.0) > 0.01:
        out = _gamma(out, d["m_gamma"])
        steps.append(f"Gamma {d['m_gamma']:.2f}")
    if d["m_clahe"]:
        out = _clahe(out, 2.5)
        steps.append("Local contrast (CLAHE)")
    if d["m_sharp"] > 0.01:
        out = _unsharp(out, d["m_sharp"])
        steps.append(f"Sharpen {d['m_sharp']:.2f}")
    return out, scale, steps


# =====================================================================================
# INFERENCE PLAN, EXECUTION, FUSION
# =====================================================================================
def make_tiles(h: int, w: int, ts: int, overlap: float):
    ts = int(ts)
    step = max(1, int(ts * (1 - overlap)))
    xs = list(range(0, max(w - ts, 0) + 1, step))
    ys = list(range(0, max(h - ts, 0) + 1, step))
    if xs[-1] + ts < w:
        xs.append(max(w - ts, 0))
    if ys[-1] + ts < h:
        ys.append(max(h - ts, 0))
    tiles = {(x, y, min(x + ts, w), min(y + ts, h)) for y in ys for x in xs}
    return sorted(tiles, key=lambda t: (t[1], t[0]))


def build_plan(h: int, w: int, d: dict, train_sz: int):
    mode, main = d["mode"], int(d["imgsz"])
    jobs = []
    if mode == "Fast":
        jobs.append({"kind": "full", "imgsz": main, "flip": False, "label": f"Full image @ {main}px"})
    else:
        jobs.append({"kind": "full", "imgsz": train_sz, "flip": False, "label": f"Full image @ {train_sz}px (training size)"})
        if main != train_sz:
            jobs.append({"kind": "full", "imgsz": main, "flip": False, "label": f"Full image @ {main}px"})
        sizes = [int(d["tile"])]
        if mode == "Maximum":
            sizes.append(int(d["tile"] * 1.5))
        for ts in sizes:
            if max(h, w) > ts * 1.15:
                tiles = make_tiles(h, w, ts, d["overlap"])
                while len(tiles) > 48:
                    ts = int(ts * 1.2)
                    tiles = make_tiles(h, w, ts, d["overlap"])
                jobs.append({"kind": "tiles", "imgsz": train_sz, "tiles": tiles, "flip": False,
                             "label": f"{len(tiles)} overlapping tiles of {ts}px @ {train_sz}px"})
    if (d["flip"] or mode == "Maximum") and mode != "Fast":
        jobs.append({"kind": "full", "imgsz": main, "flip": True, "label": f"Mirrored full image @ {main}px"})
    return jobs


def plan_units(jobs) -> int:
    return sum(len(j["tiles"]) if j["kind"] == "tiles" else 1 for j in jobs)


def infer(model, imgs, imgsz, conf, iou, max_det):
    with _INFER_LOCK:
        results = model.predict(source=imgs, imgsz=int(imgsz), conf=float(conf), iou=float(iou),
                                max_det=int(max_det), agnostic_nms=True, verbose=False)
    outs = []
    for r in results:
        b = r.boxes
        if b is None or len(b) == 0:
            outs.append(np.zeros((0, 6), np.float32))
            continue
        outs.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                    b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
    return outs


def _iou_ioma(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    b = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    return inter / np.maximum(a + b - inter, 1e-9), inter / np.minimum(a, b)


def fuse_detections(arr: np.ndarray, iou_thr: float, ioma_thr: float = 0.85):
    """arr columns: x1 y1 x2 y2 conf cls pass_id weight. Greedy clustering across all passes."""
    if len(arr) == 0:
        return []
    order = np.argsort(-arr[:, 4])
    used = np.zeros(len(arr), bool)
    boxes = arr[:, :4]
    fused = []
    for i in order:
        if used[i]:
            continue
        iou, ioma = _iou_ioma(boxes[i], boxes)
        member = (~used) & ((iou >= iou_thr) | (ioma >= ioma_thr))
        member[i] = True
        idx = np.where(member)[0]
        used[idx] = True
        grp = arr[idx]
        scores = {int(c): float(grp[grp[:, 5] == c, 4].sum()) for c in np.unique(grp[:, 5])}
        win = max(scores, key=scores.get)
        wg = grp[grp[:, 5] == win]
        wts = wg[:, 4] * wg[:, 7]
        wts = wts / max(wts.sum(), 1e-9)
        box = (wg[:, :4] * wts[:, None]).sum(axis=0)
        fused.append({
            "box": box, "cls": win, "conf": float(wg[:, 4].max()),
            "votes": int(len(np.unique(grp[:, 6]))),
            "agree": float(scores[win] / max(sum(scores.values()), 1e-9)),
        })
    return fused


def assign_ids(dets: list) -> list:
    if not dets:
        return dets
    med_h = max(float(np.median([d["y2"] - d["y1"] for d in dets])), 1.0)
    dets = sorted(dets, key=lambda d: (round(((d["y1"] + d["y2"]) / 2) / med_h), d["x1"]))
    width = max(2, len(str(len(dets))))
    for n, d in enumerate(dets, 1):
        d["id"] = f"S{n:0{width}d}"
    return dets


def run_analysis(rgb, name, file_hash, model, info, d, cb):
    t0 = time.perf_counter()
    cb(0, "Measuring image quality", 0.02)
    quality = assess_quality(rgb)
    proc, scale, steps = enhance_image(rgb, quality, d)
    cb(0, steps[0] if steps else "No enhancement needed", 0.07)
    ph, pw = proc.shape[:2]
    bgr = np.ascontiguousarray(cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))  # Ultralytics expects BGR arrays
    jobs = build_plan(ph, pw, d, info["train_imgsz"])
    units, done = plan_units(jobs), 0
    raw, log = [], []

    for j_idx, job in enumerate(jobs):
        t1 = time.perf_counter()
        found = 0
        if job["kind"] == "full":
            cb(1, f"{job['label']}", 0.08 + 0.78 * done / units)
            img = np.ascontiguousarray(bgr[:, ::-1]) if job["flip"] else bgr
            out = infer(model, [img], job["imgsz"], d["conf"], d["iou"], d["max_det"])[0]
            if job["flip"] and len(out):
                x1 = pw - out[:, 2]; x2 = pw - out[:, 0]
                out[:, 0], out[:, 2] = x1, x2
            if len(out):
                raw.append(np.hstack([out, np.full((len(out), 1), j_idx, np.float32), np.ones((len(out), 1), np.float32)]))
                found += len(out)
            done += 1
        else:
            tiles = job["tiles"]
            for s in range(0, len(tiles), 4):
                batch = tiles[s:s + 4]
                cb(1, f"Tile {min(s + len(batch), len(tiles))}/{len(tiles)} - {job['label']}", 0.08 + 0.78 * done / units)
                crops = [np.ascontiguousarray(bgr[y0:y1, x0:x1]) for (x0, y0, x1, y1) in batch]
                outs = infer(model, crops, job["imgsz"], d["conf"], d["iou"], d["max_det"])
                for (x0, y0, x1, y1), o in zip(batch, outs):
                    if not len(o):
                        continue
                    tw, th = x1 - x0, y1 - y0
                    cut = (((o[:, 0] <= 3) & (x0 > 0)) | ((o[:, 2] >= tw - 3) & (x1 < pw)) |
                           ((o[:, 1] <= 3) & (y0 > 0)) | ((o[:, 3] >= th - 3) & (y1 < ph)))
                    o[:, [0, 2]] += x0
                    o[:, [1, 3]] += y0
                    wts = np.where(cut, 0.15, 1.0).astype(np.float32)[:, None]
                    raw.append(np.hstack([o, np.full((len(o), 1), j_idx, np.float32), wts]))
                    found += len(o)
                done += len(batch)
        log.append({"pass": job["label"], "raw_detections": int(found), "seconds": round(time.perf_counter() - t1, 2)})

    cb(2, "Fusing detections from all passes", 0.88)
    arr = np.vstack(raw) if raw else np.zeros((0, 8), np.float32)
    fused = fuse_detections(arr, d["merge_iou"])
    n_fused = len(fused)

    n_passes = len(jobs)
    dropped_consensus = dropped_geo = 0
    if d["consensus"] and n_passes > 1:
        kept = [f for f in fused if f["votes"] >= 2 or f["conf"] >= d["strong"]]
        dropped_consensus = len(fused) - len(kept)
        fused = kept
    if d["geo"] and len(fused) >= 8:
        areas = np.array([(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]) for f in fused])
        med = float(np.median(areas))
        kept = [f for f, a in zip(fused, areas) if 0.12 * med <= a <= 8 * med]
        dropped_geo = len(fused) - len(kept)
        fused = kept

    H0, W0 = rgb.shape[:2]
    names, n_cls = info["names"], len(info["names"])
    dets = []
    for f in fused:
        x1, y1, x2, y2 = (f["box"] / scale).tolist()
        x1, x2 = sorted((int(round(np.clip(x1, 0, W0 - 1))), int(round(np.clip(x2, 0, W0 - 1)))))
        y1, y2 = sorted((int(round(np.clip(y1, 0, H0 - 1))), int(round(np.clip(y2, 0, H0 - 1)))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cid = int(f["cls"])
        cname = names.get(cid, str(cid))
        dets.append({
            "id": "", "status": class_type(cname, cid, n_cls), "model_class": str(cname),
            "confidence": round(f["conf"], 4), "votes": f["votes"], "agreement": round(f["agree"], 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1,
        })
    dets = assign_ids(dets)
    cb(3, "Generating insights", 0.97)
    return {
        "file_hash": file_hash, "name": name, "timestamp": datetime.now(), "rgb": rgb,
        "size": (W0, H0), "dets": dets, "quality": quality, "steps": steps, "scale": scale,
        "log": log, "passes": n_passes, "duration": round(time.perf_counter() - t0, 2),
        "dcfg": d, "raw_total": int(len(arr)), "fused_total": n_fused,
        "dropped_consensus": dropped_consensus, "dropped_geo": dropped_geo,
    }


# =====================================================================================
# ANALYTICS
# =====================================================================================
def compute_stats(dets: list) -> dict:
    occ = sum(1 for d in dets if d["status"] == "occupied")
    emp = sum(1 for d in dets if d["status"] == "empty")
    unk = sum(1 for d in dets if d["status"] == "unknown")
    total = occ + emp
    occupancy = occ / total * 100 if total else 0.0
    avail_pct = emp / total * 100 if total else 0.0
    if total == 0:
        level, text = "None", "No parking spaces were detected. Try a clearer image, a wider view, or a lower sensitivity."
    elif emp == 0:
        level, text = "High", "Parking is full. Try another location."
    elif occupancy > 75:
        level, text = "High", "Parking is highly occupied. Consider another parking area."
    elif occupancy >= 40:
        level, text = "Moderate", "Parking is moderately occupied. Check the available spaces."
    else:
        level, text = "Low", "Parking availability is good. Proceed to this area."
    if total and emp:
        text += f" {emp} of {total} detected spaces appear available in this image."
    return {"total": total, "occupied": occ, "empty": emp, "unknown": unk, "occupancy": occupancy,
            "availability": avail_pct, "level": level, "text": text}


def active_dets(res: dict, r: dict) -> list:
    return [d for d in res["dets"] if d["confidence"] >= r["disp_conf"]]


def render_annotated(rgb: np.ndarray, dets: list, r: dict, highlight: str | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    sc = max(1.0, max(h, w) / 1100.0)
    th = max(1, int(round(r["thick"] * sc)))
    visible = [d for d in dets if r["show"].get(d["status"], True)]
    overlay = rgb.copy()
    if r["fill"] > 0:
        for d in visible:
            cv2.rectangle(overlay, (d["x1"], d["y1"]), (d["x2"], d["y2"]), RGB_STATUS[d["status"]], -1)
    out = cv2.addWeighted(overlay, r["fill"], rgb, 1 - r["fill"], 0) if r["fill"] > 0 else rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in visible:
        col = RGB_STATUS[d["status"]]
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th, cv2.LINE_AA)
        cl = int(max(4, min(x2 - x1, y2 - y1) * 0.28))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(out, (px, py), (px + dx * cl, py), (255, 255, 255), th + 1, cv2.LINE_AA)
            cv2.line(out, (px, py), (px, py + dy * cl), (255, 255, 255), th + 1, cv2.LINE_AA)
        if (r["ids"] or r["conf_lbl"]) and (x2 - x1) >= 26 * sc:
            text = d["id"] if r["ids"] else ""
            if r["conf_lbl"]:
                text = (text + " " if text else "") + f"{d['confidence']:.2f}"
            fs = 0.42 * sc
            (tw, tht), bl = cv2.getTextSize(text, font, fs, 1)
            top = y1 - tht - bl - 6
            if top < 0:
                top = y1 + 1
            cv2.rectangle(out, (x1, top), (min(w - 1, x1 + tw + 8), top + tht + bl + 6), col, -1)
            cv2.putText(out, text, (x1 + 4, top + tht + 3), font, fs, (7, 11, 20), 1, cv2.LINE_AA)
    if highlight:
        for d in dets:
            if d["id"] == highlight:
                pad = int(4 * sc)
                cv2.rectangle(out, (d["x1"] - pad, d["y1"] - pad), (d["x2"] + pad, d["y2"] + pad), (255, 255, 255), th + 2, cv2.LINE_AA)
                cv2.rectangle(out, (d["x1"] - pad, d["y1"] - pad), (d["x2"] + pad, d["y2"] + pad), (56, 217, 255), max(1, th - 1), cv2.LINE_AA)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = min(a.shape[0], 1400)
    def fit(x):
        s = h / x.shape[0]
        return cv2.resize(x, (int(x.shape[1] * s), h), interpolation=cv2.INTER_AREA)
    gap = np.full((h, 16, 3), (7, 11, 20), np.uint8)
    return np.hstack([fit(a), gap, fit(b)])


def crop_slot(rgb, d, pad=0.35, min_w=240):
    h, w = rgb.shape[:2]
    pw, ph = int((d["x2"] - d["x1"]) * pad), int((d["y2"] - d["y1"]) * pad)
    crop = rgb[max(0, d["y1"] - ph):min(h, d["y2"] + ph), max(0, d["x1"] - pw):min(w, d["x2"] + pw)].copy()
    if crop.shape[1] < min_w:
        s = min_w / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


# =====================================================================================
# REPORTS / EXPORT
# =====================================================================================
def build_report_json(res, stats, r) -> dict:
    return {
        "application": "ParkVision AI",
        "analysed_at": res["timestamp"].isoformat(timespec="seconds"),
        "image": res["name"], "image_size": {"width": res["size"][0], "height": res["size"][1]},
        "summary": {
            "total_slots": stats["total"], "occupied": stats["occupied"], "available": stats["empty"],
            "unclassified_detections": stats["unknown"],
            "occupancy_percent": round(stats["occupancy"], 2), "availability_percent": round(stats["availability"], 2),
            "congestion": stats["level"], "recommendation": stats["text"],
        },
        "image_quality": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["quality"].items()},
        "preprocessing_applied": res["steps"] or ["none"],
        "inference": {
            "passes": res["passes"], "pass_log": res["log"], "duration_seconds": res["duration"],
            "raw_detections": res["raw_total"], "after_fusion": res["fused_total"],
            "removed_by_consensus_filter": res["dropped_consensus"], "removed_by_size_filter": res["dropped_geo"],
            "settings": res["dcfg"], "display_confidence_threshold": r["disp_conf"],
        },
        "slots": active_dets(res, r),
        "note": "Estimate computed from the uploaded image only; not live parking availability.",
    }


def build_report_md(res, stats, r, model_file: str) -> str:
    j = build_report_json(res, stats, r)
    s = j["summary"]
    lines = [
        "# ParkVision AI - Analysis report", "",
        f"- **Image:** {res['name']} ({res['size'][0]}x{res['size'][1]})",
        f"- **Analysed:** {j['analysed_at']}",
        f"- **Model file:** {model_file}", "",
        "## Result", "",
        f"| Total slots | Occupied | Available | Occupancy | Availability | Congestion |",
        f"|---|---|---|---|---|---|",
        f"| {s['total_slots']} | {s['occupied']} | {s['available']} | {s['occupancy_percent']}% | {s['availability_percent']}% | {s['congestion']} |", "",
        f"**Recommendation:** {s['recommendation']}", "",
        "## Image quality", "",
    ]
    q = j["image_quality"]
    lines += [f"- Resolution {q['width']}x{q['height']}, sharpness {q['sharpness']}, brightness {q['brightness']}, contrast {q['contrast']}",
              "- Preprocessing: " + "; ".join(j["preprocessing_applied"]), "", "## Inference", ""]
    for p in res["log"]:
        lines.append(f"- {p['pass']}: {p['raw_detections']} raw detections, {p['seconds']}s")
    lines += [f"- Total time {res['duration']}s; {res['raw_total']} raw -> {res['fused_total']} fused -> {len(j['slots'])} reported", "",
              "> Estimate computed from the uploaded image only. It is not live parking availability."]
    return "\n".join(lines)


def build_zip(res, stats, r, annotated, model_file) -> bytes:
    df = pd.DataFrame(active_dets(res, r))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotated.png", png_bytes(annotated))
        z.writestr("original.png", png_bytes(res["rgb"]))
        z.writestr("slots.csv", df.to_csv(index=False))
        z.writestr("report.json", json.dumps(build_report_json(res, stats, r), indent=2, default=str))
        z.writestr("report.md", build_report_md(res, stats, r, model_file))
    return buf.getvalue()


# =====================================================================================
# CSS / BACKGROUND
# =====================================================================================
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
:root{--obsidian:#070B14;--midnight:#0C1425;--cyan:#38D9FF;--violet:#9B7BFF;--green:#35E69B;--red:#FF526B;--amber:#FFB84D;
--text:#E8F0F9;--muted:#8FA3BC;--line:rgba(120,170,255,.16);
--glass:linear-gradient(145deg,rgba(22,36,66,.66),rgba(12,20,37,.56));
--glow:0 0 0 1px rgba(56,217,255,.30),0 0 30px rgba(56,217,255,.16);--blob:.55}
html,body,.stApp{background:#070B14!important;color:var(--text);font-family:'Inter',sans-serif}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:transparent!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"],#MainMenu,footer,.stDeployButton{visibility:hidden}
.block-container{position:relative;z-index:2;padding-top:2.2rem!important;padding-bottom:4rem!important;max-width:1340px}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.015em;color:#F4F8FF!important}
p,li,label,span{color:inherit}
a{color:var(--cyan)!important}
[data-testid="stElementContainer"]:has(.pv-css),.element-container:has(.pv-css){display:none}
[data-testid="stElementContainer"]:has(.pv-bg),.element-container:has(.pv-bg){height:0;margin:0 0 -1rem 0}
/* ---------- animated background ---------- */
.pv-bg{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;
background:radial-gradient(1100px 650px at 85% -10%,rgba(155,123,255,.13),transparent 60%),radial-gradient(900px 600px at 0% 105%,rgba(56,217,255,.11),transparent 60%),#070B14}
.pv-blob{position:absolute;border-radius:50%;filter:blur(90px);opacity:var(--blob);mix-blend-mode:screen;will-change:transform}
.pv-blob.b1{width:50vw;height:50vw;left:-12vw;top:-16vw;background:radial-gradient(circle,rgba(56,217,255,.55),transparent 66%);animation:drift1 26s ease-in-out infinite alternate}
.pv-blob.b2{width:46vw;height:46vw;right:-10vw;top:6vh;background:radial-gradient(circle,rgba(155,123,255,.55),transparent 66%);animation:drift2 31s ease-in-out infinite alternate}
.pv-blob.b3{width:38vw;height:38vw;left:30vw;bottom:-18vw;background:radial-gradient(circle,rgba(53,230,155,.22),transparent 66%);animation:drift3 36s ease-in-out infinite alternate}
@keyframes drift1{to{transform:translate(20vw,12vw) scale(1.18)}}
@keyframes drift2{to{transform:translate(-18vw,16vw) scale(.9)}}
@keyframes drift3{to{transform:translate(-14vw,-10vw) scale(1.25)}}
.pv-grid{position:absolute;left:-50%;right:-50%;bottom:-12%;height:62%;
background-image:linear-gradient(rgba(56,217,255,.16) 1px,transparent 1px),linear-gradient(90deg,rgba(56,217,255,.16) 1px,transparent 1px);
background-size:64px 64px;transform:perspective(560px) rotateX(64deg);transform-origin:center bottom;animation:gridmove 8s linear infinite;
-webkit-mask-image:linear-gradient(to top,rgba(0,0,0,.85),transparent 88%);mask-image:linear-gradient(to top,rgba(0,0,0,.85),transparent 88%)}
@keyframes gridmove{to{background-position:0 64px}}
.pv-p{position:absolute;bottom:-12px;width:var(--s);height:var(--s);border-radius:50%;background:var(--c);box-shadow:0 0 12px var(--c);opacity:0;animation:rise var(--d) linear var(--dl) infinite}
@keyframes rise{0%{transform:translate(0,0);opacity:0}10%{opacity:.85}90%{opacity:.5}100%{transform:translate(var(--x),-112vh);opacity:0}}
/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(12,20,37,.94),rgba(7,11,20,.97))!important;border-right:1px solid var(--line);backdrop-filter:blur(20px)}
[data-testid="stSidebar"] .stButton{width:100%}
[data-testid="stSidebar"] .stButton>button{width:100%;justify-content:flex-start;text-align:left}
.brand-row{display:flex;align-items:center;gap:12px;margin:2px 0 10px}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:19px;letter-spacing:.06em;background:linear-gradient(90deg,#fff,#38D9FF 60%,#9B7BFF);-webkit-background-clip:text;background-clip:text;color:transparent}
.brand-sub{font-size:11px;color:var(--muted);letter-spacing:.04em}
.userchip{display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid var(--line);border-radius:14px;background:rgba(14,24,44,.55);margin-bottom:12px;font-size:13px}
.userchip b{color:#fff}.avatar{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:#04101c;background:linear-gradient(135deg,#38D9FF,#9B7BFF)}
/* ---------- glass ---------- */
.glass{position:relative;background:var(--glass);border:1px solid var(--line);border-radius:20px;padding:22px 24px;margin-bottom:6px;
backdrop-filter:blur(18px) saturate(150%);-webkit-backdrop-filter:blur(18px) saturate(150%);
box-shadow:0 18px 50px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.07);overflow:hidden;
transition:transform .35s cubic-bezier(.2,.8,.2,1),border-color .35s,box-shadow .35s}
.glass:hover{transform:translateY(-3px);border-color:rgba(56,217,255,.38);box-shadow:0 22px 56px rgba(0,0,0,.5),var(--glow),inset 0 1px 0 rgba(255,255,255,.09)}
.glass::after{content:"";position:absolute;top:0;left:-70%;width:40%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.07),transparent);transform:skewX(-18deg);transition:left .9s ease;pointer-events:none}
.glass:hover::after{left:140%}
.glass.flat:hover{transform:none}
.glass h3,.glass h4{margin:0 0 6px}
.muted{color:var(--muted)}
.sect{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:20px;margin:26px 0 10px;color:#F4F8FF}
.page-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(30px,4.2vw,50px);line-height:1.08;margin:0 0 6px;letter-spacing:-.02em;color:#F4F8FF}
.page-sub{color:var(--muted);font-size:16px;max-width:760px;margin-bottom:22px;line-height:1.55}
/* ---------- KPI ---------- */
@property --n{syntax:'<integer>';initial-value:0;inherits:false}
.kpi{position:relative;padding:20px 22px;border-radius:20px;border:1px solid var(--line);background:var(--glass);backdrop-filter:blur(16px);
box-shadow:0 14px 40px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.06);overflow:hidden;min-height:132px;transition:transform .3s,border-color .3s}
.kpi:hover{transform:translateY(-3px);border-color:rgba(56,217,255,.38)}
.kpi::before{content:"";position:absolute;left:0;top:18%;bottom:18%;width:3px;border-radius:3px;background:var(--k,#38D9FF);box-shadow:0 0 14px var(--k,#38D9FF)}
.kpi-label{color:var(--muted);font-size:13.5px;font-weight:500}
.kpi-val{display:flex;align-items:baseline;gap:2px;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:46px;line-height:1.1;margin-top:8px;color:#fff}
.kpi-num{counter-reset:n var(--n);animation:countup 1.3s cubic-bezier(.2,.7,.2,1) both}
.kpi-num::after{content:counter(n)}
@keyframes countup{from{--n:0}}
.kpi-suf{font-size:22px;color:var(--k,#38D9FF);font-weight:600}
.kpi-note{color:var(--muted);font-size:12.5px;margin-top:4px}
/* ---------- chips ---------- */
.chip{display:inline-flex;align-items:center;gap:7px;padding:5px 12px;border-radius:999px;font-size:12.5px;font-weight:600;border:1px solid var(--line);background:rgba(14,24,44,.6);margin:0 6px 6px 0}
.chip i{width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor}
.chip.ok{color:var(--green);border-color:rgba(53,230,155,.35)}.chip.bad{color:var(--red);border-color:rgba(255,82,107,.4)}
.chip.warn{color:var(--amber);border-color:rgba(255,184,77,.4)}.chip.info{color:var(--cyan);border-color:rgba(56,217,255,.4)}.chip.vio{color:var(--violet);border-color:rgba(155,123,255,.4)}
.pulse i{animation:pulse 1.8s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35;transform:scale(1.5)}}
/* ---------- meter ---------- */
.meter{position:relative;height:16px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;box-shadow:inset 0 2px 6px rgba(0,0,0,.5)}
.meter-fill{height:100%;width:var(--w);border-radius:99px;background:linear-gradient(90deg,#38D9FF,var(--mc,#38D9FF));box-shadow:0 0 18px var(--mc,#38D9FF);animation:fillbar 1.4s cubic-bezier(.2,.7,.2,1) both}
@keyframes fillbar{from{width:0}}
.meter .tick{position:absolute;top:0;bottom:0;width:2px;background:rgba(7,11,20,.9)}
.meter-scale{position:relative;height:18px;margin-top:6px;font-size:12px;color:var(--muted)}
.meter-scale span{position:absolute;transform:translateX(-50%)}
.meter-scale span:first-child{transform:none}.meter-scale span:last-child{transform:translateX(-100%)}
.reco{border-left:3px solid var(--rc,#38D9FF);padding:14px 18px;border-radius:0 14px 14px 0;background:linear-gradient(90deg,rgba(56,217,255,.08),transparent);font-size:16px;line-height:1.55;color:#F4F8FF}
/* ---------- hero / welcome ---------- */
.hero{padding:6vh 0 3vh;text-align:center}
.hero-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;border:1px solid rgba(56,217,255,.35);background:rgba(56,217,255,.07);color:var(--cyan);font-size:13px;font-weight:600;animation:rev 1s both}
.hero-badge .dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green);animation:pulse 1.8s infinite}
.logo-reveal{display:flex;justify-content:center;align-items:center;gap:14px;margin:26px 0 8px;font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:.32em;font-size:15px;color:#cfe6ff;animation:rev 1.2s .15s both}
.logo-reveal svg{filter:drop-shadow(0 0 14px rgba(56,217,255,.6));animation:spin3d 9s linear infinite}
@keyframes spin3d{50%{transform:rotateY(180deg)}}
.hero-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(44px,8.2vw,104px);line-height:1.02;letter-spacing:-.035em;margin:10px 0 18px;
background:linear-gradient(100deg,#fff 0%,#38D9FF 32%,#9B7BFF 62%,#fff 100%);background-size:220% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shimmer 7s linear infinite,rev 1.3s .3s both}
@keyframes shimmer{to{background-position:220% 0}}
.hero-sub{max-width:640px;margin:0 auto;color:#a9bbd2;font-size:18px;line-height:1.6;animation:rev 1.3s .5s both}
@keyframes rev{from{opacity:0;transform:translateY(18px);filter:blur(6px)}to{opacity:1;transform:none;filter:none}}
.pwrap{position:relative}
.pgrid{display:grid;grid-template-columns:repeat(10,1fr);gap:8px}
.pslot{aspect-ratio:1/1.55;border-radius:8px;border:1.5px solid;position:relative}
.pslot.free{border-color:rgba(53,230,155,.85);background:rgba(53,230,155,.10);animation:glowg 3s ease-in-out infinite;animation-delay:calc(var(--i)*90ms)}
.pslot.occ{border-color:rgba(255,82,107,.85);background:rgba(255,82,107,.08)}
.pslot.occ i{position:absolute;inset:13% 17%;border-radius:7px 7px 9px 9px;background:linear-gradient(180deg,rgba(255,82,107,.62),rgba(255,82,107,.18))}
@keyframes glowg{50%{box-shadow:0 0 16px rgba(53,230,155,.5)}}
.proad{grid-column:1/-1;height:18px;background:repeating-linear-gradient(90deg,rgba(255,255,255,.32) 0 20px,transparent 20px 44px) center/100% 2px no-repeat}
.psweep{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--cyan);box-shadow:0 0 18px 4px rgba(56,217,255,.7);animation:sweepx 5s ease-in-out infinite alternate}
@keyframes sweepx{from{left:1%}to{left:99%}}
.feat h4{font-size:17px;margin-bottom:6px}.feat p{color:var(--muted);font-size:14.5px;line-height:1.55;margin:0}
.feat .ic{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;margin-bottom:12px;background:linear-gradient(135deg,rgba(56,217,255,.22),rgba(155,123,255,.22));border:1px solid var(--line);font-size:18px}
/* ---------- scanner ---------- */
.scanbox{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:22px;align-items:center}
@media(max-width:900px){.scanbox{grid-template-columns:1fr}}
.scan-img{position:relative;border-radius:14px;overflow:hidden;border:1px solid var(--line)}
.scan-img img{display:block;width:100%;filter:saturate(.8) brightness(.8)}
.scan-line{position:absolute;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,#38D9FF,transparent);box-shadow:0 0 28px 8px rgba(56,217,255,.55);animation:sweepy 2.2s ease-in-out infinite alternate}
@keyframes sweepy{from{top:0}to{top:calc(100% - 3px)}}
.scan-img::after{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(56,217,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(56,217,255,.12) 1px,transparent 1px);background-size:34px 34px}
.hexspin{width:54px;height:54px;animation:rot 6s linear infinite;filter:drop-shadow(0 0 10px rgba(56,217,255,.7))}
@keyframes rot{to{transform:rotate(360deg)}}
.stage{display:flex;align-items:center;gap:10px;padding:7px 0;color:var(--muted);font-size:14.5px}
.stage .sd{width:10px;height:10px;border-radius:50%;border:2px solid #3b4d68}
.stage.done{color:#d9e7f7}.stage.done .sd{background:var(--green);border-color:var(--green);box-shadow:0 0 10px var(--green)}
.stage.now{color:#fff;font-weight:600}.stage.now .sd{border-color:var(--cyan);animation:pulse 1.2s infinite;background:var(--cyan)}
.pbar{height:6px;border-radius:9px;background:rgba(255,255,255,.08);overflow:hidden;margin:12px 0 4px}
.pbar>div{height:100%;background:linear-gradient(90deg,#38D9FF,#9B7BFF);transition:width .4s}
.empty{text-align:center;padding:34px 20px}.empty h3{margin:12px 0 6px}.empty p{color:var(--muted);max-width:520px;margin:0 auto}
.thumb{display:flex;gap:14px;align-items:center}.thumb img{width:112px;height:76px;object-fit:cover;border-radius:12px;border:1px solid var(--line)}
.thumb b{color:#fff}.thumb small{color:var(--muted)}
/* ---------- native widgets ---------- */
.stButton>button,.stDownloadButton>button,[data-testid^="stBaseButton"]{border-radius:12px!important;font-weight:600!important;min-height:44px;transition:transform .2s,box-shadow .25s,border-color .25s,background .25s!important}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"],button[kind="secondary"]{background:rgba(16,28,52,.65)!important;color:var(--text)!important;border:1px solid var(--line)!important}
[data-testid="stBaseButton-secondary"]:hover,button[kind="secondary"]:hover{border-color:rgba(56,217,255,.6)!important;box-shadow:0 0 22px rgba(56,217,255,.22);transform:translateY(-2px);color:#fff!important}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"],button[kind="primary"],button[kind="primaryFormSubmit"]{
background:linear-gradient(110deg,#1FB6E0,#38D9FF 40%,#9B7BFF 80%,#38D9FF)!important;background-size:220% 100%!important;color:#04101c!important;border:0!important;
box-shadow:0 0 0 1px rgba(56,217,255,.55),0 10px 32px rgba(56,217,255,.30);animation:sheen 7s linear infinite}
@keyframes sheen{to{background-position:220% 0}}
[data-testid="stBaseButton-primary"]:hover,button[kind="primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover{transform:translateY(-2px);box-shadow:0 0 0 1px #38D9FF,0 14px 42px rgba(56,217,255,.5)}
[data-testid="stBaseButton-primary"]:active,button[kind="primary"]:active,[data-testid="stBaseButton-secondary"]:active{transform:scale(.97)}
button:focus-visible,input:focus-visible,[role="slider"]:focus-visible{outline:2px solid var(--cyan)!important;outline-offset:2px}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{background:rgba(10,18,34,.72)!important;border-radius:12px!important;border:1px solid var(--line)!important;transition:border-color .25s,box-shadow .25s}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:var(--cyan)!important;box-shadow:0 0 0 3px rgba(56,217,255,.18),0 0 26px rgba(56,217,255,.28)!important}
[data-baseweb="base-input"],[data-baseweb="input"] input{background:transparent!important;color:var(--text)!important}
[data-baseweb="slider"] div[role="slider"]{background:var(--cyan)!important;box-shadow:0 0 14px rgba(56,217,255,.8)}
[data-testid="stForm"]{background:var(--glass);border:1px solid var(--line)!important;border-radius:22px;padding:26px 26px 20px;backdrop-filter:blur(20px);box-shadow:0 24px 70px rgba(0,0,0,.55),0 0 0 1px rgba(56,217,255,.08),inset 0 1px 0 rgba(255,255,255,.07)}
[data-testid="stFileUploaderDropzone"]{background:rgba(12,22,42,.6)!important;border:1.5px dashed rgba(56,217,255,.45)!important;border-radius:18px!important;padding:26px!important;transition:all .3s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cyan)!important;box-shadow:0 0 34px rgba(56,217,255,.22);background:rgba(16,30,56,.7)!important}
[data-testid="stExpander"]{background:rgba(14,24,44,.55);border:1px solid var(--line)!important;border-radius:14px!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
button[data-baseweb="tab"]{font-weight:600}
[data-testid="stMetric"]{background:rgba(14,24,44,.6);border:1px solid var(--line);border-radius:14px;padding:14px}
[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line)}
@media(max-width:768px){.kpi-val{font-size:36px}.block-container{padding-left:1rem!important;padding-right:1rem!important}}
"""

AURORA_CSS = """
:root{--cyan:#38D9FF;--glow:0 0 0 1px rgba(155,123,255,.45),0 0 38px rgba(155,123,255,.28);--blob:.8}
.pv-bg{background:radial-gradient(1200px 700px at 80% -10%,rgba(155,123,255,.24),transparent 60%),radial-gradient(1000px 650px at 0% 100%,rgba(56,217,255,.2),transparent 60%),#070B14}
.glass{border-color:rgba(155,123,255,.22)}
"""
NO_MOTION_CSS = "*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}"
REDUCED_MEDIA = "@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}"
HIDE_SIDEBAR_CSS = '[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none!important}'


EXTRA_CSS = """
::selection{background:rgba(56,217,255,.35);color:#fff}
html{scroll-behavior:smooth}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:linear-gradient(180deg,rgba(56,217,255,.45),rgba(155,123,255,.45));border-radius:9px;border:2px solid #070B14}
.kpi::after{content:"";position:absolute;right:-34px;top:-34px;width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,var(--k,#38D9FF),transparent 70%);opacity:.14;pointer-events:none}
.ticker{overflow:hidden;white-space:nowrap;border:1px solid var(--line);border-radius:999px;background:rgba(10,18,34,.6);padding:10px 0;margin:4px 0 18px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 7%,#000 93%,transparent);mask-image:linear-gradient(90deg,transparent,#000 7%,#000 93%,transparent)}
.ticker-track{display:inline-flex;gap:34px;padding-left:34px;animation:tick 46s linear infinite}
.ticker:hover .ticker-track{animation-play-state:paused}
.ticker-track span{font-size:13px;color:#b7c8de}.ticker-track b{color:#fff;font-weight:600}.ticker-track .sep{color:var(--cyan);opacity:.7}
@keyframes tick{to{transform:translateX(-50%)}}
@property --ang{syntax:'<angle>';initial-value:0deg;inherits:false}
.cmd{position:relative;display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:26px;align-items:center;padding:30px 32px;margin-bottom:18px;border-radius:26px;overflow:hidden;
background:linear-gradient(120deg,rgba(22,36,66,.74),rgba(12,20,37,.55));border:1px solid var(--line);backdrop-filter:blur(18px);box-shadow:0 24px 70px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.07)}
.cmd::before{content:"";position:absolute;inset:0;border-radius:26px;padding:1.5px;pointer-events:none;background:conic-gradient(from var(--ang),transparent 0 55%,#38D9FF,#9B7BFF,transparent 92%);
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;animation:spinborder 8s linear infinite}
@keyframes spinborder{to{--ang:360deg}}
.cmd::after{content:"";position:absolute;right:-8%;top:-40%;width:55%;height:180%;background:radial-gradient(closest-side,rgba(155,123,255,.20),transparent);pointer-events:none}
.cmd-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:clamp(28px,3.6vw,46px);line-height:1.08;letter-spacing:-.02em;margin:12px 0 10px;background:linear-gradient(100deg,#fff,#9fe9ff 55%,#c6b5ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.cmd-sub{color:#a9bbd2;font-size:15.5px;line-height:1.62;max-width:600px}
@media(max-width:900px){.cmd{grid-template-columns:1fr;padding:22px}}
.ringwrap{position:relative;width:min(230px,100%);aspect-ratio:1;margin:0 auto;z-index:1}
.ring{width:100%;height:100%;transform:rotate(-90deg);overflow:visible}
.ring-bg{fill:none;stroke:rgba(255,255,255,.08);stroke-width:10}
.ring-fg{fill:none;stroke-width:10;stroke-linecap:round;stroke-dasharray:var(--c);stroke-dashoffset:var(--o);filter:drop-shadow(0 0 7px var(--sc));animation:ringfill 1.7s cubic-bezier(.2,.7,.2,1) both}
@keyframes ringfill{from{stroke-dashoffset:var(--c)}}
.ring-dash{fill:none;stroke:rgba(56,217,255,.38);stroke-width:1;stroke-dasharray:3 7;transform-box:fill-box;transform-origin:center;animation:rot 46s linear infinite}
.ring-c{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
.ring-c .kpi-val{justify-content:center;margin:0;font-size:44px}
.ring-c small{display:block;color:var(--muted);font-size:12.5px;margin-top:2px}
.mosaic{display:grid;grid-template-columns:repeat(auto-fill,minmax(26px,1fr));gap:6px;margin-top:12px}
.mosaic i{display:block;aspect-ratio:1/1.3;border-radius:7px;border:1.5px solid var(--mc);background:var(--mb);animation:popin .5s cubic-bezier(.2,.8,.2,1) both;animation-delay:min(calc(var(--i)*11ms),900ms);transition:transform .2s,box-shadow .2s}
.mosaic i:hover{transform:scale(1.25);box-shadow:0 0 14px var(--mc);position:relative;z-index:2}
@keyframes popin{from{opacity:0;transform:scale(.4)}}
.insight{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid var(--line);font-size:14.5px;line-height:1.55;color:#dbe7f5}
.insight:last-child{border-bottom:0}
.insight .ib{flex:0 0 4px;align-self:stretch;border-radius:4px;background:var(--ic);box-shadow:0 0 10px var(--ic)}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.mini{padding:12px 14px;border-radius:14px;border:1px solid var(--line);background:rgba(10,18,34,.55)}
.mini b{display:block;font-family:'Space Grotesk',sans-serif;font-size:24px;color:#fff;margin-top:2px}
.mini span{color:var(--muted);font-size:12.5px}
.rowbar{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:13px;color:var(--muted)}
.rowbar .lab{width:46px}.rowbar .num{width:56px;text-align:right;color:#fff}
.rowbar .trk{flex:1;height:12px;border-radius:9px;overflow:hidden;display:flex;background:rgba(255,255,255,.06)}
.rowbar .trk i{display:block;height:100%;width:var(--w);animation:fillbar 1.2s cubic-bezier(.2,.7,.2,1) both}
.spark{width:100%;height:92px;overflow:visible;margin-top:8px}
.spark .line{fill:none;stroke:url(#spg);stroke-width:2.6;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.7s ease forwards}
.spark .area{fill:url(#spa);opacity:.9}
@keyframes draw{to{stroke-dashoffset:0}}
.step{display:flex;gap:16px;align-items:flex-start}
.step .sn{flex:0 0 40px;height:40px;border-radius:12px;display:grid;place-items:center;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#04101c;background:linear-gradient(135deg,#38D9FF,#9B7BFF);box-shadow:0 0 20px rgba(56,217,255,.35)}
.step h4{margin:0 0 4px;font-size:17px}.step p{margin:0;color:var(--muted);font-size:14px;line-height:1.55}
.legend{display:flex;flex-wrap:wrap;gap:4px}
"""


def inject_css(theme="Midnight", reduced=False, effects=True, hide_sidebar=False) -> None:
    css = BASE_CSS + EXTRA_CSS + (AURORA_CSS if theme == "Aurora" else "") + REDUCED_MEDIA
    if reduced:
        css += NO_MOTION_CSS
    if hide_sidebar:
        css += HIDE_SIDEBAR_CSS
    st.markdown(_clean(f'<div class="pv-css"></div><style>{css}</style>'), unsafe_allow_html=True)
    if effects:
        rnd = random.Random(5)
        parts = []
        for _ in range(26):
            col = rnd.choice([C_CYAN, C_VIOLET, "#ffffff"])
            parts.append(
                f'<span class="pv-p" style="left:{rnd.randint(1, 99)}%;--s:{rnd.choice([2, 2, 3, 4])}px;--c:{col};'
                f'--d:{rnd.randint(14, 30)}s;--dl:-{rnd.randint(0, 28)}s;--x:{rnd.randint(-60, 60)}px"></span>')
        H(f'<div class="pv-bg"><div class="pv-blob b1"></div><div class="pv-blob b2"></div><div class="pv-blob b3"></div>'
          f'<div class="pv-grid"></div>{"".join(parts)}</div>')
    else:
        H('<div class="pv-bg"></div>')


# =====================================================================================
# UI COMPONENTS
# =====================================================================================
def kpi(label, value, note="", color=C_CYAN, suffix="", decimals=0):
    if decimals:
        ip, fp = f"{float(value):.{decimals}f}".split(".")
        tail = f".{fp}{suffix}"
    else:
        ip, tail = str(int(value)), suffix
    return (f'<div class="kpi" style="--k:{color}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val"><span class="kpi-num" style="--n:{ip}"></span><span class="kpi-suf">{tail}</span></div>'
            f'<div class="kpi-note">{note}</div></div>')


def kpi_row(stats):
    c = st.columns(4)
    items = [
        ("Total parking slots", stats["total"], "Classified empty + occupied", C_CYAN, "", 0),
        ("Available spaces", stats["empty"], f"{stats['availability']:.1f}% of detected slots", C_GREEN, "", 0),
        ("Occupied spaces", stats["occupied"], "Cars detected in slots", C_RED, "", 0),
        ("Parking utilisation", stats["occupancy"], "Occupied / total x 100", C_VIOLET, "%", 1),
    ]
    for col, (a, b, n, k, s, dec) in zip(c, items):
        with col:
            H(kpi(a, b, n, k, s, dec))


def meter_html(occ: float, level: str) -> str:
    mc = LEVEL_COLOR.get(level, C_CYAN)
    return (f'<div class="meter" style="--mc:{mc}"><div class="meter-fill" style="--w:{min(100, max(0, occ)):.1f}%"></div>'
            f'<span class="tick" style="left:40%"></span><span class="tick" style="left:75%"></span></div>'
            f'<div class="meter-scale"><span style="left:0">0% Low</span><span style="left:40%">40% Moderate</span>'
            f'<span style="left:75%">75% High</span><span style="left:100%">100%</span></div>')


def congestion_panel(stats):
    lvl = stats["level"]
    mc = LEVEL_COLOR.get(lvl, C_CYAN)
    chip = {"Low": "info", "Moderate": "warn", "High": "bad", "None": "vio"}[lvl]
    label = "No congestion data" if lvl == "None" else f"{lvl} congestion"
    H(f'<div class="glass flat"><div class="chip {chip} pulse"><i></i>{label}</div>'
      f'<h4 style="margin:6px 0 14px">Parking utilisation: {stats["occupancy"]:.1f}%</h4>'
      f'{meter_html(stats["occupancy"], lvl)}'
      f'<div class="reco" style="--rc:{mc};margin-top:18px">{stats["text"]}</div></div>')


def parking_preview(seed=11, rows=4, cols=10, occ_rate=0.62):
    rnd = random.Random(seed)
    cells, free, total = [], 0, 0
    for r in range(rows):
        if r == rows // 2:
            cells.append('<div class="proad"></div>')
        for c in range(cols):
            occ = rnd.random() < occ_rate
            total += 1
            free += 0 if occ else 1
            cells.append(f'<div class="pslot {"occ" if occ else "free"}" style="--i:{r * cols + c}">{"<i></i>" if occ else ""}</div>')
    return f'<div class="pwrap"><div class="pgrid">{"".join(cells)}</div><div class="psweep"></div></div>', free, total


def empty_state(title, text):
    grid, _, _ = parking_preview(seed=3, rows=2, cols=8, occ_rate=0.5)
    H(f'<div class="glass empty flat"><div style="max-width:420px;margin:0 auto;opacity:.55">{grid}</div><h3>{title}</h3><p>{text}</p></div>')


def style_fig(fig, height=330):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#C9D6E6", family="Inter"), margin=dict(l=10, r=10, t=36, b=10),
                      legend=dict(orientation="h", y=-0.15), title_font=dict(family="Space Grotesk", size=16, color="#F4F8FF"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def gauge_fig(occ, level):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=occ,
        number={"suffix": "%", "valueformat": ".1f", "font": {"color": "#F4F8FF", "family": "Space Grotesk", "size": 42}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#8FA3BC"}, "bar": {"color": LEVEL_COLOR.get(level, C_CYAN), "thickness": 0.3},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(56,217,255,.16)"}, {"range": [40, 75], "color": "rgba(255,184,77,.16)"},
                         {"range": [75, 100], "color": "rgba(255,82,107,.18)"}]}))
    return style_fig(fig, 250)


# =====================================================================================
# SIDEBAR (all settings)
# =====================================================================================
def sidebar_ui():
    with st.sidebar:
        H(f'<div class="brand-row">{LOGO_SVG}<div><div class="brand-name">PARKVISION AI</div><div class="brand-sub">Urban intelligence platform</div></div></div>')
        user = st.session_state.get("user") or "guest"
        H(f'<div class="userchip"><div class="avatar">{esc(user[:1].upper())}</div><div>Signed in as <b>{esc(user)}</b><br><span class="muted" style="font-size:11.5px">Demonstration access</span></div></div>')
        n_runs = len(st.session_state.history)
        H(f'<span class="chip ok pulse"><i></i>Engine ready</span><span class="chip vio"><i></i>{n_runs} analyses</span>')
        for key, label in NAV:
            st.button(label, key=f"nav_{key}", type="primary" if st.session_state.page == key else "secondary", on_click=nav, args=(key,))
        st.markdown("---")
        st.markdown("##### Settings")

        with st.expander("Detection engine", expanded=True):
            mode = st.select_slider("Detection mode", options=["Fast", "Accurate", "Maximum"], value="Accurate", key="s_mode",
                                    help="Fast: one pass. Accurate: training-size + large pass + overlapping tiles. Maximum: adds a wider tile grid and mirrored pass.")
            conf = st.slider("Sensitivity (confidence floor)", 0.05, 0.80, 0.15, 0.01, key="s_conf",
                             help="Lower finds more spaces but may add false positives. Final cleaning filters help offset this.")
            imgsz = st.select_slider("Full-image resolution", options=[640, 800, 960, 1280, 1536], value=960, key="s_imgsz",
                                     help="Extra full-image pass at this size (the model's own training size is always used too).")
            tile = st.slider("Tile size (px)", 320, 1280, 640, 32, key="s_tile",
                             help="Smaller tiles zoom in more, helping tiny / distant spaces. Used by Accurate and Maximum.")
            overlap = st.slider("Tile overlap", 0.15, 0.50, 0.30, 0.05, key="s_overlap")
            iou = st.slider("Per-pass overlap suppression (IoU)", 0.30, 0.85, 0.60, 0.05, key="s_iou",
                            help="Raise if adjacent spaces are merged, lower if you see doubled boxes.")
            flip = st.checkbox("Mirror test-time augmentation", value=False, key="s_flip", help="Adds a horizontally flipped pass (always on in Maximum).")
            auto_run = st.checkbox("Analyse automatically on upload", value=True, key="s_auto")

        with st.expander("Image enhancement"):
            enh_mode = st.radio("Mode", ["Auto", "Off", "Manual"], horizontal=True, key="s_enh",
                                help="Auto measures blur, brightness, contrast and size, then only applies what is needed.")
            m_up, m_clahe, m_gamma, m_sharp, m_dn = 1.0, False, 1.0, 0.0, False
            if enh_mode == "Manual":
                m_up = st.slider("Upscale", 1.0, 3.0, 1.0, 0.25, key="s_up")
                m_gamma = st.slider("Gamma (below 1 brightens)", 0.5, 1.8, 1.0, 0.05, key="s_gamma")
                m_sharp = st.slider("Sharpen", 0.0, 1.5, 0.0, 0.1, key="s_sharp")
                m_clahe = st.checkbox("Local contrast (CLAHE)", value=False, key="s_clahe")
                m_dn = st.checkbox("Denoise", value=False, key="s_dn")

        with st.expander("Result cleaning"):
            merge_iou = st.slider("Merge duplicate boxes (IoU)", 0.30, 0.80, 0.45, 0.05, key="s_merge")
            consensus = st.checkbox("Require agreement for weak detections", value=True, key="s_cons",
                                    help="A detection seen in only one pass must exceed the threshold below.")
            strong = st.slider("Accept single-pass detections above", 0.15, 0.90, 0.30, 0.05, key="s_strong")
            geo = st.checkbox("Remove size outliers", value=True, key="s_geo", help="Drops boxes far smaller / larger than the typical space (needs 8+ detections).")
            max_det = st.slider("Max detections per pass", 200, 2000, 1000, 100, key="s_maxdet")

        with st.expander("Overlay style"):
            disp_conf = st.slider("Display threshold", 0.05, 0.95, 0.15, 0.01, key="s_disp",
                                  help="Live filter. Does not re-run the model; counts update instantly.")
            show_empty = st.checkbox("Show available (green)", True, key="s_se")
            show_occ = st.checkbox("Show occupied (red)", True, key="s_so")
            show_unk = st.checkbox("Show unrecognised (amber)", True, key="s_su")
            fill = st.slider("Fill opacity", 0.0, 0.6, 0.18, 0.02, key="s_fill")
            thick = st.slider("Line thickness", 1, 6, 2, key="s_thick")
            ids = st.checkbox("Slot IDs", True, key="s_ids")
            conf_lbl = st.checkbox("Confidence labels", False, key="s_cl")

        with st.expander("Interface"):
            theme = st.selectbox("Theme", ["Midnight", "Aurora"], key="s_theme")
            reduced = st.toggle("Reduce motion", False, key="s_reduce")
            effects = st.toggle("Animated background", True, key="s_fx")

        with st.expander("Model file"):
            up = st.file_uploader("Use a different YOLO weights file (.pt)", type=["pt"], key="s_modelup")
            if up is not None:
                data = up.getvalue()
                h = hashlib.md5(data).hexdigest()
                if st.session_state.custom_model_hash != h:
                    path = Path(tempfile.gettempdir()) / f"pv_model_{h[:10]}.pt"
                    path.write_bytes(data)
                    st.session_state.custom_model_path = str(path)
                    st.session_state.custom_model_hash = h
                st.caption(f"Using uploaded weights: {up.name}")
            elif st.session_state.custom_model_path:
                st.session_state.custom_model_path = None
                st.session_state.custom_model_hash = None

        st.markdown("---")
        st.button("Sign out", key="logout", on_click=logout)

    dcfg = {
        "mode": mode, "conf": float(conf), "imgsz": int(imgsz), "tile": int(tile), "overlap": float(overlap), "iou": float(iou),
        "flip": bool(flip), "enh_mode": enh_mode, "m_up": float(m_up), "m_clahe": bool(m_clahe), "m_gamma": float(m_gamma),
        "m_sharp": float(m_sharp), "m_denoise": bool(m_dn), "merge_iou": float(merge_iou), "consensus": bool(consensus),
        "strong": float(strong), "geo": bool(geo), "max_det": int(max_det),
    }
    rcfg = {"disp_conf": float(disp_conf), "show": {"empty": show_empty, "occupied": show_occ, "unknown": show_unk},
            "fill": float(fill), "thick": int(thick), "ids": bool(ids), "conf_lbl": bool(conf_lbl)}
    ui = {"theme": theme, "reduced": reduced, "effects": effects, "auto_run": auto_run}
    return dcfg, rcfg, ui


# =====================================================================================
# PAGES: WELCOME + LOGIN
# =====================================================================================
def page_welcome():
    H(f'''<div class="hero">
        <div class="hero-badge"><span class="dot"></span>Computer vision for smarter cities</div>
        <div class="logo-reveal">{LOGO_SVG}<span>PARKVISION AI</span></div>
        <div class="hero-title">See the city.<br>Find your space.</div>
        <div class="hero-sub">Drop in a photo of any car park. ParkVision finds each bay, marks it free or taken,
        and turns the picture into a clear parking decision.</div></div>''')
    _, mid, _ = st.columns([2, 1.4, 2])
    with mid:
        st.button("Enter Command Center", type="primary", key="enter", on_click=go_stage, args=("login",))
    st.markdown("<br>", unsafe_allow_html=True)
    grid, free, total = parking_preview()
    _, centre, _ = st.columns([1, 3, 1])
    with centre:
        H(f'<div class="glass flat"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap">'
          f'<div><h4 style="margin:0">Live-style availability preview</h4>'
          f'<span class="muted" style="font-size:13px">Illustrative sample pattern. Not model output and not live data.</span></div>'
          f'<div><span class="chip ok"><i></i>{free} free</span><span class="chip bad"><i></i>{total - free} taken</span></div></div>{grid}</div>')
    H('<div style="text-align:center;margin:6px 0 4px"><span class="chip info"><i></i>Runs on your own YOLO model</span>'
      '<span class="chip ok"><i></i>Real detections only</span><span class="chip vio"><i></i>Exports you can submit</span></div>')
    f1, f2, f3 = st.columns(3)
    feats = [("⌖", "Sees small and distant bays", "Overlapping tile inference zooms into the image so far-away spaces are not missed."),
             ("◐", "Adapts to image quality", "Sharpness, brightness and contrast are measured, and only the fixes that are needed are applied."),
             ("⇩", "Evidence you can export", "Annotated image, slot table and a report with the real counts, ready to download.")]
    for col, (ic, t, p) in zip((f1, f2, f3), feats):
        with col:
            H(f'<div class="glass feat"><div class="ic">{ic}</div><h4>{t}</h4><p>{p}</p></div>')


def page_login():
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        H(f'<div style="text-align:center;margin:3vh 0 18px"><div class="logo-reveal" style="margin:0 0 8px">{LOGO_SVG}</div>'
          f'<div class="page-title" style="font-size:36px">Secure access</div>'
          f'<div class="muted">Sign in to open the command center.</div></div>')
        now = time.time()
        locked = now < st.session_state.lock_until
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="assessor", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Use the eye icon to show or hide", key="login_pw")
            submit = st.form_submit_button("Sign in", type="primary", disabled=locked)
        if locked:
            st.error(f"Too many failed attempts. Try again in {int(st.session_state.lock_until - now) + 1} seconds.")
        if submit:
            if not username or not password:
                st.warning("Enter both a username and a password.")
            elif try_login(username, password):
                st.session_state.update(authed=True, user=username.strip().lower(), fails=0, page="Dashboard")
                st.rerun()
            else:
                st.session_state.fails += 1
                if st.session_state.fails >= 5:
                    st.session_state.lock_until = time.time() + 30
                    st.session_state.fails = 0
                st.error("Incorrect username or password.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Continue as guest", key="guest", help="Demo access for assessors"):
                st.session_state.update(authed=True, user="guest", page="Dashboard")
                st.rerun()
        with c2:
            st.button("Back", key="back", on_click=go_stage, args=("welcome",))
        with st.expander("Demo credentials"):
            st.markdown("Username `assessor`, password `parkvision2026` (or use **Continue as guest**).")
        st.caption("Demonstration access: credentials are checked locally against salted hashes. "
                   "This is not production-grade security.")


# =====================================================================================
# PAGES: DASHBOARD
# =====================================================================================
def system_panel(info, model_file):
    try:
        import ultralytics
        uv = ultralytics.__version__
    except Exception:
        uv = "n/a"
    try:
        import torch
        tv = torch.__version__.split("+")[0]
        dev = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    except Exception:
        tv, dev = "n/a", "CPU"
    classes = ", ".join(f"{k}: {v}" for k, v in info["names"].items())
    rows = [("Model file", model_file), ("Task", info["task"]), ("Classes", classes), ("Trained at", f"{info['train_imgsz']} px"),
            ("Parameters", fmt_int(info["params"]) if info["params"] else "n/a"), ("Compute", dev), ("Ultralytics / PyTorch", f"{uv} / {tv}")]
    body = "".join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px">'
                   f'<span class="muted">{a}</span><span style="text-align:right;color:#fff">{b}</span></div>' for a, b in rows)
    H(f'<div class="glass flat"><div class="chip ok pulse"><i></i>System ready</div><h4 style="margin:4px 0 8px">Detection engine</h4>{body}</div>')


# ---------- additive dashboard components (all values come from real analyses) ----------
def row_breakdown(dets: list) -> list:
    classified = [d for d in dets if d["status"] in ("empty", "occupied")]
    if not classified:
        return []
    med_h = max(float(np.median([d["y2"] - d["y1"] for d in classified])), 1.0)
    bands: dict = {}
    for d in classified:
        k = round(((d["y1"] + d["y2"]) / 2) / med_h)
        bands.setdefault(k, {"empty": 0, "occupied": 0})[d["status"]] += 1
    out = []
    for n, k in enumerate(sorted(bands), 1):
        r = bands[k]
        out.append({"row": n, "empty": r["empty"], "occupied": r["occupied"], "total": r["empty"] + r["occupied"]})
    return out


def build_insights(res: dict, stats: dict, dets: list) -> list:
    items = []
    lvl_kind = {"Low": "info", "Moderate": "warn", "High": "bad", "None": "vio"}[stats["level"]]
    items.append((lvl_kind, stats["text"]))
    rows = row_breakdown(dets)
    if rows and stats["empty"] > 0:
        best = max(rows, key=lambda r: (r["empty"], -r["row"]))
        if best["empty"] > 0 and len(rows) > 1:
            items.append(("ok", f"Best chance of a space: row {best['row']} from the top has {best['empty']} of {best['total']} spaces free."))
    if dets:
        mean_c = float(np.mean([d["confidence"] for d in dets]))
        if mean_c < 0.5:
            items.append(("warn", f"Average model confidence is low ({mean_c:.2f}). Try Maximum mode, a lower tile size, or a sharper photo."))
        elif mean_c >= 0.8:
            items.append(("ok", f"Detections are high-confidence (average {mean_c:.2f})."))
    q = res["quality"]
    if q["sharpness"] < 80:
        items.append(("warn", f"The image is soft (blur score {q['sharpness']:.0f}). A sharper photo would improve reliability."))
    if q["brightness"] < 75 or q["brightness"] > 205:
        items.append(("warn", f"Lighting is difficult (mean brightness {q['brightness']:.0f}/255)."))
    if res["steps"]:
        items.append(("info", "Preprocessing applied: " + "; ".join(res["steps"]) + "."))
    if res["dropped_consensus"] or res["dropped_geo"]:
        items.append(("info", f"Cleaning removed {res['dropped_consensus']} weak single-pass and {res['dropped_geo']} size-outlier detections."))
    if 0 < stats["total"] < 5:
        items.append(("warn", f"Only {stats['total']} spaces were found. If the lot is larger, try Maximum mode or a smaller tile size."))
    if stats["unknown"]:
        items.append(("warn", f"{stats['unknown']} detections had an unrecognised class and were left out."))
    return items


def insights_html(items: list) -> str:
    col = {"ok": C_GREEN, "info": C_CYAN, "warn": C_AMBER, "bad": C_RED, "vio": C_VIOLET}
    body = "".join(f'<div class="insight"><span class="ib" style="--ic:{col[k]}"></span><div>{esc(t)}</div></div>' for k, t in items)
    return f'<div class="glass flat"><h4>Insights from this image</h4>{body}</div>'


def ring_html(pct: float, sub: str, color: str = C_GREEN, empty: bool = False) -> str:
    circ = 339.29
    off = circ * (1 - min(max(pct, 0), 100) / 100)
    centre = ('<div class="kpi-val" style="--k:#8FA3BC"><span style="color:#8FA3BC">-</span></div>' if empty else
              f'<div class="kpi-val" style="--k:{color}"><span class="kpi-num" style="--n:{int(pct)}"></span><span class="kpi-suf">%</span></div>')
    return (f'<div class="ringwrap"><svg viewBox="0 0 140 140" class="ring"><circle class="ring-bg" cx="70" cy="70" r="54"/>'
            f'<circle class="ring-fg" cx="70" cy="70" r="54" style="--c:{circ};--o:{off:.2f};--sc:{color};stroke:{color}"/>'
            f'<circle class="ring-dash" cx="70" cy="70" r="66"/></svg><div class="ring-c"><div>{centre}<small>{sub}</small></div></div></div>')


def ticker_html(info: dict, model_file: str) -> str:
    hist = st.session_state.history
    items = [("Model", esc(model_file)), ("Classes", esc(", ".join(str(v) for v in info["names"].values()))),
             ("Trained at", f"{info['train_imgsz']} px"), ("Analyses this session", str(len(hist)))]
    if hist:
        items.append(("Last run", f"{hist[-1]['seconds']}s in {esc(hist[-1]['mode'])} mode"))
    one = '<span class="sep">◆</span>'.join(f"<span><b>{a}</b> {b}</span>" for a, b in items)
    return f'<div class="ticker"><div class="ticker-track">{one}<span class="sep">◆</span>{one}<span class="sep">◆</span></div></div>'


def mosaic_html(dets: list, cap: int = 360) -> str:
    shown = sorted(dets, key=lambda d: d["id"])[:cap]
    bg = {"empty": "rgba(53,230,155,.16)", "occupied": "rgba(255,82,107,.16)", "unknown": "rgba(255,184,77,.16)"}
    cells = "".join(f'<i title="{d["id"]} · {d["status"]} · {d["confidence"]:.2f}" style="--mc:{HEX_STATUS[d["status"]]};--mb:{bg[d["status"]]};--i:{n}"></i>'
                    for n, d in enumerate(shown))
    more = f'<div class="muted" style="font-size:12.5px;margin-top:8px">+{len(dets) - cap} more slots not drawn</div>' if len(dets) > cap else ""
    return f'<div class="mosaic">{cells}</div>{more}'


def sparkline_html(values: list) -> str:
    if not values:
        return ""
    w, h, pad = 200.0, 60.0, 5.0
    def y(v):
        return pad + (h - 2 * pad) * (1 - min(max(v, 0), 100) / 100)
    n = len(values)
    xs = [w / 2] if n == 1 else [pad + (w - 2 * pad) * i / (n - 1) for i in range(n)]
    pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, values))
    area = f"{xs[0]:.1f},{h - pad:.1f} {pts} {xs[-1]:.1f},{h - pad:.1f}"
    guides = "".join(f'<line x1="0" x2="{w}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="rgba(255,255,255,.18)" stroke-dasharray="3 4"/>' for t in (40, 75))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="3.2" fill="#fff" stroke="#38D9FF" stroke-width="1.5"/>' for x, v in zip(xs, values))
    line = f'<polyline class="line" pathLength="1" points="{pts}"/>' if n > 1 else ""
    return (f'<svg class="spark" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="spg" x1="0" x2="1"><stop offset="0" stop-color="#38D9FF"/><stop offset="1" stop-color="#9B7BFF"/></linearGradient>'
            f'<linearGradient id="spa" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="rgba(56,217,255,.28)"/><stop offset="1" stop-color="rgba(56,217,255,0)"/></linearGradient></defs>'
            f'{guides}{f'<polygon class="area" points="{area}"/>' if n > 1 else ""}{line}{dots}</svg>')


def rowbars_html(rows: list, limit: int = 10) -> str:
    if not rows:
        return '<div class="muted" style="font-size:14px">No classified rows yet.</div>'
    out = []
    for r in rows[:limit]:
        t = max(r["total"], 1)
        out.append(f'<div class="rowbar"><span class="lab">Row {r["row"]}</span><div class="trk">'
                   f'<i style="--w:{r["empty"] / t * 100:.1f}%;background:{C_GREEN}"></i><i style="--w:{r["occupied"] / t * 100:.1f}%;background:{C_RED}"></i></div>'
                   f'<span class="num">{r["empty"]}/{r["total"]}</span></div>')
    more = f'<div class="muted" style="font-size:12.5px">+{len(rows) - limit} more rows</div>' if len(rows) > limit else ""
    return "".join(out) + more


def dashboard_hero(res, stats, info, model_file) -> None:
    user = esc(st.session_state.get("user") or "guest")
    if res is None:
        ring = ring_html(0, "Awaiting image", C_GREEN, empty=True)
        sub = "Upload a parking photo in the Vision Lab to light up this dashboard. Every figure shown here is computed from your own images."
        chips = '<span class="chip ok pulse"><i></i>System ready</span><span class="chip vio"><i></i>No analysis yet</span>'
    else:
        ring = ring_html(stats["availability"], "Spaces available", C_GREEN, empty=stats["total"] == 0)
        sub = (f'Latest scan <b style="color:#fff">{esc(res["name"])}</b> found {stats["total"]} spaces, {stats["empty"]} available. {esc(stats["text"].split(". ")[0])}.'
               if stats["total"] else "The latest scan found no classified spaces. Try Maximum mode or a clearer photo.")
        chip = {"Low": "info", "Moderate": "warn", "High": "bad", "None": "vio"}[stats["level"]]
        chips = (f'<span class="chip ok pulse"><i></i>System ready</span><span class="chip {chip}"><i></i>{stats["level"]} congestion</span>'
                 f'<span class="chip vio"><i></i>{len(st.session_state.history)} analyses</span>')
    H(f'<div class="cmd"><div style="position:relative;z-index:1">{chips}<div class="cmd-title">Welcome back, {user}.</div>'
      f'<div class="cmd-sub">{sub}</div></div>{ring}</div>')


def dashboard_extras(res, stats, dets) -> None:
    H('<div class="sect">Lot at a glance</div>')
    a, b = st.columns([1.5, 1])
    with a:
        H(f'<div class="glass flat"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px"><h4 style="margin:0">Slot mosaic</h4>'
          f'<div class="legend"><span class="chip ok"><i></i>{stats["empty"]} available</span><span class="chip bad"><i></i>{stats["occupied"]} occupied</span></div></div>'
          f'<div class="muted" style="font-size:13px;margin-top:4px">One tile per detected slot, ordered by slot ID. Hover a tile for its ID and confidence.</div>'
          f'{mosaic_html(dets) if dets else "<div class=muted style=margin-top:12px>No slots to draw.</div>"}</div>')
    with b:
        H(insights_html(build_insights(res, stats, dets)))
    H('<div class="sect">Run summary</div>')
    c1, c2, c3 = st.columns(3)
    mean_c = f"{np.mean([d['confidence'] for d in dets]):.2f}" if dets else "-"
    with c1:
        H(f'<div class="glass flat"><h4>This run</h4><div class="mini-grid">'
          f'<div class="mini"><span>Average confidence</span><b>{mean_c}</b></div>'
          f'<div class="mini"><span>Inference passes</span><b>{res["passes"]}</b></div>'
          f'<div class="mini"><span>Processing time</span><b>{res["duration"]}s</b></div>'
          f'<div class="mini"><span>Raw to fused</span><b>{res["raw_total"]} → {res["fused_total"]}</b></div></div></div>')
    hist = st.session_state.history
    with c2:
        occs = [h["occupancy"] for h in hist if h["total"] > 0]
        mean_occ = f"{np.mean(occs):.1f}%" if occs else "-"
        H(f'<div class="glass flat"><h4>Session overview</h4><div class="muted" style="font-size:13px">Utilisation across your analyses, with 40% and 75% guides</div>'
          f'{sparkline_html([h["occupancy"] for h in hist])}'
          f'<div class="mini-grid"><div class="mini"><span>Analyses</span><b>{len(hist)}</b></div><div class="mini"><span>Mean utilisation</span><b>{mean_occ}</b></div></div></div>')
    with c3:
        H(f'<div class="glass flat"><h4>Availability by row</h4><div class="muted" style="font-size:13px;margin-bottom:6px">Rows are estimated from slot positions, top to bottom. Green is free.</div>'
          f'{rowbars_html(row_breakdown(dets))}</div>')


def getting_started() -> None:
    H('<div class="sect">Get started</div>')
    cols = st.columns(3)
    steps = [("1", "Upload a photo", "Use a JPG, PNG or WEBP of a parking area, ideally from above or at a slight angle."),
             ("2", "Let the engine scan", "Several inference passes run on your own model, then the results are fused into one clean set of slots."),
             ("3", "Read the decision", "Get availability, congestion level and a recommendation, then export the evidence.")]
    for col, (n, t, p) in zip(cols, steps):
        with col:
            H(f'<div class="glass"><div class="step"><div class="sn">{n}</div><div><h4>{t}</h4><p>{p}</p></div></div></div>')


def page_dashboard(info, model_file, rcfg):
    H('<div class="page-title">Parking intelligence.</div><div class="page-sub">Turn parking imagery into decisions. Metrics below come from your most recent analysis in this session.</div>')
    res = st.session_state.analysis
    H(ticker_html(info, model_file))
    dashboard_hero(res, compute_stats(active_dets(res, rcfg)) if res is not None else None, info, model_file)
    if res is None:
        empty_state("No analysis yet", "Upload a parking-lot image in the Vision Lab and the real results will appear here.")
        st.button("Open Vision Lab", type="primary", key="dash_open", on_click=nav, args=("Vision Lab",))
        st.markdown("")
        _, right = st.columns([2, 1])
        with right:
            system_panel(info, model_file)
        getting_started()
    else:
        stats = compute_stats(active_dets(res, rcfg))
        H(f'<span class="chip info"><i></i>{esc(res["name"])}</span><span class="chip vio"><i></i>{res["timestamp"]:%d %b %Y, %H:%M}</span>')
        kpi_row(stats)
        st.markdown("")
        left, right = st.columns([2, 1])
        with left:
            congestion_panel(stats)
        with right:
            system_panel(info, model_file)
        dashboard_extras(res, stats, active_dets(res, rcfg))
    hist = st.session_state.history
    H('<div class="sect">Recent analyses</div>')
    if not hist:
        st.caption("Recent analyses will be listed here after you analyse an image.")
    else:
        cols = st.columns(3)
        for i, rec in enumerate(reversed(hist[-6:])):
            with cols[i % 3]:
                chip = {"Low": "info", "Moderate": "warn", "High": "bad", "None": "vio"}[rec["level"]]
                H(f'<div class="glass thumb"><img src="data:image/jpeg;base64,{rec["thumb"]}"/><div><b>{esc(rec["name"][:22])}</b><br>'
                  f'<small>{rec["time"]:%H:%M} · {rec["total"]} slots · {rec["occupancy"]:.1f}% used</small><br>'
                  f'<span class="chip {chip}" style="margin-top:6px"><i></i>{rec["level"]}</span></div></div>')
    H('<div class="sect">Quick actions</div>')
    q1, q2, q3, q4 = st.columns(4)
    for col, (label, page) in zip((q1, q2, q3, q4), [("Analyse an image", "Vision Lab"), ("Open analytics", "Analytics"),
                                                      ("Model performance", "Model Lab"), ("Export results", "Export Centre")]):
        with col:
            st.button(label, key=f"qa_{page}", on_click=nav, args=(page,))


# =====================================================================================
# PAGES: VISION LAB
# =====================================================================================
def scan_html(img_b64, stage_idx, message, frac):
    items = []
    for i, s in enumerate(STAGES):
        cls = "done" if i < stage_idx else ("now" if i == stage_idx else "")
        items.append(f'<div class="stage {cls}"><span class="sd"></span>{s}</div>')
    hexsvg = ('<svg class="hexspin" viewBox="0 0 48 48"><path d="M24 3 43 14v20L24 45 5 34V14z" fill="none" stroke="#38D9FF" stroke-width="2.4" stroke-dasharray="10 6"/>'
              '<path d="M24 12 35 18v12L24 36 13 30V18z" fill="none" stroke="#9B7BFF" stroke-width="2"/></svg>')
    return (f'<div class="glass scanbox flat"><div class="scan-img"><img src="data:image/jpeg;base64,{img_b64}"/><div class="scan-line"></div></div>'
            f'<div><div style="display:flex;gap:14px;align-items:center">{hexsvg}<div><h4 style="margin:0">Analysing your parking image</h4>'
            f'<span class="muted" style="font-size:13px">Vision engine running on your model</span></div></div>'
            f'<div style="margin-top:14px">{"".join(items)}</div>'
            f'<div class="pbar"><div style="width:{int(frac * 100)}%"></div></div>'
            f'<div class="muted" style="font-size:13px">{message}</div></div></div>')


def decode_upload(data: bytes):
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.array(img)


def quality_tiles(q):
    side = max(q["width"], q["height"])
    items = [
        ("Resolution", f"{q['width']} x {q['height']}", "ok" if side >= 800 else "warn"),
        ("Sharpness", f"{q['sharpness']:.0f}", "ok" if q["sharpness"] >= 80 else "warn"),
        ("Brightness", f"{q['brightness']:.0f} / 255", "ok" if 75 <= q["brightness"] <= 205 else "warn"),
        ("Contrast", f"{q['contrast']:.0f}", "ok" if q["contrast"] >= 42 else "warn"),
    ]
    return "".join(f'<span class="chip {c}"><i></i>{a}: {b}</span>' for a, b, c in items)


def page_lab(model, info, dcfg, rcfg, ui):
    H('<div class="page-title">AI Vision Laboratory</div><div class="page-sub">Upload a parking-lot photo. Every box below is a real model detection, fused across several inference passes.</div>')
    data, fname = None, ""
    up = st.file_uploader("Drop a parking image here", type=["jpg", "jpeg", "png", "webp", "bmp"], key="lab_up", label_visibility="collapsed")
    if up is not None:
        data, fname = up.getvalue(), up.name
    if st.toggle("Take a photo with the camera instead", value=False, key="lab_usecam"):
        cam = st.camera_input("Take a photo of a parking area", key="lab_cam")
        if cam is not None:
            data, fname = cam.getvalue(), "camera_capture.jpg"
    cur = st.session_state.analysis
    if data is None:
        if cur is None:
            empty_state("Drop an image to begin", "JPG, PNG, WEBP or BMP. Photos taken from higher up with the bays clearly visible work best.")
            return
        st.info("Showing your last analysis. Upload another image to run a new one.")
    else:
        try:
            rgb = decode_upload(data)
        except Exception as exc:
            st.error("This file could not be read as an image. Upload a valid JPG, PNG, WEBP or BMP.")
            st.caption(f"Details: {exc}")
            return
        fhash = hashlib.md5(data).hexdigest()
        key = fhash + "|" + json.dumps(dcfg, sort_keys=True)
        run_clicked = st.button("Run analysis", type="primary", key="run_btn")
        need_first = cur is None or cur["file_hash"] != fhash
        if run_clicked or (ui["auto_run"] and need_first):
            slot = st.empty()
            thumb = b64_jpeg(rgb, 900)

            def cb(stage, msg, frac):
                slot.markdown(_clean(scan_html(thumb, stage, msg, frac)), unsafe_allow_html=True)

            try:
                cb(0, "Starting", 0.01)
                res = run_analysis(rgb, fname, fhash, model, info, dcfg, cb)
                res["key"] = key
            except Exception as exc:
                slot.empty()
                st.error("The analysis failed. Check that the weights are a valid YOLO detection model, or try Fast mode.")
                st.exception(exc)
                return
            slot.empty()
            st.session_state.analysis = res
            try:
                st.toast(f"Analysis complete: {len(res['dets'])} detections in {res['duration']}s", icon="✅")
            except Exception:
                pass
            stats0 = compute_stats(active_dets(res, rcfg))
            thumb_img = render_annotated(res["rgb"], active_dets(res, rcfg), {**rcfg, "ids": False, "conf_lbl": False})
            st.session_state.history.append({
                "name": fname, "time": res["timestamp"], "total": stats0["total"], "occupied": stats0["occupied"],
                "empty": stats0["empty"], "occupancy": stats0["occupancy"], "level": stats0["level"],
                "mode": dcfg["mode"], "seconds": res["duration"], "thumb": b64_jpeg(thumb_img, 240, 70),
            })
            st.session_state.history = st.session_state.history[-30:]
            cur = res
        elif cur is not None and cur["file_hash"] == fhash and cur.get("key") != key:
            st.warning("Detection settings changed since this result. Press Run analysis to apply them (overlay settings update instantly).")
    res = st.session_state.analysis
    if res is None:
        return

    dets = active_dets(res, rcfg)
    stats = compute_stats(dets)
    H(f'<div class="sect" style="margin-top:8px">Result for {esc(res["name"])}</div>')
    kpi_row(stats)
    H(f'<div style="margin-top:12px">{quality_tiles(res["quality"])}'
      f'<span class="chip vio"><i></i>{res["passes"]} passes · {res["duration"]}s · {dcfg["mode"] if res["dcfg"]["mode"] == dcfg["mode"] else res["dcfg"]["mode"]} mode</span></div>')
    if stats["total"] == 0:
        st.warning("No parking spaces were detected. Try Maximum mode, lower the sensitivity, or use a clearer, wider image.")
    if stats["unknown"]:
        st.warning(f"{stats['unknown']} detections have an unrecognised class and are excluded from occupancy.")

    ids = ["None"] + [d["id"] for d in dets]
    skey = f"lab_sel_{res['timestamp']:%H%M%S%f}"
    if st.session_state.get(skey) not in ids:
        st.session_state[skey] = "None"
    sel = st.selectbox("Slot inspector: highlight a slot", ids, key=skey)
    highlight = None if sel == "None" else sel
    annotated = render_annotated(res["rgb"], dets, rcfg, highlight)
    view = st.radio("View", ["Annotated", "Side by side", "Original"], horizontal=True, key="lab_view", label_visibility="collapsed")
    if view == "Annotated":
        show_image(annotated, caption="Green = available, red = occupied")
    elif view == "Original":
        show_image(res["rgb"])
    else:
        a, b = st.columns(2)
        with a:
            st.caption("Original"); show_image(res["rgb"])
        with b:
            st.caption("Detections"); show_image(annotated)

    L, R = st.columns([1.5, 1])
    with L:
        congestion_panel(stats)
    with R:
        if highlight:
            d = next(x for x in dets if x["id"] == highlight)
            show_image(crop_slot(res["rgb"], d), caption=f"{d['id']} close-up")
            chip = {"empty": "ok", "occupied": "bad", "unknown": "warn"}[d["status"]]
            H(f'<div class="glass flat"><span class="chip {chip}"><i></i>{d["status"].title()}</span>'
              f'<div class="muted" style="font-size:14px;line-height:1.8">Confidence <b style="color:#fff">{d["confidence"]:.2f}</b> · '
              f'passes agreeing <b style="color:#fff">{d["votes"]}</b> · class agreement <b style="color:#fff">{d["agreement"]:.0%}</b><br>'
              f'Box {d["w"]} x {d["h"]} px at ({d["x1"]}, {d["y1"]})</div></div>')
        else:
            H('<div class="glass flat"><h4>Slot inspector</h4><p class="muted" style="font-size:14px">Pick a slot ID above to highlight it on the image, see a close-up and its confidence. '
              'Confidence is the model score (not a calibrated probability).</p></div>')

    H(insights_html(build_insights(res, stats, dets)))
    with st.expander("Slot table and processing details"):
        if dets:
            df = pd.DataFrame(dets)[["id", "status", "confidence", "votes", "agreement", "x1", "y1", "x2", "y2", "w", "h"]]
            show_df(df)
        st.markdown("**Preprocessing:** " + ("; ".join(res["steps"]) if res["steps"] else "none needed"))
        st.dataframe(pd.DataFrame(res["log"]), hide_index=True)
        st.caption(f"{res['raw_total']} raw detections -> {res['fused_total']} after fusion; removed {res['dropped_consensus']} weak single-pass and {res['dropped_geo']} size outliers.")

    with st.expander("Result looks wrong? Flag it as a failure example"):
        note = st.text_input("What went wrong? (for example: shadowed bays missed)", key="flag_note")
        if st.button("Save as failure example", key="flag_btn"):
            st.session_state.failures.append({"name": res["name"], "note": note or "No note", "time": datetime.now(),
                                              "thumb": b64_jpeg(annotated, 420, 75), "counts": f"{stats['occupied']} occupied / {stats['empty']} available"})
            st.success("Saved. It will appear in the Model Lab.")


# =====================================================================================
# PAGES: ANALYTICS
# =====================================================================================
def page_analytics(rcfg):
    H('<div class="page-title">Analytics Studio</div><div class="page-sub">Charts use only analyses you have run in this session. Nothing is simulated.</div>')
    res, hist = st.session_state.analysis, st.session_state.history
    if res is None and not hist:
        empty_state("Nothing to chart yet", "Analyse at least one image in the Vision Lab to unlock these charts.")
        return
    if res is not None:
        dets = active_dets(res, rcfg)
        stats = compute_stats(dets)
        c1, c2 = st.columns([1, 1.6])
        with c1:
            H('<div class="sect" style="margin-top:0">Current image</div>')
            show_plot(gauge_fig(stats["occupancy"], stats["level"]), "g_gauge")
            fig = go.Figure(go.Pie(labels=["Available", "Occupied"], values=[stats["empty"], stats["occupied"]], hole=.62,
                                   marker=dict(colors=[C_GREEN, C_RED]), textinfo="label+percent"))
            show_plot(style_fig(fig, 260), "g_pie")
        with c2:
            if dets:
                fig = go.Figure()
                for s in ("empty", "occupied"):
                    pts = [d for d in dets if d["status"] == s]
                    if pts:
                        fig.add_trace(go.Scatter(x=[(d["x1"] + d["x2"]) / 2 for d in pts], y=[(d["y1"] + d["y2"]) / 2 for d in pts], mode="markers",
                                                 name="Available" if s == "empty" else "Occupied", text=[d["id"] for d in pts],
                                                 marker=dict(symbol="square", size=11, color=HEX_STATUS[s], line=dict(color="white", width=1), opacity=.9)))
                fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
                fig.update_layout(title="Slot map (centre of each detected slot)")
                show_plot(style_fig(fig, 420), "g_map")
        d1, d2 = st.columns(2)
        with d1:
            fig = go.Figure()
            for s in ("empty", "occupied"):
                vals = [d["confidence"] for d in dets if d["status"] == s]
                if vals:
                    fig.add_trace(go.Histogram(x=vals, name="Available" if s == "empty" else "Occupied", marker_color=HEX_STATUS[s], opacity=.75, xbins=dict(size=.05)))
            fig.update_layout(barmode="overlay", title="Model confidence distribution")
            show_plot(style_fig(fig, 300), "g_hist")
        with d2:
            if dets:
                fig = go.Figure(go.Scatter(x=[d["w"] for d in dets], y=[d["h"] for d in dets], mode="markers",
                                           marker=dict(color=[HEX_STATUS[d["status"]] for d in dets], size=8, opacity=.8)))
                fig.update_layout(title="Box width vs height (px)", xaxis_title="width", yaxis_title="height")
                show_plot(style_fig(fig, 300), "g_wh")
        rows = row_breakdown(dets)
        if rows:
            fig = go.Figure()
            labels = [f"Row {r['row']}" for r in rows]
            fig.add_trace(go.Bar(y=labels, x=[r["empty"] for r in rows], name="Available", orientation="h", marker_color=C_GREEN))
            fig.add_trace(go.Bar(y=labels, x=[r["occupied"] for r in rows], name="Occupied", orientation="h", marker_color=C_RED))
            fig.update_layout(barmode="stack", title="Availability by row (top to bottom)")
            fig.update_yaxes(autorange="reversed")
            show_plot(style_fig(fig, max(240, 34 * len(rows) + 120)), "g_rows")
    if hist:
        H('<div class="sect">Across analysed images</div>')
        a, b = st.columns([1.6, 1])
        with a:
            fig = go.Figure(go.Bar(x=[f"#{i + 1} {h['name'][:14]}" for i, h in enumerate(hist)], y=[h["occupancy"] for h in hist],
                                   marker_color=[LEVEL_COLOR[h["level"]] for h in hist], text=[f"{h['occupancy']:.0f}%" for h in hist], textposition="outside"))
            fig.add_hline(y=40, line_dash="dot", line_color="rgba(255,255,255,.35)", annotation_text="40%")
            fig.add_hline(y=75, line_dash="dot", line_color="rgba(255,255,255,.35)", annotation_text="75%")
            fig.update_layout(title="Utilisation by analysis", yaxis=dict(range=[0, 110], title="Occupancy %"))
            show_plot(style_fig(fig, 340), "g_hist_bar")
        with b:
            counts = pd.Series([h["level"] for h in hist]).value_counts()
            fig = go.Figure(go.Pie(labels=list(counts.index), values=list(counts.values), hole=.6, marker=dict(colors=[LEVEL_COLOR[k] for k in counts.index])))
            fig.update_layout(title="Congestion distribution")
            show_plot(style_fig(fig, 340), "g_levels")
        tbl = pd.DataFrame([{"time": f"{h['time']:%H:%M:%S}", "image": h["name"], "slots": h["total"], "occupied": h["occupied"], "available": h["empty"],
                             "occupancy %": round(h["occupancy"], 1), "congestion": h["level"], "mode": h["mode"], "seconds": h["seconds"]} for h in hist])
        show_df(tbl)
        if len(hist) < 2:
            st.caption("Analyse more images to compare them. Trends appear only from analyses recorded in this session.")


# =====================================================================================
# PAGES: MODEL LAB
# =====================================================================================
def page_model(model, info, model_file):
    H('<div class="page-title">Model Performance Lab</div><div class="page-sub">Only metrics that genuinely exist are shown: those stored inside your weights file, and anything you evaluate or upload here.</div>')
    m = info["train_metrics"]
    if m:
        P, R = m.get("metrics/precision(B)"), m.get("metrics/recall(B)")
        f1 = (2 * P * R / (P + R)) if (P and R) else None
        c = st.columns(5)
        vals = [("Precision", P), ("Recall", R), ("F1 (from P and R)", f1), ("mAP@50", m.get("metrics/mAP50(B)")), ("mAP@50-95", m.get("metrics/mAP50-95(B)"))]
        for col, (lab, v), colr in zip(c, vals, [C_CYAN, C_GREEN, C_VIOLET, C_CYAN, C_AMBER]):
            with col:
                H(kpi(lab, (v or 0) * 100, "Stored in the checkpoint", colr, "%", 1) if v is not None else kpi(lab, 0, "Not available", colr))
        st.caption("These are the validation-set metrics Ultralytics saved at the best epoch during training. They are not a separate held-out test-set evaluation; "
                   "use the validation tool below to generate one.")
    else:
        st.info("This weights file does not contain stored training metrics. Run the validation tool below or upload results.csv.")

    left, right = st.columns([1.2, 1])
    with left:
        ta = info["train_args"]
        rows = {"Model file": f"{model_file} ({info['size_mb']:.1f} MB)", "Architecture": str(ta.get("model", "n/a")), "Task": info["task"],
                "Classes": ", ".join(f"{k}: {v}" for k, v in info["names"].items()), "Training image size": f"{info['train_imgsz']} px",
                "Epochs configured": ta.get("epochs", "n/a"), "Batch size": ta.get("batch", "n/a"), "Optimizer": ta.get("optimizer", "n/a"),
                "Best epoch (0-indexed)": info["epoch"] if info["epoch"] is not None else "n/a", "Trained with Ultralytics": info["version"] or "n/a",
                "Saved": info["date"] or "n/a", "Dataset config (training machine)": ta.get("data", "n/a")}
        body = "".join(f'<div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px"><span class="muted">{k}</span>'
                       f'<span style="text-align:right;color:#fff;word-break:break-all">{v}</span></div>' for k, v in rows.items())
        H(f'<div class="glass flat"><h4>Training record from the checkpoint</h4>{body}</div>')
    with right:
        H('''<div class="glass flat"><h4>Known limitations</h4><div class="muted" style="font-size:14px;line-height:1.75">
        The weights are a small YOLOv8n trained for a short run, so very different cameras, night scenes or heavy occlusion can lower accuracy.<br>
        A bay hidden behind another car or a tree may not be detected at all.<br>
        A free bay marked occupied (false positive for occupancy) makes the lot look busier than it is; the reverse sends drivers to a full area.<br>
        Results describe the uploaded photo, not live availability.<br>
        Tile and mirror passes improve recall but cost time on CPU.</div></div>''')

    H('<div class="sect">Evaluate on your own labelled split</div>')
    st.caption("Works when running locally with your dataset available. Enter the path to your data.yaml, choose the split, and run validation with the loaded model.")
    v1, v2, v3 = st.columns([3, 1, 1])
    with v1:
        yaml_path = st.text_input("data.yaml path", placeholder=r"C:\Users\you\parkvision\data.yaml", key="val_yaml")
    with v2:
        split = st.selectbox("Split", ["test", "val"], key="val_split")
    with v3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go_val = st.button("Run validation", key="val_go")
    if go_val:
        if not yaml_path or not Path(yaml_path).is_file():
            st.error("data.yaml was not found at that path (it must be reachable from the machine running this app).")
        else:
            try:
                with st.spinner("Validating on your dataset…"):
                    with _INFER_LOCK:
                        mt = model.val(data=yaml_path, split=split, imgsz=info["train_imgsz"], plots=False, verbose=False)
                st.session_state["val_result"] = {
                    "p": float(mt.box.mp), "r": float(mt.box.mr), "m50": float(mt.box.map50), "m5095": float(mt.box.map),
                    "cm": np.array(mt.confusion_matrix.matrix).tolist() if getattr(mt, "confusion_matrix", None) is not None else None,
                    "names": dict(mt.names), "split": split}
            except Exception as exc:
                st.error("Validation could not run. Check the dataset paths inside data.yaml and that the chosen split exists.")
                st.exception(exc)
    vr = st.session_state.get("val_result")
    if vr:
        f1 = 2 * vr["p"] * vr["r"] / max(vr["p"] + vr["r"], 1e-9)
        c = st.columns(5)
        for col, (lab, v, colr) in zip(c, [("Precision", vr["p"], C_CYAN), ("Recall", vr["r"], C_GREEN), ("F1", f1, C_VIOLET), ("mAP@50", vr["m50"], C_CYAN), ("mAP@50-95", vr["m5095"], C_AMBER)]):
            with col:
                H(kpi(lab, v * 100, f"Your '{vr['split']}' split", colr, "%", 1))
        if vr["cm"] is not None:
            labels = [vr["names"].get(i, str(i)) for i in range(len(vr["names"]))] + ["background"]
            cm = np.array(vr["cm"])
            fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#0C1425"], [1, "#38D9FF"]], text=cm.astype(int), texttemplate="%{text}", showscale=False))
            fig.update_layout(title="Confusion matrix (columns = true, rows = predicted)", xaxis_title="True", yaxis_title="Predicted")
            fig.update_yaxes(autorange="reversed")
            show_plot(style_fig(fig, 380), "g_cm")

    H('<div class="sect">Training curves and confusion matrix from your training run</div>')
    u1, u2 = st.columns(2)
    with u1:
        csv = st.file_uploader("Upload results.csv from your training run", type=["csv"], key="up_results")
        if csv is not None:
            try:
                df = pd.read_csv(csv)
                df.columns = [c.strip() for c in df.columns]
                fig = go.Figure()
                for col, colr in [("metrics/precision(B)", C_CYAN), ("metrics/recall(B)", C_GREEN), ("metrics/mAP50(B)", C_VIOLET), ("metrics/mAP50-95(B)", C_AMBER)]:
                    if col in df.columns:
                        fig.add_trace(go.Scatter(x=df.get("epoch", df.index), y=df[col], name=col.split("/")[1], line=dict(color=colr, width=2.5)))
                fig.update_layout(title="Validation metrics per epoch")
                show_plot(style_fig(fig, 320), "g_curves")
                loss = [c for c in df.columns if "loss" in c]
                if loss:
                    fig2 = go.Figure([go.Scatter(x=df.get("epoch", df.index), y=df[c], name=c) for c in loss])
                    fig2.update_layout(title="Loss per epoch")
                    show_plot(style_fig(fig2, 300), "g_loss")
            except Exception as exc:
                st.error("That CSV could not be parsed.")
                st.caption(str(exc))
    with u2:
        img = st.file_uploader("Upload confusion_matrix.png from your training run", type=["png", "jpg", "jpeg"], key="up_cm")
        if img is not None:
            show_image(Image.open(img), caption="Confusion matrix from your training run")

    H('<div class="sect">Failure examples</div>')
    fails = st.session_state.failures
    if not fails:
        st.caption("Flag a wrong result in the Vision Lab and it will be listed here with your note.")
    else:
        cols = st.columns(3)
        for i, f in enumerate(fails):
            with cols[i % 3]:
                H(f'<div class="glass flat"><img style="width:100%;border-radius:12px" src="data:image/jpeg;base64,{f["thumb"]}"/>'
                  f'<div style="margin-top:10px"><b>{esc(f["name"][:26])}</b><br><span class="muted" style="font-size:13px">{f["counts"]}</span><br>'
                  f'<span style="font-size:14px">{esc(f["note"])}</span></div></div>')


# =====================================================================================
# PAGES: EXPORT
# =====================================================================================
def page_export(rcfg, model_file):
    H('<div class="page-title">Export Centre</div><div class="page-sub">Everything here is generated from your real analysis, including the current overlay and display threshold.</div>')
    res = st.session_state.analysis
    if res is None:
        empty_state("Nothing to export yet", "Run an analysis in the Vision Lab first.")
        return
    dets = active_dets(res, rcfg)
    stats = compute_stats(dets)
    annotated = render_annotated(res["rgb"], dets, rcfg)
    stem = Path(res["name"]).stem or "parkvision"
    stamp = f"{res['timestamp']:%Y%m%d_%H%M%S}"
    kpi_row(stats)
    st.markdown("")
    a, b = st.columns([1.5, 1])
    with a:
        show_image(annotated, caption="Annotated image that will be exported")
    with b:
        H(f'<div class="glass flat"><h4>Downloads</h4><p class="muted" style="font-size:14px">{stats["text"]}</p></div>')
        _stretch(st.download_button, "Annotated image (PNG)", png_bytes(annotated), file_name=f"{stem}_annotated_{stamp}.png", mime="image/png", key="dl_png")
        _stretch(st.download_button, "Slot table (CSV)", pd.DataFrame(dets).to_csv(index=False).encode("utf-8"), file_name=f"{stem}_slots_{stamp}.csv", mime="text/csv", key="dl_csv")
        _stretch(st.download_button, "Report (Markdown)", build_report_md(res, stats, rcfg, model_file).encode("utf-8"), file_name=f"{stem}_report_{stamp}.md", mime="text/markdown", key="dl_md")
        _stretch(st.download_button, "Report (JSON)", json.dumps(build_report_json(res, stats, rcfg), indent=2, default=str).encode("utf-8"), file_name=f"{stem}_report_{stamp}.json", mime="application/json", key="dl_json")
        _stretch(st.download_button, "Everything (ZIP)", build_zip(res, stats, rcfg, annotated, model_file), file_name=f"{stem}_parkvision_{stamp}.zip", mime="application/zip", key="dl_zip", type="primary")
        st.markdown("**Shareable summary**")
        st.code(f"ParkVision AI | {res['name']} | {stats['total']} slots: {stats['empty']} available, {stats['occupied']} occupied "
                f"({stats['occupancy']:.1f}% utilisation, {stats['level']} congestion). {stats['text']}", language=None)


# =====================================================================================
# MAIN
# =====================================================================================
def page_missing_model():
    H('<div class="glass flat"><h3>Model weights not found</h3><p class="muted">Place your trained <code>best.pt</code> in a <code>models</code> folder next to <code>app.py</code> '
      '(<code>models/best.pt</code>), or upload it below.</p></div>')
    up = st.file_uploader("Upload best.pt", type=["pt"], key="missing_up")
    if up is not None:
        data = up.getvalue()
        h = hashlib.md5(data).hexdigest()
        path = Path(tempfile.gettempdir()) / f"pv_model_{h[:10]}.pt"
        path.write_bytes(data)
        st.session_state.custom_model_path, st.session_state.custom_model_hash = str(path), h
        st.rerun()


def main():
    init_state()
    if not st.session_state.authed:
        inject_css("Midnight", False, True, hide_sidebar=True)
        page_welcome() if st.session_state.stage == "welcome" else page_login()
        return

    dcfg, rcfg, ui = sidebar_ui()
    inject_css(ui["theme"], ui["reduced"], ui["effects"])

    if YOLO is None:
        st.error("The detection library (ultralytics) could not be imported. Check requirements.txt.")
        st.exception(YOLO_IMPORT_ERROR)
        return
    path = resolve_model_path()
    if path is None:
        page_missing_model()
        return
    try:
        model, info = load_model(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.error("The YOLO model could not be loaded. Make sure the file is a valid Ultralytics detection checkpoint.")
        st.exception(exc)
        return

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard(info, path.name, rcfg)
    elif page == "Vision Lab":
        page_lab(model, info, dcfg, rcfg, ui)
    elif page == "Analytics":
        page_analytics(rcfg)
    elif page == "Model Lab":
        page_model(model, info, path.name)
    else:
        page_export(rcfg, path.name)
    st.markdown("")
    st.caption("ParkVision AI estimates availability from the uploaded image only. It is not a live parking feed.")


if os.environ.get("PARKVISION_TEST") != "1":
    main()