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

NEW: "ParkVision Reserve" page (see the PARKVISION RESERVE section near the end of this file):
smart reservations with a BookMyShow-style slot map, vacancy alerts driven by the existing
detector, and a configurable live camera / device layer. Everything above that section is the
original application; the few lines touched there are marked with  # [RESERVE].
"""
from __future__ import annotations

import atexit  # [RESERVE]
import base64
import hashlib
import hmac
import html as _html
import io
import json
import logging  # [RESERVE]
import os
import random
import re  # [RESERVE]
import secrets  # [RESERVE]
import sqlite3  # [RESERVE]
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager  # [RESERVE]
from datetime import datetime, timedelta, timezone  # [RESERVE]  (timedelta / timezone added)
from functools import lru_cache  # [RESERVE]
from pathlib import Path
from urllib.parse import urlparse  # [RESERVE]

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components  # [RESERVE]
from PIL import Image, ImageDraw, ImageFont, ImageOps  # [RESERVE]  (ImageDraw / ImageFont added)

try:  # [RESERVE] time zones (tzdata is listed in requirements.txt for Windows)
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

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
    ("ParkVision Reserve", "◈   ParkVision Reserve"),  # [RESERVE]
    ("Export Centre", "⇩   Export Centre"),
]
STAGES = ["Preparing image", "Analysing parking spaces", "Merging detections", "Generating insights"]


@st.cache_resource  # [RESERVE] one lock for the whole server process, so the new camera monitor
def _shared_infer_lock():  # [RESERVE]  and the Vision Lab never call model.predict() at the same time
    return threading.Lock()


_INFER_LOCK = _shared_infer_lock()  # [RESERVE] (was: threading.Lock())

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
    try:  # [RESERVE] release the camera / monitor if this user owns them
        rv_release_for_user(st.session_state.get("user", ""))
    except Exception:
        pass
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
    poly = f'<polygon class="area" points="{area}"/>' if n > 1 else ""  # [RESERVE] (pre-computed: nested same-quote f-strings need Python 3.12)
    return (f'<svg class="spark" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none"><defs>'
            f'<linearGradient id="spg" x1="0" x2="1"><stop offset="0" stop-color="#38D9FF"/><stop offset="1" stop-color="#9B7BFF"/></linearGradient>'
            f'<linearGradient id="spa" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="rgba(56,217,255,.28)"/><stop offset="1" stop-color="rgba(56,217,255,0)"/></linearGradient></defs>'
            f'{guides}{poly}{line}{dots}</svg>')


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
# PARKVISION RESERVE  (additive section: reservations, vacancy alerts, camera hub)
# -------------------------------------------------------------------------------------
# Nothing above this line was changed in behaviour. This section only READS the existing
# detector (run_analysis / assign_ids / render_annotated) and never alters it.
# Three separate ideas are kept apart on purpose:
#   1. AI-observed occupancy  -> what the camera sees right now (live) or demo data
#   2. Reservation status     -> rows in SQLite (bookings / holds)
#   3. Bookable availability  -> slot rules + (1) only when arrival is imminent + (2)
# =====================================================================================
log = logging.getLogger("parkvision.reserve")

CURRENCY = "₹"                                   # change to "$", "€" ... if you prefer
DEFAULT_TZ = os.environ.get("PARKVISION_TZ", "Asia/Kolkata")
TZ_CHOICES = ["Asia/Kolkata", "UTC", "Europe/London", "Europe/Paris", "America/New_York",
              "America/Los_Angeles", "Asia/Dubai", "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney"]
_TZ_FIXED_MIN = {"Asia/Kolkata": 330, "UTC": 0, "Europe/London": 0, "Europe/Paris": 60, "America/New_York": -300,
                 "America/Los_Angeles": -480, "Asia/Dubai": 240, "Asia/Singapore": 480, "Asia/Tokyo": 540,
                 "Australia/Sydney": 600}  # only used if the tz database is unavailable

BOOKING_WINDOW_H = 24        # arrival must be within this many hours from now
GRACE_SEC = 120              # tolerated latency between choosing a time and confirming
HOLD_SECONDS = 300           # a selected slot is held (amber for others) for 5 minutes
MIN_DURATION_MIN, MAX_DURATION_MIN = 30, 720
DURATIONS = [30, 60, 90, 120, 180, 240, 360, 480, 600, 720]
NOW_BLOCK_MIN = 30           # arrival this close to "now": a slot observed occupied cannot be booked
LIVE_FRESH_SEC = 45          # a camera observation older than this is not used as "live"
STALE_FRAME_SEC = 8          # camera frames older than this count as "no signal"
ARM_OBS = 2                  # consecutive "occupied" observations before a slot can raise a vacancy alert
DISCREPANCY_OBS = 3          # consecutive occupied observations on a reserved, un-checked-in slot
TAX_RATE = 0.05
CANCEL_FREE_MIN = 30
WALK_SPEED_MPS = 1.3

CATEGORIES = ["Shopping malls", "Grocery stores", "Supermarkets", "Office complexes", "Hospitals",
              "Airports and railway stations", "Other public parking"]
VEHICLES = {"Car": 1.0, "EV car": 1.0, "Motorbike": 0.5, "Accessible vehicle": 0.8}
SLOT_TYPES = {"standard": "Standard bay", "covered": "Covered bay", "ev": "EV charging bay", "accessible": "Accessible bay"}
ELIGIBLE = {"Car": {"standard", "covered"}, "EV car": {"standard", "covered", "ev"},
            "Motorbike": {"standard", "covered"}, "Accessible vehicle": {"standard", "covered", "accessible"}}
SURCHARGE_PER_H = {"standard": 0, "covered": 10, "ev": 25, "accessible": 0}   # prototype rates (not real venue pricing)
FACILITIES = {"standard": ["CCTV coverage"], "covered": ["Roof cover", "CCTV coverage"],
              "ev": ["EV charger (7 kW, demo)", "CCTV coverage"], "accessible": ["Wider bay", "Step-free route", "CCTV coverage"]}
SLOT_COLORS = {"green": C_GREEN, "red": C_RED, "cyan": C_CYAN, "grey": "#8FA3BC", "amber": C_AMBER}

# Demo destinations. All names, addresses, capacities and rates are SIMULATED (not real venues).
LOTS = [
    dict(id="ORB", name="Orbit Central Mall", cat="Shopping malls", addr="12 Sample Avenue, Demo City", rows=6, cols=14, rate=60, fee=15, cap=420, acc=2, ev=3, covered_rows=(0, 1)),
    dict(id="SKY", name="Skyline Galleria", cat="Shopping malls", addr="48 Example Road, Demo City", rows=4, cols=12, rate=80, fee=15, cap=560, acc=2, ev=2, covered_rows=(0, 1, 2, 3)),
    dict(id="FRB", name="FreshBasket Grocery", cat="Grocery stores", addr="5 Market Lane, Demo City", rows=4, cols=10, rate=20, fee=5, cap=100, acc=2, ev=1, covered_rows=()),
    dict(id="MGM", name="MegaMart Supermarket", cat="Supermarkets", addr="90 Sample Highway, Demo City", rows=6, cols=12, rate=30, fee=10, cap=180, acc=2, ev=2, covered_rows=(0,)),
    dict(id="VTX", name="Vertex Tech Park", cat="Office complexes", addr="Block C, Demo Business Park", rows=8, cols=14, rate=50, fee=10, cap=400, acc=2, ev=4, covered_rows=(0, 1, 2, 3), restricted=True),
    dict(id="MER", name="Meridian Towers", cat="Office complexes", addr="1 Example Plaza, Demo City", rows=6, cols=12, rate=70, fee=10, cap=500, acc=2, ev=3, covered_rows=(0, 1, 2, 3, 4, 5), restricted=True),
    dict(id="AUR", name="St. Aurelia Hospital", cat="Hospitals", addr="22 Care Street, Demo City", rows=6, cols=10, rate=40, fee=5, cap=300, acc=4, ev=2, covered_rows=(0, 1)),
    dict(id="HCA", name="Harbor City Airport - P2", cat="Airports and railway stations", addr="Terminal Road, Demo City", rows=8, cols=14, rate=120, fee=20, cap=900, acc=3, ev=4, covered_rows=(0, 1, 2, 3)),
    dict(id="CJN", name="Central Junction Railway Station", cat="Airports and railway stations", addr="Station Approach, Demo City", rows=6, cols=12, rate=40, fee=10, cap=320, acc=2, ev=2, covered_rows=(0,)),
    dict(id="RIV", name="Riverside Public Garage", cat="Other public parking", addr="7 Riverside Walk, Demo City", rows=4, cols=12, rate=25, fee=5, cap=250, acc=2, ev=2, covered_rows=(0, 1, 2, 3)),
    dict(id="LIVE-CAM", name="Your Camera Lot (live)", cat="Other public parking", addr="Whatever your connected camera is looking at",
         rate=30, fee=5, cap=300, live_only=True),
]
LOT_BY_ID = {lot["id"]: lot for lot in LOTS}


class BookingError(Exception):
    """A reservation rule was violated (shown to the user as-is)."""


# =====================================================================================
# TIME + MONEY HELPERS
# =====================================================================================
def get_tz(name: str):
    try:
        if ZoneInfo is not None:
            return ZoneInfo(name)
    except Exception:
        pass
    return timezone(timedelta(minutes=_TZ_FIXED_MIN.get(name, 0)))


def rv_tzname() -> str:
    try:
        return st.session_state.get("rv_tz") or DEFAULT_TZ
    except Exception:
        return DEFAULT_TZ


def now_ts() -> int:
    return int(time.time())


def local_now() -> datetime:
    return datetime.now(get_tz(rv_tzname()))


def fmt_clock(ts, fmt="%H:%M") -> str:
    return datetime.fromtimestamp(int(ts), get_tz(rv_tzname())).strftime(fmt)


def fmt_dt(ts) -> str:
    return datetime.fromtimestamp(int(ts), get_tz(rv_tzname())).strftime("%a %d %b, %H:%M")


def local_to_ts(d, hh: int, mm: int) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=get_tz(rv_tzname())).timestamp())


def fmt_dur(mins: int) -> str:
    h, m = divmod(int(mins), 60)
    return f"{h} h {m} min" if h and m else (f"{h} h" if h else f"{m} min")


def money(cents) -> str:
    return f"{CURRENCY}{int(cents) / 100:,.2f}"


def arrival_options(d, now: int) -> list:
    """Valid arrival timestamps (15-min steps) on local date d, inside [next quarter hour, now + 24 h]."""
    earliest = ((now + 899) // 900) * 900
    latest = now + BOOKING_WINDOW_H * 3600
    out = []
    for hh in range(24):
        for mm in (0, 15, 30, 45):
            t = local_to_ts(d, hh, mm)
            if earliest <= t <= latest:
                out.append(t)
    return out


def validate_window(start: int, end: int, now: int) -> list:
    errs = []
    if start < now - GRACE_SEC:
        errs.append("The arrival time has already passed. Pick a later time.")
    if start > now + BOOKING_WINDOW_H * 3600:
        errs.append(f"Reservations can only be made up to {BOOKING_WINDOW_H} hours ahead.")
    mins = (end - start) / 60
    if mins < MIN_DURATION_MIN or mins > MAX_DURATION_MIN:
        errs.append(f"Duration must be between {fmt_dur(MIN_DURATION_MIN)} and {fmt_dur(MAX_DURATION_MIN)}.")
    return errs


# =====================================================================================
# URL / TEXT REDACTION  (camera credentials must never reach the screen, logs or DB)
# =====================================================================================
_URL_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*)://([^\s/?#]*)([^\s]*)")


def redact_text(s) -> str:
    def _sub(m):
        scheme, auth, rest = m.group(1), m.group(2), m.group(3)
        host = auth.split("@")[-1]
        return f"{scheme}://{host}" + ("/…" if rest and rest not in ("/",) else "")
    return _URL_RE.sub(_sub, str(s))


def validate_stream_url(url: str):
    url = (url or "").strip()
    if not url:
        return False, "Enter a stream URL."
    if len(url) > 512 or any(c.isspace() for c in url):
        return False, "The URL is too long or contains spaces."
    try:
        p = urlparse(url)
    except Exception:
        return False, "That does not look like a valid URL."
    if p.scheme.lower() not in ("rtsp", "rtsps", "http", "https"):
        return False, "Only rtsp://, rtsps://, http:// and https:// streams are supported."
    if not p.hostname:
        return False, "The URL has no host name."
    return True, ""


# =====================================================================================
# SQLITE PERSISTENCE
# =====================================================================================
def _pick_db_file() -> Path:
    env = os.environ.get("PARKVISION_DB")
    if env:
        return Path(env)
    for base in (APP_DIR / "data", Path(tempfile.gettempdir()) / "parkvision"):
        try:
            base.mkdir(parents=True, exist_ok=True)
            if os.access(base, os.W_OK):
                return base / "parkvision_reserve.db"
        except Exception:
            continue
    return Path(tempfile.gettempdir()) / "parkvision_reserve.db"


_DB_FILE = _pick_db_file()
_DB_READY = False
_DB_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS bookings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ref TEXT UNIQUE NOT NULL, uname TEXT NOT NULL, lot_id TEXT NOT NULL,
  slot_id TEXT NOT NULL, vehicle TEXT NOT NULL, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL,
  total_cents INTEGER NOT NULL, quote_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'confirmed',
  created_ts INTEGER NOT NULL, cancelled_ts INTEGER, refund_cents INTEGER, checked_in_ts INTEGER, discrepancy_ts INTEGER);
CREATE INDEX IF NOT EXISTS ix_b_slot ON bookings(lot_id, slot_id, status, start_ts, end_ts);
CREATE INDEX IF NOT EXISTS ix_b_user ON bookings(uname, status, start_ts);
CREATE TABLE IF NOT EXISTS holds(
  lot_id TEXT, slot_id TEXT, uname TEXT, start_ts INTEGER, end_ts INTEGER, expires_ts INTEGER,
  PRIMARY KEY(lot_id, slot_id, uname));
CREATE TABLE IF NOT EXISTS anchors(
  lot_id TEXT, slot_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL, ref_w INTEGER, ref_h INTEGER,
  PRIMARY KEY(lot_id, slot_id));
CREATE TABLE IF NOT EXISTS watches(
  id INTEGER PRIMARY KEY AUTOINCREMENT, uname TEXT, kind TEXT, lot_id TEXT, slots TEXT, confirm_obs INTEGER,
  confirm_sec REAL, cooldown_sec REAL, vehicle TEXT, active INTEGER DEFAULT 1, snooze_until INTEGER DEFAULT 0,
  created_ts INTEGER);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, uname TEXT, watch_id INTEGER, kind TEXT, source TEXT,
  lot_id TEXT, slot_id TEXT, conf REAL, detail TEXT, bookable INTEGER, ack_ts INTEGER, ack_action TEXT);
CREATE TABLE IF NOT EXISTS observations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, lot_id TEXT, ts INTEGER, total INTEGER, occupied INTEGER, empty INTEGER);
CREATE INDEX IF NOT EXISTS ix_o ON observations(lot_id, ts);
CREATE TABLE IF NOT EXISTS slot_stats(lot_id TEXT, slot_id TEXT, occ INTEGER DEFAULT 0, emp INTEGER DEFAULT 0,
  PRIMARY KEY(lot_id, slot_id));
CREATE TABLE IF NOT EXISTS activity(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, uname TEXT, kind TEXT, msg TEXT);
"""


def _conn(raw: bool = False):
    if not raw and not _DB_READY:
        db_init()                       # first use from any thread creates the schema
    c = sqlite3.connect(str(_DB_FILE), timeout=10, isolation_level=None)  # autocommit; transactions are explicit
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def db_init() -> None:
    global _DB_READY
    if _DB_READY:
        return
    with _DB_LOCK:
        if _DB_READY:
            return
        c = _conn(raw=True)
        try:
            try:
                c.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass
            c.executescript(_SCHEMA)
        finally:
            c.close()
        _DB_READY = True


def db_q(sql: str, args=()) -> list:
    c = _conn()
    try:
        return [dict(r) for r in c.execute(sql, args).fetchall()]
    finally:
        c.close()


def db_x(sql: str, args=()):
    c = _conn()
    try:
        return c.execute(sql, args).lastrowid
    finally:
        c.close()


@contextmanager
def db_tx():
    """BEGIN IMMEDIATE takes the write lock up-front, so conflict checks and the insert are atomic."""
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        yield c
        c.execute("COMMIT")
    except BaseException:
        try:
            c.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        c.close()


def log_event(kind: str, msg: str, uname: str = "") -> None:
    """Activity log. Messages are URL-redacted; no footage or personal data is stored."""
    try:
        db_x("INSERT INTO activity(ts,uname,kind,msg) VALUES(?,?,?,?)", (now_ts(), uname or "", kind, redact_text(msg)[:400]))
        db_x("DELETE FROM activity WHERE id < (SELECT COALESCE(MAX(id),0) - 2000 FROM activity)")
    except Exception:
        log.exception("activity log write failed")


# =====================================================================================
# PASS SIGNING + QR
# =====================================================================================
def _secret() -> bytes:
    env = os.environ.get("PARKVISION_PASS_SECRET")
    if env:
        return env.encode()
    rows = db_q("SELECT v FROM meta WHERE k='pass_secret'")
    if not rows:
        db_x("INSERT OR IGNORE INTO meta(k,v) VALUES('pass_secret',?)", (secrets.token_hex(32),))
        rows = db_q("SELECT v FROM meta WHERE k='pass_secret'")
    return rows[0]["v"].encode()


def _sign(ref, lot_id, slot_id, start, end) -> str:
    msg = f"{ref}|{lot_id}|{slot_id}|{start}|{end}".encode()
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:12]


def pass_payload(b: dict) -> str:
    return f"PV1|{b['ref']}|{b['lot_id']}|{b['slot_id']}|{b['start_ts']}|{b['end_ts']}|{_sign(b['ref'], b['lot_id'], b['slot_id'], b['start_ts'], b['end_ts'])}"


def verify_payload(text: str):
    """Returns (ok, message, booking_or_None)."""
    parts = (text or "").strip().split("|")
    if len(parts) != 7 or parts[0] != "PV1":
        return False, "Not a ParkVision pass payload.", None
    _, ref, lot_id, slot_id, s, e, sig = parts
    try:
        s, e = int(s), int(e)
    except ValueError:
        return False, "Malformed pass.", None
    if not hmac.compare_digest(sig, _sign(ref, lot_id, slot_id, s, e)):
        return False, "Signature check failed. This pass was not issued by this system.", None
    rows = db_q("SELECT * FROM bookings WHERE ref=?", (ref,))
    if not rows:
        return False, "Signature is valid but no booking record exists (different database?).", None
    b = rows[0]
    st_ = booking_status(b, now_ts())
    if st_ == "cancelled":
        return False, f"Booking {ref} was cancelled.", b
    return True, f"Booking {ref} is valid ({st_}).", b


@lru_cache(maxsize=64)
def make_qr_png(text: str, scale: int = 8):
    """QR as PNG bytes. Uses OpenCV's built-in encoder (no extra dependency). Each code is read back with
    OpenCV's decoder and a stronger error-correction level is tried if it does not round-trip; falls back to
    the optional `qrcode` package, then to the best unverified code."""
    best = None
    try:
        try:
            params_cls = cv2.QRCodeEncoder.Params
        except AttributeError:  # older OpenCV naming
            params_cls = cv2.QRCodeEncoder_Params
        for ecc in (3, 2, 1):  # H, Q, M
            prm = params_cls()
            prm.correction_level = ecc
            m = cv2.QRCodeEncoder.create(prm).encode(text)
            if m is None or not m.size:
                continue
            big = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            big = cv2.copyMakeBorder(big, 4 * scale, 4 * scale, 4 * scale, 4 * scale, cv2.BORDER_CONSTANT, value=255)
            ok, buf = cv2.imencode(".png", big)
            if not ok:
                continue
            if best is None:
                best = buf.tobytes()
            if cv2.QRCodeDetector().detectAndDecode(big)[0] == text:
                return buf.tobytes()
    except Exception:
        pass
    try:
        import qrcode  # optional
        img = qrcode.make(text)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return best


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def build_pass_png(b: dict, lot: dict, qr_png) -> bytes:
    W, Hh = 640, 980
    img = Image.new("RGB", (W, Hh), (7, 11, 20))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, W - 16, Hh - 16), radius=28, outline=(56, 217, 255), width=3, fill=(12, 20, 37))
    d.text((48, 48), "PARKVISION RESERVE", font=_font(30), fill=(56, 217, 255))
    d.text((48, 92), "Parking pass (demo - payment simulated)", font=_font(20), fill=(143, 163, 188))
    if qr_png:
        qr = Image.open(io.BytesIO(qr_png)).convert("RGB").resize((420, 420))
        img.paste(qr, ((W - 420) // 2, 150))
    y = 600
    for label, val in (("Booking reference", b["ref"]), ("Destination", lot["name"]), ("Slot", f"{b['slot_id']}  ({b['vehicle']})"),
                       ("Arrival", fmt_dt(b["start_ts"])), ("Departure", fmt_dt(b["end_ts"])), ("Total (simulated)", money(b["total_cents"]))):
        d.text((48, y), label.upper(), font=_font(16), fill=(143, 163, 188))
        d.text((48, y + 20), str(val), font=_font(26), fill=(232, 240, 249))
        y += 58
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


# =====================================================================================
# LOT LAYOUTS (rows of slots with geometry, so distance and routes are real, not decorative)
# =====================================================================================
CELL_M, LANE_M = 2.6, 2.75      # metres per column / per y-unit
ENT_X, MAIN_X, ENT_Y = -1.5, -0.8, 1.0


def _mk_slot(r: int, c: int, sid: str, typ: str, blocked, rowlabel: str) -> dict:
    p, within = divmod(r, 2)
    lane_y = p * 3.0 + 1.0
    y = p * 3.0 + (0.0 if within == 0 else 2.0)
    dist = ((MAIN_X - ENT_X) * CELL_M + abs(lane_y - ENT_Y) * LANE_M + (c - MAIN_X) * CELL_M + abs(y - lane_y) * LANE_M)
    route = [(ENT_X, ENT_Y), (MAIN_X, ENT_Y), (MAIN_X, lane_y), (float(c), lane_y), (float(c), y)]
    return {"id": sid, "row": r, "col": c, "rowlabel": rowlabel, "type": typ, "blocked": blocked,
            "x": float(c), "y": y, "lane_y": lane_y, "dist_m": round(dist, 1), "route": route}


@lru_cache(maxsize=32)
def _demo_layout(lot_id: str) -> tuple:
    lot = LOT_BY_ID[lot_id]
    rows, cols = lot["rows"], lot["cols"]
    rnd = random.Random(lot_id)
    out = []
    for r in range(rows):
        for c in range(cols):
            typ, blocked = "standard", None
            if r < 2 and c < lot.get("acc", 2):
                typ = "accessible"
            elif r >= rows - 2 and c >= cols - lot.get("ev", 2):
                typ = "ev"
            elif r in lot.get("covered_rows", ()):
                typ = "covered"
            if typ in ("standard", "covered") and rnd.random() < 0.03:
                blocked = "Closed for maintenance"
            if lot.get("restricted") and r == rows - 1 and c < 3:
                typ, blocked = "standard", "Staff only"
            out.append(_mk_slot(r, c, f"{chr(65 + r)}{c + 1:02d}", typ, blocked, chr(65 + r)))
    return tuple(out)


def load_anchors(lot_id: str) -> list:
    return db_q("SELECT * FROM anchors WHERE lot_id=?", (lot_id,))


def layout_from_anchors(lot_id: str) -> list:
    """Layout for the live-camera lot: rows are the horizontal bands of the calibrated boxes."""
    anchors = load_anchors(lot_id)
    if not anchors:
        return []
    cy = [(a["y1"] + a["y2"]) / 2 for a in anchors]
    med_h = max(float(np.median([a["y2"] - a["y1"] for a in anchors])), 1.0)
    bands = sorted({round(v / med_h) for v in cy})
    out = []
    for r, band in enumerate(bands):
        row = sorted([a for a, v in zip(anchors, cy) if round(v / med_h) == band], key=lambda a: a["x1"])
        for c, a in enumerate(row):
            out.append(_mk_slot(r, c, a["slot_id"], "standard", None, chr(65 + (r % 26))))
    return out


def get_layout(lot_id: str) -> list:
    lot = LOT_BY_ID[lot_id]
    return layout_from_anchors(lot_id) if lot.get("live_only") else list(_demo_layout(lot_id))


def get_slot(lot_id: str, slot_id: str):
    for s in get_layout(lot_id):
        if s["id"] == slot_id:
            return s
    return None


# =====================================================================================
# OBSERVED OCCUPANCY  (live camera if fresh, otherwise clearly-labelled demo data)
# =====================================================================================
def observed_map(lot_id: str, now: int | None = None) -> dict:
    """slot_id -> {status: occupied|empty, src: live|demo, conf}. Live data overrides demo per mapped slot."""
    now = now or now_ts()
    lot = LOT_BY_ID[lot_id]
    out = {}
    if not lot.get("live_only"):
        rnd = random.Random(f"{lot_id}:{now // 1800}")
        rate = random.Random(lot_id).uniform(0.40, 0.93)
        for s in get_layout(lot_id):
            out[s["id"]] = {"status": "occupied" if rnd.random() < rate else "empty", "src": "demo", "conf": None}
    live = get_hub().live_observation(lot_id)
    if live:
        for sid, (status, conf) in live["slots"].items():
            out[sid] = {"status": status, "src": "live", "conf": conf}
    return out


def lot_has_live(lot_id: str) -> bool:
    return get_hub().live_observation(lot_id) is not None


def lot_summary(lot_id: str, now: int | None = None) -> dict:
    now = now or now_ts()
    layout = get_layout(lot_id)
    obs = observed_map(lot_id, now)
    booked_now = {r["slot_id"] for r in db_q(
        "SELECT slot_id FROM bookings WHERE lot_id=? AND status='confirmed' AND start_ts<=? AND end_ts>?", (lot_id, now, now))}
    occ = res = avail = blocked = 0
    for s in layout:
        if s["blocked"]:
            blocked += 1
            continue
        o = obs.get(s["id"])
        if o and o["status"] == "occupied":
            occ += 1
        elif s["id"] in booked_now:
            res += 1
        else:
            avail += 1
    usable = len(layout) - blocked
    src = "live" if any(o["src"] == "live" for o in obs.values()) else ("demo" if obs else "none")
    return {"capacity": len(layout), "blocked": blocked, "usable": usable, "occupied": occ, "reserved": res,
            "available": avail, "occ_pct": (occ + res) / usable * 100 if usable else 0.0, "src": src}


def availability_label(sm: dict):
    if sm["usable"] == 0:
        return "No data", "vio"
    frac = sm["available"] / sm["usable"]
    if sm["available"] == 0:
        return "Full", "bad"
    if frac < 0.10:
        return "Almost full", "bad"
    if frac < 0.30:
        return "Filling up", "warn"
    return "Available", "ok"


# =====================================================================================
# SLOT RULES, STATES, PRICING
# =====================================================================================
def slot_rule_error(slot: dict, vehicle: str):
    if slot is None:
        return "That slot does not exist."
    if slot["blocked"]:
        return f"Slot {slot['id']} is not available ({slot['blocked']})."
    if slot["type"] not in ELIGIBLE.get(vehicle, set()):
        return f"{SLOT_TYPES[slot['type']]} {slot['id']} is not available for a {vehicle.lower()}."
    return None


def slot_states(lot_id: str, start: int, end: int, vehicle: str, uname: str, selected) -> dict:
    """Bookable state per slot for the chosen window: green / red / cyan / grey / amber."""
    now = now_ts()
    booked = {r["slot_id"]: r["uname"] for r in db_q(
        "SELECT slot_id, uname FROM bookings WHERE lot_id=? AND status='confirmed' AND start_ts<? AND end_ts>?", (lot_id, end, start))}
    held = {r["slot_id"]: r["uname"] for r in db_q(
        "SELECT slot_id, uname FROM holds WHERE lot_id=? AND uname<>? AND expires_ts>? AND start_ts<? AND end_ts>?",
        (lot_id, uname, now, end, start))}
    obs = observed_map(lot_id, now)
    near_now = start - now <= NOW_BLOCK_MIN * 60
    out = {}
    for s in get_layout(lot_id):
        sid = s["id"]
        o = obs.get(sid)
        occ_now = bool(o and o["status"] == "occupied")
        info = {"state": "green", "reason": "Available to reserve", "occ_now": occ_now, "src": o["src"] if o else None}
        err = slot_rule_error(s, vehicle)
        if err:
            info.update(state="grey", reason=err)
        elif sid in booked:
            info.update(state="red", reason="Already booked for an overlapping time" + (" (by you)" if booked[sid] == uname else ""))
        elif sid in held:
            info.update(state="amber", reason="Temporarily held by another user while they confirm")
        elif near_now and occ_now:
            info.update(state="red", reason=f"Occupied right now ({'live camera' if o['src'] == 'live' else 'demo data'})")
        elif sid == selected:
            info.update(state="cyan", reason="Selected by you")
        elif occ_now:
            info["reason"] = "Occupied now, but your arrival is later: it may free up"
        out[sid] = info
    return out


def price_quote(lot: dict, slot: dict, vehicle: str, start: int, end: int) -> dict:
    units = max(1, -(-int((end - start) // 60) // 30))        # 30-minute billing units, rounded up
    hours = units / 2
    base = int(round(lot["rate"] * VEHICLES[vehicle] * hours * 100))
    base = min(base, int(lot["cap"] * 100))                   # daily cap (prototype)
    sur = int(round(SURCHARGE_PER_H.get(slot["type"], 0) * hours * 100))
    fee = int(lot["fee"] * 100)
    sub = base + sur + fee
    tax = int(round(sub * TAX_RATE))
    first_hour = int(round(min(lot["rate"] * VEHICLES[vehicle] * min(hours, 1.0) * 100, lot["cap"] * 100)))
    lines = [(f"Parking {fmt_dur(units * 30)} @ {CURRENCY}{lot['rate'] * VEHICLES[vehicle]:g}/h", base)]
    if sur:
        lines.append((f"{SLOT_TYPES[slot['type']]} surcharge", sur))
    lines += [("Booking fee", fee), (f"Tax ({TAX_RATE * 100:g}%)", tax)]
    return {"lines": lines, "base": base, "surcharge": sur, "fee": fee, "tax": tax, "total": sub + tax,
            "hours": hours, "first_hour": first_hour}


def refund_for(b: dict, now: int):
    """(refund_cents, policy_text). Free cancellation until CANCEL_FREE_MIN before arrival; payment is simulated."""
    q = json.loads(b["quote_json"])
    if now <= b["start_ts"] - CANCEL_FREE_MIN * 60:
        return b["total_cents"], f"Free cancellation (more than {CANCEL_FREE_MIN} min before arrival): full simulated refund."
    kept = int(round((q["fee"] + q["first_hour"]) * (1 + TAX_RATE)))
    kept = min(kept, b["total_cents"])
    return b["total_cents"] - kept, (f"Late cancellation: booking fee and the first hour are kept ({money(kept)}), "
                                     f"the rest is refunded (simulated).")


CANCELLATION_POLICY = (f"Free cancellation up to {CANCEL_FREE_MIN} minutes before arrival. After that, the booking fee and the first "
                       f"hour are kept. No cancellation after the end time. Payment and refunds are SIMULATED in this prototype.")


def booking_status(b: dict, now: int) -> str:
    if b["status"] == "cancelled":
        return "cancelled"
    if now >= b["end_ts"]:
        return "completed"
    if now >= b["start_ts"]:
        return "active"
    return "upcoming"


# =====================================================================================
# BOOKING / HOLD LOGIC  (all conflict checks run inside one write transaction)
# =====================================================================================
def _conflict_error(c, lot_id, slot_id, start, end, uname, now, ignore_own_hold=True):
    c.execute("DELETE FROM holds WHERE expires_ts<=?", (now,))
    row = c.execute("SELECT ref FROM bookings WHERE lot_id=? AND slot_id=? AND status='confirmed' AND start_ts<? AND end_ts>?",
                    (lot_id, slot_id, end, start)).fetchone()
    if row:
        return "That slot is already booked for an overlapping time."
    row = c.execute("SELECT 1 FROM holds WHERE lot_id=? AND slot_id=? AND uname<>? AND start_ts<? AND end_ts>?",
                    (lot_id, slot_id, uname, end, start)).fetchone()
    if row:
        return "That slot is being held by another user right now. Try again in a few minutes or pick another slot."
    row = c.execute("SELECT ref FROM bookings WHERE uname=? AND status='confirmed' AND start_ts<? AND end_ts>?",
                    (uname, end, start)).fetchone()
    if row:
        return f"You already have a booking ({row['ref']}) that overlaps this time. Cancel it first or choose another time."
    return None


def place_hold(uname, lot_id, slot_id, start, end, vehicle) -> None:
    now = now_ts()
    errs = validate_window(start, end, now)
    if errs:
        raise BookingError(errs[0])
    err = slot_rule_error(get_slot(lot_id, slot_id), vehicle)
    if err:
        raise BookingError(err)
    obs = observed_map(lot_id, now).get(slot_id)
    if start - now <= NOW_BLOCK_MIN * 60 and obs and obs["status"] == "occupied":
        raise BookingError(f"Slot {slot_id} is occupied right now ({'live camera' if obs['src'] == 'live' else 'demo data'}).")
    with db_tx() as c:
        err = _conflict_error(c, lot_id, slot_id, start, end, uname, now)
        if err:
            raise BookingError(err)
        c.execute("DELETE FROM holds WHERE uname=?", (uname,))
        c.execute("INSERT OR REPLACE INTO holds(lot_id,slot_id,uname,start_ts,end_ts,expires_ts) VALUES(?,?,?,?,?,?)",
                  (lot_id, slot_id, uname, start, end, now + HOLD_SECONDS))


def release_holds(uname: str) -> None:
    try:
        db_x("DELETE FROM holds WHERE uname=?", (uname,))
    except Exception:
        log.exception("release holds failed")


def _new_ref() -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "PV-" + "".join(secrets.choice(alphabet) for _ in range(8))


def create_booking(uname, lot_id, slot_id, vehicle, start, end) -> dict:
    now = now_ts()
    lot = LOT_BY_ID.get(lot_id)
    if lot is None or vehicle not in VEHICLES:
        raise BookingError("Unknown destination or vehicle type.")
    errs = validate_window(start, end, now)
    if errs:
        raise BookingError(errs[0])
    slot = get_slot(lot_id, slot_id)
    err = slot_rule_error(slot, vehicle)
    if err:
        raise BookingError(err)
    obs = observed_map(lot_id, now).get(slot_id)
    if start - now <= NOW_BLOCK_MIN * 60 and obs and obs["status"] == "occupied":
        raise BookingError(f"Slot {slot_id} is occupied right now, so it cannot be reserved for an immediate arrival.")
    quote = price_quote(lot, slot, vehicle, start, end)
    for _ in range(5):
        ref = _new_ref()
        try:
            with db_tx() as c:
                err = _conflict_error(c, lot_id, slot_id, start, end, uname, now)
                if err:
                    raise BookingError(err)
                c.execute("INSERT INTO bookings(ref,uname,lot_id,slot_id,vehicle,start_ts,end_ts,total_cents,quote_json,status,created_ts) "
                          "VALUES(?,?,?,?,?,?,?,?,?,'confirmed',?)",
                          (ref, uname, lot_id, slot_id, vehicle, start, end, quote["total"], json.dumps(quote), now))
                c.execute("DELETE FROM holds WHERE uname=?", (uname,))
            break
        except sqlite3.IntegrityError:
            continue  # ref collision: try another reference
    else:
        raise BookingError("Could not allocate a booking reference. Please try again.")
    log_event("booking", f"Booking {ref} confirmed: {lot['name']} slot {slot_id}, {fmt_dt(start)} to {fmt_dt(end)}", uname)
    return db_q("SELECT * FROM bookings WHERE ref=?", (ref,))[0]


def cancel_booking(uname: str, ref: str):
    now = now_ts()
    with db_tx() as c:
        row = c.execute("SELECT * FROM bookings WHERE ref=? AND uname=?", (ref, uname)).fetchone()
        if not row:
            raise BookingError("Booking not found.")
        b = dict(row)
        if b["status"] == "cancelled":
            raise BookingError("This booking is already cancelled.")
        if now >= b["end_ts"]:
            raise BookingError("This booking has already finished and cannot be cancelled.")
        refund, text = refund_for(b, now)
        c.execute("UPDATE bookings SET status='cancelled', cancelled_ts=?, refund_cents=? WHERE id=?", (now, refund, b["id"]))
    log_event("booking", f"Booking {ref} cancelled (simulated refund {money(refund)})", uname)
    return refund, text


def check_in(uname: str, ref: str) -> None:
    now = now_ts()
    rows = db_q("SELECT * FROM bookings WHERE ref=? AND uname=? AND status='confirmed'", (ref, uname))
    if not rows:
        raise BookingError("Booking not found.")
    b = rows[0]
    if not (b["start_ts"] - 15 * 60 <= now < b["end_ts"]):
        raise BookingError("Check-in opens 15 minutes before arrival and closes at the end time.")
    db_x("UPDATE bookings SET checked_in_ts=? WHERE id=?", (now, b["id"]))
    log_event("booking", f"Checked in to booking {ref}", uname)


def slot_bookable_now(lot_id: str, slot_id: str, vehicle: str, minutes: int = 60) -> bool:
    """Rules + reservations only (physical occupancy is deliberately not part of this check)."""
    now = now_ts()
    if slot_rule_error(get_slot(lot_id, slot_id), vehicle):
        return False
    c = _conn()
    try:
        c.execute("DELETE FROM holds WHERE expires_ts<=?", (now,))
        end = now + minutes * 60
        if c.execute("SELECT 1 FROM bookings WHERE lot_id=? AND slot_id=? AND status='confirmed' AND start_ts<? AND end_ts>?",
                     (lot_id, slot_id, end, now)).fetchone():
            return False
        if c.execute("SELECT 1 FROM holds WHERE lot_id=? AND slot_id=? AND start_ts<? AND end_ts>?",
                     (lot_id, slot_id, end, now)).fetchone():
            return False
        return True
    finally:
        c.close()


def recommend_slots(lot_id, start, end, vehicle, uname, must: list, priority: str, top: int = 3) -> list:
    lot = LOT_BY_ID[lot_id]
    states = slot_states(lot_id, start, end, vehicle, uname, None)
    need_types = {"EV charging": "ev", "Covered": "covered", "Accessible": "accessible"}
    wanted = {need_types[m] for m in must}
    cands = []
    for s in get_layout(lot_id):
        if states[s["id"]]["state"] != "green":
            continue
        if wanted and s["type"] not in wanted:
            continue
        cands.append((s, price_quote(lot, s, vehicle, start, end)))
    if not cands:
        return []
    max_d = max(s["dist_m"] for s, _ in cands) or 1.0
    max_p = max(q["total"] for _, q in cands) or 1
    wd, wp = {"Closest": (0.85, 0.15), "Cheapest": (0.15, 0.85)}.get(priority, (0.5, 0.5))
    scored = sorted(cands, key=lambda sq: wd * sq[0]["dist_m"] / max_d + wp * sq[1]["total"] / max_p)
    return [{"slot": s, "quote": q} for s, q in scored[:top]]


# =====================================================================================
# WATCHES + ALERTS (persistence)
# =====================================================================================
def add_watch(uname, kind, lot_id, slots, confirm_obs, confirm_sec, cooldown_sec, vehicle="Car") -> int:
    wid = db_x("INSERT INTO watches(uname,kind,lot_id,slots,confirm_obs,confirm_sec,cooldown_sec,vehicle,active,snooze_until,created_ts) "
               "VALUES(?,?,?,?,?,?,?,?,1,0,?)",
               (uname, kind, lot_id, json.dumps(slots) if slots != "*" else "*", int(confirm_obs), float(confirm_sec),
                float(cooldown_sec), vehicle, now_ts()))
    where = "the whole lot" if slots == "*" else ", ".join(slots)
    log_event("watch", f"{'Waitlist' if kind == 'waitlist' else 'Vacancy watch'} created for {LOT_BY_ID[lot_id]['name']} ({where})", uname)
    return wid


def list_watches(uname: str) -> list:
    return db_q("SELECT * FROM watches WHERE uname=? ORDER BY id DESC", (uname,))


def watch_slots(w: dict) -> list:
    if w["slots"] == "*":
        return [s["id"] for s in get_layout(w["lot_id"]) if not s["blocked"]]
    try:
        return list(json.loads(w["slots"]))
    except Exception:
        return []


def count_active_watches(uname: str) -> int:
    r = db_q("SELECT COUNT(*) n FROM watches WHERE uname=? AND active=1", (uname,))
    return int(r[0]["n"]) if r else 0


def create_alert(uname, watch_id, kind, source, lot_id, slot_id, conf, detail, bookable) -> int:
    aid = db_x("INSERT INTO alerts(ts,uname,watch_id,kind,source,lot_id,slot_id,conf,detail,bookable) VALUES(?,?,?,?,?,?,?,?,?,?)",
               (now_ts(), uname, watch_id, kind, source, lot_id, slot_id, conf, detail[:400],
                None if bookable is None else int(bool(bookable))))
    log_event("alert", f"{kind.title()} alert{' (DEMO)' if source == 'demo' else ''}: {LOT_BY_ID[lot_id]['name']} slot {slot_id}", uname)
    return aid


def list_alerts(uname: str, unacked_only: bool = False, limit: int = 100) -> list:
    sql = "SELECT * FROM alerts WHERE uname=?" + (" AND ack_ts IS NULL" if unacked_only else "") + " ORDER BY id DESC LIMIT ?"
    return db_q(sql, (uname, limit))


def count_unacked(uname: str) -> int:
    r = db_q("SELECT COUNT(*) n FROM alerts WHERE uname=? AND ack_ts IS NULL", (uname,))
    return int(r[0]["n"]) if r else 0


def ack_alert(alert_id: int, action: str) -> None:
    db_x("UPDATE alerts SET ack_ts=?, ack_action=? WHERE id=?", (now_ts(), action, alert_id))


# =====================================================================================
# VACANCY STATE MACHINE (pure logic: unit-testable, no camera needed)
# =====================================================================================
class SlotTrack:
    """occupied -> checking vacancy -> vacancy confirmed. A missing observation resets the 'checking' run,
    so one missed vehicle detection can never raise an alert. An alert needs the slot to have been
    seen occupied (armed) first, then N consecutive 'empty' observations spanning at least T seconds."""

    def __init__(self):
        self.state = "unknown"      # unknown | occupied | checking | confirmed | empty | unavailable
        self.occ_run = 0
        self.armed = False
        self.empty_run = 0
        self.empty_since = None
        self.confs = []
        self.last_alert = 0.0

    def update(self, status, conf, now, need_obs, need_sec, cooldown, snoozed=False):
        if status is None:                                   # no detection mapped to this slot this frame
            self.occ_run = 0
            self.empty_run, self.empty_since, self.confs = 0, None, []
            if self.state in ("checking", "unavailable"):
                self.state = "occupied" if self.armed else "unknown"
            return None
        if status == "occupied":
            self.occ_run += 1
            self.empty_run, self.empty_since, self.confs = 0, None, []
            if self.occ_run >= ARM_OBS:
                self.armed = True
            self.state = "occupied"
            return None
        # status == "empty"
        self.occ_run = 0
        if not self.armed:
            if self.state != "confirmed":
                self.state = "empty"
            return None
        if self.empty_run == 0:
            self.empty_since = now
        self.empty_run += 1
        self.confs.append(float(conf or 0.0))
        if self.empty_run >= need_obs and (now - self.empty_since) >= need_sec:
            ev = None
            self.state = "confirmed"
            if (now - self.last_alert) >= cooldown and not snoozed:
                self.last_alert = now
                ev = {"obs": self.empty_run, "span": now - self.empty_since, "conf": sum(self.confs) / len(self.confs)}
            self.armed = False                               # needs a vehicle again before the next alert
            self.empty_run, self.empty_since, self.confs = 0, None, []
            return ev
        self.state = "checking"
        return None

    def mark_unavailable(self):
        self.state = "unavailable"
        self.occ_run = 0
        self.empty_run, self.empty_since, self.confs = 0, None, []

    def snapshot(self):
        return {"state": self.state, "empty_run": self.empty_run, "armed": self.armed}


TRACK_LABEL = {"occupied": ("Occupied", "bad"), "checking": ("Checking vacancy", "warn"), "confirmed": ("Vacancy confirmed", "ok"),
               "unavailable": ("Monitoring unavailable", "vio"), "empty": ("Empty (alert not armed yet)", "info"),
               "unknown": ("Waiting for observations", "vio")}


# =====================================================================================
# DETECTION -> LAYOUT MAPPING (configurable slot-mapping layer; the detector itself is untouched)
# =====================================================================================
def match_detections(anchors: list, dets: list, size) -> dict:
    """Assign each calibrated slot the best-overlapping detection: slot_id -> (status, confidence)."""
    W, Hh = size
    out = {}
    usable = [d for d in dets if d["status"] in ("empty", "occupied")]
    for a in anchors:
        sx, sy = W / max(a["ref_w"], 1), Hh / max(a["ref_h"], 1)
        ax1, ay1, ax2, ay2 = a["x1"] * sx, a["y1"] * sy, a["x2"] * sx, a["y2"] * sy
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-9)
        best, best_score = None, 0.0
        for d in usable:
            iw = max(0.0, min(ax2, d["x2"]) - max(ax1, d["x1"]))
            ih = max(0.0, min(ay2, d["y2"]) - max(ay1, d["y1"]))
            inter = iw * ih
            iou = inter / max(area_a + d["w"] * d["h"] - inter, 1e-9)
            cx, cy = (d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2
            inside = ax1 <= cx <= ax2 and ay1 <= cy <= ay2
            if iou >= 0.25 or inside:
                score = iou + (0.25 if inside else 0.0)
                if score > best_score:
                    best, best_score = d, score
        if best is not None:
            out[a["slot_id"]] = (best["status"], float(best["confidence"]))
    return out


def save_anchors(lot_id: str, items: list, ref_size) -> None:
    """items: [(slot_id, (x1,y1,x2,y2))]. Replaces the previous calibration of this lot."""
    with db_tx() as c:
        c.execute("DELETE FROM anchors WHERE lot_id=?", (lot_id,))
        c.executemany("INSERT INTO anchors(lot_id,slot_id,x1,y1,x2,y2,ref_w,ref_h) VALUES(?,?,?,?,?,?,?,?)",
                      [(lot_id, sid, *map(float, box), int(ref_size[0]), int(ref_size[1])) for sid, box in items])
        c.execute("DELETE FROM slot_stats WHERE lot_id=?", (lot_id,))
    log_event("calibration", f"Slot mapping saved for {LOT_BY_ID[lot_id]['name']} ({len(items)} slots)")


def derive_live_slot_ids(dets: list) -> dict:
    """For the live-camera lot: name slots A01.. by horizontal band (row) and x-order. det id -> slot id."""
    usable = [d for d in dets if d["status"] in ("empty", "occupied")]
    if not usable:
        return {}
    med_h = max(float(np.median([d["h"] for d in usable])), 1.0)
    bands = sorted({round(((d["y1"] + d["y2"]) / 2) / med_h) for d in usable})
    out = {}
    for r, band in enumerate(bands):
        row = sorted([d for d in usable if round(((d["y1"] + d["y2"]) / 2) / med_h) == band], key=lambda d: d["x1"])
        for c, d in enumerate(row):
            out[d["id"]] = f"{chr(65 + (r % 26))}{c + 1:02d}"
    return out


# =====================================================================================
# CAMERA HUB + MONITOR WORKER (one per server process; created via st.cache_resource)
# =====================================================================================
_noop_cb = lambda *a, **k: None
_PREVIEW_RC = {"disp_conf": 0.0, "show": {"empty": True, "occupied": True, "unknown": True}, "fill": 0.18, "thick": 2,
               "ids": False, "conf_lbl": False}


class CameraHub:
    """Owns the single camera resource and the single monitor thread. Never touches Streamlit APIs."""

    def __init__(self):
        self.lock = threading.RLock()
        self.gen = 0
        self.cfg = None
        self.status = "disconnected"        # disconnected | connecting | connected | error
        self.error = ""
        self.label = ""
        self.owner = ""
        self.frame = None
        self.fid = 0
        self.last_ok = 0.0
        self.fps = 0.0
        self._thread = None
        self._stop = threading.Event()
        self.monitor = None
        self.obs = {}
        self.tracks = {}
        self.mon_info = {}
        self.preview = None
        self.devices = {"state": "idle", "items": [], "note": ""}

    # ----- connection -----
    def connect(self, cfg: dict, owner: str) -> None:
        self.disconnect(quiet=True)
        with self.lock:
            self.gen += 1
            gen = self.gen
            self.cfg, self.owner, self.label = cfg, owner, cfg["label"]
            self.status, self.error = "connecting", ""
            self.frame, self.fid, self.last_ok, self.preview = None, 0, 0.0, None
            self._stop = threading.Event()
            if cfg["kind"] == "browser":
                self.status = "connected"
            else:
                self._thread = threading.Thread(target=self._run, args=(cfg, gen, self._stop), daemon=True, name="pv-camera")
                self._thread.start()
        log_event("camera", f"Connecting: {cfg['label']}", owner)

    def disconnect(self, quiet: bool = False) -> None:
        self.stop_monitor()                    # never join threads while holding the lock
        with self.lock:
            self.gen += 1                      # invalidates a grabber that is still opening the device
            self._stop.set()
            t, was = self._thread, self.status
            self._thread = None
        if t is not None:
            t.join(timeout=3.0)
        with self.lock:
            self.status, self.error = "disconnected", ""
            self.frame, self.preview, self.cfg = None, None, None
            self.owner, self.label = "", ""
            self.obs, self.tracks = {}, {}
        if was != "disconnected" and not quiet:
            log_event("camera", "Camera disconnected and released")

    def _run(self, cfg, gen, stop):
        kind, cap = cfg["kind"], None
        try:
            if kind == "image":
                img = cv2.imread(cfg["path"])
                if img is None:
                    raise RuntimeError("The image file could not be read.")
                with self.lock:
                    if gen != self.gen:
                        return
                    self.frame, self.fid, self.last_ok, self.status = img, 1, time.time(), "connected"
                while not stop.wait(1.0):      # static test image: keep 'last frame' fresh, frame id never changes
                    with self.lock:
                        if gen != self.gen:
                            return
                        self.last_ok = time.time()
                return
            if kind == "local":
                cap = cv2.VideoCapture(int(cfg["index"]), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            elif kind == "ip":
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
                cap = cv2.VideoCapture(cfg["url"], cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(cfg["path"])
            if gen != self.gen or stop.is_set():
                return                          # superseded while opening; `finally` releases the device
            if not cap.isOpened():
                raise RuntimeError("The camera or stream could not be opened. Check the device, the URL and the network.")
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            delay = 1.0 / min(max(src_fps, 5.0), 25.0) if kind == "video" else 0.01
            bad, n, t_fps = 0, 0, time.time()
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or getattr(frame, "ndim", 0) != 3 or frame.size == 0:
                    bad += 1
                    if kind == "video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if bad > 5:
                            raise RuntimeError("The video file cannot be decoded.")
                        continue
                    if bad > 60:
                        raise RuntimeError("The camera stopped delivering frames (disconnected?).")
                    stop.wait(0.1)
                    continue
                bad = 0
                with self.lock:
                    if gen != self.gen:
                        return
                    self.frame, self.last_ok, self.status = frame, time.time(), "connected"
                    self.fid += 1
                n += 1
                if n % 20 == 0:
                    now = time.time()
                    self.fps = 20.0 / max(now - t_fps, 1e-6)
                    t_fps = now
                stop.wait(delay)
        except Exception as exc:
            msg = redact_text(str(exc))[:200]
            with self.lock:
                if gen == self.gen:
                    self.status, self.error = "error", msg
            log_event("camera", f"Camera error: {msg}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def push_frame(self, bgr) -> bool:
        with self.lock:
            if self.status == "connected" and self.cfg and self.cfg["kind"] == "browser" and bgr is not None and bgr.size:
                self.frame, self.last_ok = bgr, time.time()
                self.fid += 1
                return True
        return False

    def get_frame(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.fid, self.last_ok

    # ----- monitoring -----
    def start_monitor(self, lot_id, model, info, dcfg, interval, owner) -> None:
        with self.lock:
            if self.status != "connected":
                raise RuntimeError("Connect a camera source first.")
            if self.monitor is not None and self.monitor.is_alive():
                raise RuntimeError("Monitoring is already running. Stop it first.")
            self.monitor = MonitorWorker(self, lot_id, model, info, dcfg, interval, owner)
            self.monitor.start()

    def stop_monitor(self) -> None:
        with self.lock:
            m, self.monitor = self.monitor, None
        if m is not None:
            m.stop()
            m.join(timeout=8.0)

    def monitor_running(self) -> bool:
        m = self.monitor
        return bool(m is not None and m.is_alive())

    def live_observation(self, lot_id: str):
        with self.lock:
            o = self.obs.get(lot_id)
            if o and self.monitor_running() and self.mon_info.get("lot_id") == lot_id and time.time() - o["ts"] <= LIVE_FRESH_SEC:
                return o
        return None

    def shutdown(self) -> None:
        try:
            self.disconnect(quiet=True)
        except Exception:
            pass

    # ----- device discovery (bounded, background) -----
    def discover(self, max_index: int = 5) -> None:
        with self.lock:
            if self.devices.get("state") == "scanning":
                return
            self.devices = {"state": "scanning", "items": [], "note": ""}
        threading.Thread(target=self._discover, args=(max_index,), daemon=True, name="pv-discover").start()

    def _discover(self, max_index):
        items, note = [], ""
        try:
            if os.name == "posix" and sys_platform() == "linux" and not any(os.path.exists(f"/dev/video{i}") for i in range(max_index)):
                note = "No camera device nodes were found on the machine running this app."
            else:
                for i in range(max_index):
                    if os.name == "posix" and sys_platform() == "linux" and not os.path.exists(f"/dev/video{i}"):
                        continue
                    with self.lock:
                        in_use = self.cfg and self.cfg.get("kind") == "local" and self.cfg.get("index") == i and self.status in ("connecting", "connected")
                    if in_use:
                        items.append({"index": i, "label": f"Camera {i} (in use by this app)", "ok": True})
                        continue
                    box = {}

                    def probe(idx=i, box=box):
                        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                        try:
                            if c.isOpened():
                                ok, fr = c.read()
                                box["res"] = (fr.shape[1], fr.shape[0]) if ok and fr is not None else (0, 0)
                        finally:
                            c.release()
                    t = threading.Thread(target=probe, daemon=True)
                    t.start()
                    t.join(timeout=3.0)             # a hung probe is abandoned, never blocks the UI
                    if "res" in box:
                        w, h = box["res"]
                        items.append({"index": i, "label": f"Camera {i}" + (f" ({w}x{h})" if w else ""), "ok": True})
                if not items:
                    note = "No working camera was found. On a cloud deployment the server has no webcam: use an IP camera, a test video, or the browser camera."
        except Exception as exc:
            note = redact_text(str(exc))[:160]
        with self.lock:
            self.devices = {"state": "done", "items": items, "note": note}


def sys_platform() -> str:
    import sys
    return sys.platform


class MonitorWorker(threading.Thread):
    """Runs the EXISTING detector (run_analysis) on fresh camera frames and feeds the vacancy state machine."""

    def __init__(self, hub, lot_id, model, info, dcfg, interval, owner):
        super().__init__(daemon=True, name="pv-monitor")
        self.hub, self.lot_id, self.model, self.info = hub, lot_id, model, info
        self.dcfg, self.interval, self.owner = dict(dcfg), float(interval), owner
        self._stop_evt = threading.Event()
        self.tracks = {}
        self.occ_runs = {}
        self._last_obs_write = 0.0
        hub.mon_info = {"lot_id": lot_id, "started": time.time(), "last_cycle": 0.0, "cycles": 0, "error": "", "ms": 0}

    def stop(self):
        self._stop_evt.set()

    def run(self):
        hub = self.hub
        log_event("monitor", f"Monitoring started: {LOT_BY_ID[self.lot_id]['name']} (every {self.interval:g}s, mode {self.dcfg['mode']})", self.owner)
        last_fid, last_run = -1, 0.0
        try:
            while not self._stop_evt.is_set():
                try:
                    frame, fid, ts = hub.get_frame()
                    now = time.time()
                    if frame is None or now - ts > STALE_FRAME_SEC:
                        self._unavailable("No fresh frames from the camera")
                        self._stop_evt.wait(1.0)
                        continue
                    if fid == last_fid or now - last_run < self.interval:
                        self._stop_evt.wait(0.2)
                        continue
                    last_fid, last_run = fid, now
                    self._cycle(frame, fid)
                except Exception as exc:
                    log.exception("monitor cycle failed")
                    hub.mon_info["error"] = redact_text(str(exc))[:160]
                    self._stop_evt.wait(2.0)
        finally:
            log_event("monitor", "Monitoring stopped", self.owner)

    def _unavailable(self, why):
        for t in self.tracks.values():
            t.mark_unavailable()
        with self.hub.lock:
            self.hub.tracks = {}
            for (wid, sid), t in self.tracks.items():
                self.hub.tracks.setdefault(wid, {})[sid] = t.snapshot()
            self.hub.mon_info["error"] = why

    def _cycle(self, bgr, fid):
        t0 = time.time()
        hub = self.hub
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = MAX_SIDE / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        res = run_analysis(rgb, "camera", f"cam{fid}", self.model, self.info, self.dcfg, _noop_cb)
        anchors = load_anchors(self.lot_id)
        obs = match_detections(anchors, res["dets"], res["size"])
        now = time.time()
        with hub.lock:
            hub.obs[self.lot_id] = {"ts": now, "slots": obs, "n_det": len(res["dets"]), "size": res["size"]}
        # --- vacancy trackers, one per (watch, slot) so each watch keeps its own thresholds ---
        watches = db_q("SELECT * FROM watches WHERE lot_id=? AND active=1", (self.lot_id,))
        live_keys, snap = set(), {}
        for w in watches:
            for sid in watch_slots(w):
                key = (w["id"], sid)
                live_keys.add(key)
                trk = self.tracks.setdefault(key, SlotTrack())
                o = obs.get(sid)
                ev = trk.update(o[0] if o else None, o[1] if o else None, now, w["confirm_obs"], w["confirm_sec"],
                                w["cooldown_sec"], snoozed=w["snooze_until"] > now)
                snap.setdefault(w["id"], {})[sid] = trk.snapshot()
                if ev:
                    self._fire(w, sid, ev)
        for key in list(self.tracks):
            if key not in live_keys:
                del self.tracks[key]
        with hub.lock:
            hub.tracks = snap
        self._check_discrepancies(obs, now)
        self._record(obs, now)
        # --- preview with calibrated slot ids drawn on top ---
        try:
            img = render_annotated(res["rgb"], res["dets"], _PREVIEW_RC)
            sc = max(img.shape[:2]) / 1100.0
            for a in anchors:
                sx, sy = img.shape[1] / max(a["ref_w"], 1), img.shape[0] / max(a["ref_h"], 1)
                cv2.putText(img, a["slot_id"], (int(a["x1"] * sx) + 2, int(a["y2"] * sy) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.4, 0.45 * sc), (255, 255, 255), 1, cv2.LINE_AA)
            k = 960.0 / max(img.shape[:2])
            if k < 1:
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            with hub.lock:
                hub.preview = img
        except Exception:
            log.exception("preview render failed")
        with hub.lock:
            hub.mon_info.update(last_cycle=now, cycles=hub.mon_info.get("cycles", 0) + 1, error="", ms=int((time.time() - t0) * 1000))

    def _fire(self, w, sid, ev):
        lot_id = w["lot_id"]
        detail = (f"Vacancy confirmed after {ev['obs']} consecutive observations over {ev['span']:.0f}s "
                  f"(mean detector confidence {ev['conf']:.2f}).")
        if w["kind"] == "waitlist":
            if not slot_bookable_now(lot_id, sid, w["vehicle"]):
                log_event("alert", f"Waitlist: slot {sid} is vacant but not bookable for a {w['vehicle'].lower()} (reserved or restricted)", w["uname"])
                return
            create_alert(w["uname"], w["id"], "waitlist", "live", lot_id, sid, ev["conf"],
                         detail + " It has no conflicting reservation, so you can book it now.", True)
            db_x("UPDATE watches SET active=0 WHERE id=?", (w["id"],))   # one-shot
        else:
            bookable = slot_bookable_now(lot_id, sid, w["vehicle"])
            create_alert(w["uname"], w["id"], "vacancy", "live", lot_id, sid, ev["conf"],
                         detail + (" It can be reserved." if bookable else " It is reserved or restricted, so it may not be bookable."), bookable)

    def _check_discrepancies(self, obs, now):
        n = int(now)
        rows = db_q("SELECT * FROM bookings WHERE lot_id=? AND status='confirmed' AND start_ts<=? AND end_ts>? "
                    "AND checked_in_ts IS NULL AND discrepancy_ts IS NULL", (self.lot_id, n, n))
        for b in rows:
            o = obs.get(b["slot_id"])
            if o and o[0] == "occupied":
                self.occ_runs[b["ref"]] = self.occ_runs.get(b["ref"], 0) + 1
                if self.occ_runs[b["ref"]] >= DISCREPANCY_OBS:
                    db_x("UPDATE bookings SET discrepancy_ts=? WHERE id=?", (n, b["id"]))
                    create_alert(b["uname"], None, "discrepancy", "live", self.lot_id, b["slot_id"], o[1],
                                 f"Your reserved slot {b['slot_id']} (booking {b['ref']}) looks occupied but you have not checked in. "
                                 f"If it is your vehicle, check in under My Bookings; otherwise verify with the operator. "
                                 f"The booking was NOT cancelled automatically.", None)
            else:
                self.occ_runs[b["ref"]] = 0

    def _record(self, obs, now):
        if not obs or now - self._last_obs_write < 10:
            return
        self._last_obs_write = now
        occ = sum(1 for v in obs.values() if v[0] == "occupied")
        with db_tx() as c:
            c.execute("INSERT INTO observations(lot_id,ts,total,occupied,empty) VALUES(?,?,?,?,?)",
                      (self.lot_id, int(now), len(obs), occ, len(obs) - occ))
            c.executemany("INSERT INTO slot_stats(lot_id,slot_id,occ,emp) VALUES(?,?,?,?) "
                          "ON CONFLICT(lot_id,slot_id) DO UPDATE SET occ=occ+excluded.occ, emp=emp+excluded.emp",
                          [(self.lot_id, sid, 1 if v[0] == "occupied" else 0, 0 if v[0] == "occupied" else 1) for sid, v in obs.items()])


@st.cache_resource
def get_hub() -> CameraHub:
    hub = CameraHub()
    atexit.register(hub.shutdown)        # release the camera when the server process exits
    return hub


def rv_release_for_user(uname: str) -> None:
    """On sign-out: stop monitoring / release the camera only if this user owns them."""
    hub = get_hub()
    if hub.owner and hub.owner == uname:
        hub.disconnect()
    release_holds(uname)


def monitor_state(hub: CameraHub) -> dict:
    """Honest status for the UI: never claims monitoring is working when it is not."""
    with hub.lock:
        info = dict(hub.mon_info)
        running = hub.monitor_running()
        cam = hub.status
    if not running:
        return {"code": "off", "text": "Monitoring is not running", "kind": "vio", "info": info}
    if cam != "connected":
        return {"code": "unavailable", "text": "Monitoring unavailable: camera not connected", "kind": "bad", "info": info}
    age = time.time() - info.get("last_cycle", 0) if info.get("last_cycle") else None
    if age is None:
        return {"code": "starting", "text": "Monitoring started: waiting for the first detection", "kind": "warn", "info": info}
    if age > max(20, 4 * hub_interval(hub)):
        return {"code": "stale", "text": f"Monitoring unavailable: no new observation for {age:.0f}s", "kind": "bad", "info": info}
    return {"code": "ok", "text": f"Monitoring live (last observation {age:.0f}s ago)", "kind": "ok", "info": info}


def hub_interval(hub: CameraHub) -> float:
    m = hub.monitor
    return float(m.interval) if m is not None else 2.0


# =====================================================================================
# DEMO SIMULATION (clearly labelled; runs the REAL state machine with virtual time)
# =====================================================================================
def simulate_sequence(kind: str, need_obs: int, need_sec: float, cooldown: float = 0.0):
    """Returns (steps, event). kind: 'real' (vehicle leaves) or 'blip' (a single missed detection)."""
    trk = SlotTrack()
    seq = ["occupied"] * 3
    seq += ["empty"] * (need_obs + 1) if kind == "real" else [None, "occupied", "occupied"]
    steps, event, t = [], None, 0.0
    step_dt = max(need_sec / max(need_obs - 1, 1), 0.5) if need_obs > 1 else 1.0
    for s in seq:
        ev = trk.update(s, 0.9 if s else None, t, need_obs, need_sec, cooldown)
        steps.append({"t": round(t, 1), "observation": s or "no detection", "state": TRACK_LABEL[trk.state][0], "alert": bool(ev)})
        event = event or ev
        t += step_dt
    return steps, event


# =====================================================================================
# RESERVE UI: styles, scripts, helpers
# =====================================================================================
RV_CSS = """
.rv-chiprow{display:flex;flex-wrap:wrap;gap:2px;align-items:center}
.rv-src{display:inline-block;padding:2px 9px;border-radius:7px;font-size:10.5px;font-weight:700;letter-spacing:.08em;border:1px solid}
.rv-src.demo{color:#FFB84D;border-color:rgba(255,184,77,.5);background:rgba(255,184,77,.08)}
.rv-src.live{color:#35E69B;border-color:rgba(53,230,155,.55);background:rgba(53,230,155,.10)}
.rv-card{padding:18px 20px}
.rv-card h4{font-size:18px;margin:8px 0 2px}
.rv-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0 10px}
.rv-stats div{padding:9px 10px;border-radius:12px;border:1px solid var(--line);background:rgba(10,18,34,.55)}
.rv-stats b{display:block;font-family:'Space Grotesk',sans-serif;font-size:22px;color:#fff}
.rv-stats span{font-size:11.5px;color:var(--muted)}
.rv-bar{height:7px;border-radius:9px;background:rgba(255,255,255,.08);overflow:hidden;display:flex}
.rv-bar i{display:block;height:100%}
.rv-foot{display:flex;justify-content:space-between;align-items:center;margin-top:12px;font-size:13px;color:var(--muted);flex-wrap:wrap;gap:6px}
.rv-lane{height:26px;margin:6px 0;border-radius:8px;display:flex;align-items:center;gap:14px;padding:0 12px;font-size:11px;letter-spacing:.18em;color:rgba(143,163,188,.6);
 background:repeating-linear-gradient(90deg,rgba(255,255,255,.07) 0 22px,transparent 22px 46px) center/100% 2px no-repeat}
.rv-lane b{color:#38D9FF;letter-spacing:.12em;background:#0C1425;padding:2px 8px;border-radius:7px;border:1px solid rgba(56,217,255,.35)}
.rv-gap{height:12px}
.rv-rowlab{height:46px;display:grid;place-items:center;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#8FA3BC}
.rv-legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 10px;font-size:12.5px;color:#b7c8de}
.rv-legend span{display:inline-flex;align-items:center;gap:7px}
.rv-legend i{width:15px;height:15px;border-radius:5px;border:1.5px solid var(--c);background:var(--b)}
.rv-sum{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 4px}
.rv-sum div{padding:9px 14px;border-radius:12px;border:1px solid var(--line);background:rgba(10,18,34,.6);font-size:12.5px;color:var(--muted)}
.rv-sum b{font-family:'Space Grotesk',sans-serif;font-size:20px;color:var(--c,#fff);margin-right:6px}
.rv-price{width:100%;border-collapse:collapse;font-size:14.5px}
.rv-price td{padding:9px 0;border-bottom:1px solid var(--line);color:#dbe7f5}
.rv-price td:last-child{text-align:right;color:#fff;font-variant-numeric:tabular-nums}
.rv-price tr.tot td{border-bottom:0;font-family:'Space Grotesk',sans-serif;font-size:19px;font-weight:700;color:#38D9FF;padding-top:14px}
.rv-kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 18px;margin-top:8px}
.rv-kv span{display:block;font-size:11.5px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.rv-kv b{color:#fff;font-size:15px}
.rv-note{border-left:3px solid var(--rc,#FFB84D);padding:10px 14px;border-radius:0 12px 12px 0;background:rgba(255,184,77,.07);font-size:13.5px;color:#e6dcc7;line-height:1.55}
.rv-alert{position:relative;overflow:hidden;display:flex;gap:18px;align-items:center;padding:18px 22px;margin:4px 0 10px;border-radius:20px;border:1px solid rgba(53,230,155,.55);
 background:linear-gradient(120deg,rgba(14,60,48,.78),rgba(12,20,37,.7));box-shadow:0 0 0 1px rgba(53,230,155,.25),0 0 44px rgba(53,230,155,.22)}
.rv-alert.disc{border-color:rgba(255,82,107,.6);background:linear-gradient(120deg,rgba(70,16,28,.78),rgba(12,20,37,.7));box-shadow:0 0 0 1px rgba(255,82,107,.3),0 0 44px rgba(255,82,107,.2)}
.rv-radar{position:relative;flex:0 0 58px;height:58px;border-radius:50%;background:radial-gradient(circle,#35E69B 0 22%,transparent 24%)}
.rv-alert.disc .rv-radar{background:radial-gradient(circle,#FF526B 0 22%,transparent 24%)}
.rv-radar::before,.rv-radar::after{content:"";position:absolute;inset:0;border-radius:50%;border:2px solid #35E69B;animation:rvping 1.9s ease-out infinite}
.rv-alert.disc .rv-radar::before,.rv-alert.disc .rv-radar::after{border-color:#FF526B}
.rv-radar::after{animation-delay:.95s}
@keyframes rvping{from{transform:scale(.35);opacity:.95}to{transform:scale(1.35);opacity:0}}
.rv-alert h3{margin:0 0 2px;font-size:19px;letter-spacing:.04em}
.rv-alert .slot{font-family:'Space Grotesk',sans-serif;font-size:34px;font-weight:700;color:#fff}
.rv-pass{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:22px;align-items:center}
@media(max-width:800px){.rv-pass{grid-template-columns:1fr}}
.rv-pass img{width:100%;border-radius:14px;background:#fff;padding:8px}
.rv-ref{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:700;letter-spacing:.06em;color:#38D9FF}
/* ----- slot buttons: the state lives in the container key, e.g. rvsgreenA01 ----- */
.st-key-rvgrid{overflow-x:auto;padding:14px 14px 8px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,rgba(12,22,42,.7),rgba(8,14,26,.7));box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.st-key-rvgrid [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:6px!important;min-width:max-content}
.st-key-rvgrid [data-testid="stColumn"],.st-key-rvgrid [data-testid="column"]{min-width:46px!important;flex:1 1 46px!important;width:auto!important}
.st-key-rvgrid [data-testid="stColumn"]:first-child,.st-key-rvgrid [data-testid="column"]:first-child{flex:0 0 30px!important;min-width:30px!important}
.stApp [class*="st-key-rvs"] .stButton{width:100%}
.stApp [class*="st-key-rvs"] button{width:100%;min-height:46px!important;padding:0!important;border-radius:9px!important;font-weight:700!important;transition:transform .15s,box-shadow .2s!important}
.stApp [class*="st-key-rvs"] button p{font-size:11.5px!important;margin:0!important;letter-spacing:.02em}
.stApp [class*="st-key-rvs"] button:disabled{opacity:1!important;cursor:not-allowed!important}
.stApp [class*="st-key-rvsgreen"] button{background:rgba(53,230,155,.10)!important;border:1.5px solid rgba(53,230,155,.85)!important;color:#c9ffe6!important;box-shadow:0 0 12px rgba(53,230,155,.16)!important}
.stApp [class*="st-key-rvsgreen"] button:hover{background:rgba(53,230,155,.30)!important;box-shadow:0 0 24px rgba(53,230,155,.65)!important;transform:translateY(-2px) scale(1.05)!important}
.stApp [class*="st-key-rvsred"] button{background:rgba(255,82,107,.14)!important;border:1.5px solid rgba(255,82,107,.7)!important;color:#ffb3bf!important}
.stApp [class*="st-key-rvscyan"] button{background:rgba(56,217,255,.34)!important;border:2px solid #38D9FF!important;color:#fff!important;animation:rvsel 1.6s ease-in-out infinite}
.stApp [class*="st-key-rvscyan"] button:hover{transform:scale(1.05)!important}
@keyframes rvsel{50%{box-shadow:0 0 26px rgba(56,217,255,.85)}}
.stApp [class*="st-key-rvsgrey"] button{background:rgba(143,163,188,.07)!important;border:1.5px dashed rgba(143,163,188,.35)!important;color:#667a94!important}
.stApp [class*="st-key-rvsamber"] button{background:rgba(255,184,77,.16)!important;border:1.5px solid rgba(255,184,77,.85)!important;color:#ffe0ad!important;animation:rvhold 2s ease-in-out infinite}
@keyframes rvhold{50%{opacity:.62}}
@media(max-width:768px){.rv-kv{grid-template-columns:1fr}.rv-alert{flex-direction:column;align-items:flex-start}}
"""

_JS_LIB = """
var P=(function(){try{void window.parent.document;return window.parent;}catch(e){return window;}})();
var SOUNDS={"Chime":{t:"sine",s:[[880,.18],[1175,.18],[1568,.36]]},
"Radar ping":{t:"triangle",s:[[1320,.12],[0,.1],[1320,.12],[0,.1],[1980,.32]]},
"Siren":{t:"sawtooth",s:[[760,.28],[980,.28],[760,.28],[980,.28]]},
"Beep-beep":{t:"square",s:[[1000,.12],[0,.08],[1000,.12],[0,.08],[1000,.12]]}};
function ctxOK(){return !!(P.__pvCtx&&P.__pvCtx.state==="running");}
function unlock(){try{var AC=P.AudioContext||P.webkitAudioContext;if(!P.__pvCtx)P.__pvCtx=new AC();
return P.__pvCtx.resume().then(function(){return ctxOK();}).catch(function(){return false;});}catch(e){return Promise.resolve(false);}}
function play(name,vol){if(!ctxOK())return false;var c=P.__pvCtx,s=SOUNDS[name]||SOUNDS["Chime"],t=c.currentTime+0.03;
var v=Math.max(0.0002,Math.min(1,vol))*0.35;
s.s.forEach(function(n){var f=n[0],d=n[1];if(f>0){var o=c.createOscillator(),g=c.createGain();o.type=s.t;o.frequency.value=f;
g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(v,t+0.015);g.gain.exponentialRampToValueAtTime(0.0001,t+d);
o.connect(g);g.connect(c.destination);o.start(t);o.stop(t+d+0.02);}t+=d;});return true;}
function notifState(){try{var N=P.Notification;return N?("notifications: "+N.permission):"notifications: not supported";}catch(e){return "notifications: not supported";}}
"""

_SOUND_PANEL_HTML = """
<style>body{margin:0;font-family:Inter,system-ui,sans-serif;color:#E8F0F9}
.row{display:flex;gap:8px;flex-wrap:wrap}
button{cursor:pointer;border:1px solid rgba(56,217,255,.55);background:rgba(16,28,52,.9);color:#E8F0F9;border-radius:10px;padding:9px 14px;font-weight:600;font-size:13px}
button.p{background:linear-gradient(110deg,#1FB6E0,#38D9FF 45%,#9B7BFF);color:#04101c;border:0}
#s{margin-top:8px;font-size:12.5px;color:#8FA3BC}</style>
<div class="row"><button class="p" id="en">🔊 ENABLE SOUND</button><button id="te">▶ Test sound</button><button id="nt">🔔 Enable browser notifications</button></div>
<div id="s"></div>
<script>(function(){var CFG=__CFG__;__LIB__
function render(){document.getElementById("s").textContent=(ctxOK()?"🔊 Sound is enabled in this browser tab":"🔇 Sound is OFF: browsers need one click to allow audio")+" · "+notifState();}
document.getElementById("en").onclick=function(){unlock().then(function(ok){if(ok)play(CFG.name,CFG.vol);render();});};
document.getElementById("te").onclick=function(){unlock().then(function(ok){if(ok)play(CFG.name,CFG.vol);render();});};
document.getElementById("nt").onclick=function(){try{P.Notification.requestPermission().then(render);}catch(e){render();}};
render();})();</script>
"""

_ALARM_HTML = """
<style>body{margin:0;font-family:Inter,system-ui,sans-serif}
button{cursor:pointer;border:0;background:linear-gradient(110deg,#1FB6E0,#38D9FF 45%,#9B7BFF);color:#04101c;border-radius:10px;padding:9px 14px;font-weight:700;font-size:13px}</style>
<div id="b"></div>
<script>(function(){var CFG=__CFG__,A=__ALERTS__;__LIB__
var now=Date.now();P.__pvLast=P.__pvLast||{};var ring=false;
A.forEach(function(a){var last=P.__pvLast[a.id]||0;if(now-last>CFG.repeat*1000){ring=true;P.__pvLast[a.id]=now;
if(!last&&CFG.notify){try{var N=P.Notification;if(N&&N.permission==="granted"){new N(a.title+": "+a.slot,{body:a.text});}}catch(e){}}}});
if(ring&&CFG.sound){play(CFG.name,CFG.vol);}
var b=document.getElementById("b");
if(A.length&&CFG.sound&&!ctxOK()){b.innerHTML='<button id="en">🔊 Sound is blocked by the browser: click to enable</button>';
document.getElementById("en").onclick=function(){unlock().then(function(ok){if(ok)play(CFG.name,CFG.vol);b.innerHTML="";});};}
try{if(A.length){if(!P.__pvT0)P.__pvT0=P.document.title;P.document.title="🔔 "+A[0].title+" "+A[0].slot+" · ParkVision";}
else if(P.__pvT0){P.document.title=P.__pvT0;P.__pvT0=null;}}catch(e){}
})();</script>
"""


def _embed(html: str, height: int = 0) -> None:
    """components.html wrapper: a failing/removed embed API must never break the page."""
    try:
        components.html(html, height=height)
    except Exception:
        log.exception("embedded component failed")


def _json_js(obj) -> str:
    return json.dumps(obj).replace("</", "<\\/")


def rv_css() -> None:
    H(f'<div class="pv-css"></div><style>{RV_CSS}</style>')


def _fragment(run_every=None):
    fr = getattr(st, "fragment", None)
    if fr is None:                                  # very old Streamlit: degrade to a normal function
        return lambda fn: fn
    return fr(run_every=run_every)


def _sel(label, options, pkey, default, widget="selectbox", **kw):
    """Select widget whose value survives leaving the page (Streamlit drops widget state when a widget is not rendered)."""
    ss = st.session_state
    cur = ss.get(pkey, default)
    idx = options.index(cur) if cur in options else 0
    v = (st.radio if widget == "radio" else st.selectbox)(label, options, index=idx, **kw)
    ss[pkey] = v
    return v


def _chip(text, kind="info", pulse=False) -> str:
    return f'<span class="chip {kind}{" pulse" if pulse else ""}"><i></i>{esc(text)}</span>'


def _src_badge(src) -> str:
    if src == "live":
        return '<span class="rv-src live">LIVE CAMERA</span>'
    return '<span class="rv-src demo">DEMO DATA</span>'


def _uname() -> str:
    return st.session_state.get("user") or "guest"


RV_TABS = ["Reserve", "My Bookings", "Vacancy Alerts", "Live Camera & Devices", "Insights & Heatmap", "Activity Log"]


def rv_init() -> None:
    db_init()
    ss = st.session_state
    defaults = {"rv_step": "browse", "rv_lot": None, "rv_sel": None, "rv_sig": None, "rv_p_vehicle": "Car", "rv_p_dur": 120,
                "rv_tz": DEFAULT_TZ, "rv_tab": "Reserve", "rv_p_must": [], "rv_p_prio": "Balanced", "rv_last_ref": None,
                "rv_snd_on": True, "rv_snd_name": "Chime", "rv_snd_vol": 0.6, "rv_bn_on": False, "rv_snooze_min": 10,
                "rv_msg": None, "rv_p_cam": None}
    for k, v in defaults.items():
        ss.setdefault(k, v)


def rv_goto(tab: str) -> None:
    ss = st.session_state
    ss["rv_tab"] = tab
    ss["rv_tab_w"] = tab


# =====================================================================================
# RESERVE FLOW: browse -> pick (slot map) -> review -> done
# =====================================================================================
def rv_choose_lot(lot_id: str) -> None:
    ss = st.session_state
    release_holds(_uname())
    ss.update(rv_lot=lot_id, rv_step="pick", rv_sel=None, rv_sig=None)


def rv_back_to_browse() -> None:
    release_holds(_uname())
    st.session_state.update(rv_step="browse", rv_sel=None, rv_sig=None)


def lot_card_html(lot: dict, sm: dict) -> str:
    label, kind = availability_label(sm)
    u = max(sm["usable"], 1)
    bar = (f'<div class="rv-bar"><i style="width:{sm["occupied"] / u * 100:.1f}%;background:{C_RED}"></i>'
           f'<i style="width:{sm["reserved"] / u * 100:.1f}%;background:{C_AMBER}"></i>'
           f'<i style="width:{sm["available"] / u * 100:.1f}%;background:{C_GREEN}"></i></div>')
    if sm["capacity"] == 0:
        return (f'<div class="glass rv-card"><div class="rv-chiprow">{_chip(lot["cat"], "info")}{_src_badge("live")}</div><h4>{esc(lot["name"])}</h4>'
                f'<div class="muted" style="font-size:13px">{esc(lot["addr"])}</div>'
                f'<div class="rv-note" style="margin-top:14px">Not calibrated yet. Connect a camera and map its slots in <b>Live Camera &amp; Devices</b> to create this lot.</div></div>')
    color = {"ok": C_GREEN, "warn": C_AMBER, "bad": C_RED, "vio": C_VIOLET}[kind]
    return (f'<div class="glass rv-card"><div class="rv-chiprow">{_chip(lot["cat"], "info")}{_src_badge(sm["src"])}</div>'
            f'<h4>{esc(lot["name"])}</h4><div class="muted" style="font-size:13px">{esc(lot["addr"])}</div>'
            f'<div class="rv-stats"><div><b>{sm["capacity"]}</b><span>capacity</span></div>'
            f'<div><b style="color:{C_RED}">{sm["occupied"]}</b><span>occupied</span></div>'
            f'<div><b style="color:{color}">{sm["available"]}</b><span>available</span></div></div>{bar}'
            f'<div class="rv-foot"><span>from {CURRENCY}{lot["rate"]:g}/h · fee {CURRENCY}{lot["fee"]:g}</span>{_chip(label, kind, pulse=kind == "ok")}</div></div>')


def rv_browse() -> None:
    ss = st.session_state
    H('<div class="sect" style="margin-top:6px">Find a place to park</div>')
    H(f'<div class="rv-note" style="--rc:{C_AMBER};margin-bottom:12px">Destinations, capacities, occupancy and fees are <b>simulated demo data</b> '
      f'unless a card says LIVE CAMERA. Reservations are real records in this app\'s database; payment is simulated.</div>')
    up_ = [r for r in db_q("SELECT ref, end_ts FROM bookings WHERE uname=? AND status='confirmed' AND end_ts>?", (_uname(), now_ts()))]
    if up_:
        st.info(f"You have {len(up_)} active or upcoming booking(s). A new booking can't overlap their times; see My Bookings to cancel one.")
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.2, 1.1])
    q = c1.text_input("Search destinations", placeholder="Search by name or address", key="rv_q", label_visibility="collapsed")
    cat = c2.selectbox("Category", ["All categories"] + CATEGORIES, key="rv_cat", label_visibility="collapsed")
    sort = c3.selectbox("Sort", ["Most available", "Name", "Lowest fee"], key="rv_sort", label_visibility="collapsed")
    only = c4.toggle("Free spaces only", key="rv_only")
    now = now_ts()
    rows = []
    for lot in LOTS:
        if cat != "All categories" and lot["cat"] != cat:
            continue
        if q and q.lower().strip() not in (lot["name"] + " " + lot["addr"] + " " + lot["cat"]).lower():
            continue
        sm = lot_summary(lot["id"], now)
        if only and sm["available"] == 0:
            continue
        rows.append((lot, sm))
    if sort == "Most available":
        rows.sort(key=lambda r: -r[1]["available"])
    elif sort == "Name":
        rows.sort(key=lambda r: r[0]["name"])
    else:
        rows.sort(key=lambda r: r[0]["rate"])
    if not rows:
        st.info("No destinations match these filters.")
        return
    cols = st.columns(3)
    for i, (lot, sm) in enumerate(rows):
        with cols[i % 3]:
            H(lot_card_html(lot, sm))
            st.button("Select this destination ▸", key=f"rv_pick_{lot['id']}", type="primary", disabled=sm["capacity"] == 0,
                      on_click=rv_choose_lot, args=(lot["id"],))
            st.markdown("")


def rv_select_slot(sid: str) -> None:
    ss = st.session_state
    uname = _uname()
    if ss.get("rv_sel") == sid:
        ss["rv_sel"] = None
        release_holds(uname)
        return
    arr = ss.get("rv_p_arr")
    try:
        place_hold(uname, ss["rv_lot"], sid, arr, arr + ss["rv_p_dur"] * 60, ss["rv_p_vehicle"])
        ss["rv_sel"] = sid
        ss["rv_flash"] = None
    except BookingError as exc:
        ss["rv_sel"] = None
        ss["rv_flash"] = str(exc)


def rv_pick_recommended(sid: str) -> None:
    st.session_state["rv_sel"] = None
    rv_select_slot(sid)


def route_fig(lot_id: str, states: dict, selected):
    layout = get_layout(lot_id)
    fig = go.Figure()
    for s_name, label in (("green", "Available"), ("red", "Booked / occupied"), ("amber", "Held"), ("grey", "Unavailable")):
        pts = [s for s in layout if states.get(s["id"], {}).get("state") == s_name]
        if pts:
            fig.add_trace(go.Scatter(x=[p["x"] for p in pts], y=[p["y"] for p in pts], mode="markers", name=label, text=[p["id"] for p in pts],
                                     hovertemplate="%{text}<extra></extra>",
                                     marker=dict(symbol="square", size=11, color=SLOT_COLORS[s_name], opacity=.85)))
    if selected:
        s = get_slot(lot_id, selected)
        if s:
            fig.add_trace(go.Scatter(x=[p[0] for p in s["route"]], y=[p[1] for p in s["route"]], mode="lines+markers", name="Route",
                                     line=dict(color=C_CYAN, width=4, dash="dot"), marker=dict(size=[12, 4, 4, 4, 15], color=C_CYAN)))
            fig.add_annotation(x=s["route"][0][0], y=s["route"][0][1], text="Entrance", showarrow=False, yshift=16, font=dict(color=C_CYAN, size=12))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=False, showticklabels=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_layout(title="Entry guidance" if selected else "Lot map")
    return style_fig(fig, 300)


def _rv_map_impl() -> None:
    ss = st.session_state
    lot_id, uname = ss["rv_lot"], _uname()
    lot = LOT_BY_ID[lot_id]
    arr = ss.get("rv_p_arr")
    if arr is None:
        return
    end, veh = arr + ss["rv_p_dur"] * 60, ss["rv_p_vehicle"]
    layout = get_layout(lot_id)
    if not layout:
        st.info("This lot has no slot layout yet.")
        return
    sel = ss.get("rv_sel")
    states = slot_states(lot_id, arr, end, veh, uname, sel)
    if sel and states.get(sel, {}).get("state") != "cyan":
        ss["rv_sel"], sel = None, None
        release_holds(uname)
        states = slot_states(lot_id, arr, end, veh, uname, None)
    flash = ss.get("rv_flash")
    if flash:
        st.warning(flash)
    cnt = {k: sum(1 for v in states.values() if v["state"] == k) for k in ("green", "red", "cyan", "grey", "amber")}
    usable = max(len(states) - cnt["grey"], 1)
    has_live = any(v["src"] == "live" for v in states.values())
    H(f'<div class="rv-sum"><div style="--c:{C_GREEN}"><b>{cnt["green"]}</b>available</div><div style="--c:{C_RED}"><b>{cnt["red"]}</b>booked / occupied</div>'
      f'<div style="--c:{C_AMBER}"><b>{cnt["amber"]}</b>held</div><div style="--c:#8FA3BC"><b>{cnt["grey"]}</b>unavailable</div>'
      f'<div style="--c:{C_CYAN}"><b>{cnt["cyan"]}</b>selected</div><div style="--c:#fff"><b>{(usable - cnt["green"] - cnt["cyan"]) / usable * 100:.0f}%</b>utilisation for this window</div>'
      f'<div>{_src_badge("live" if has_live else "demo")}</div></div>')
    H('<div class="rv-legend">'
      f'<span style="--c:{C_GREEN};--b:rgba(53,230,155,.12)"><i></i>Available</span><span style="--c:{C_RED};--b:rgba(255,82,107,.18)"><i></i>Occupied / booked</span>'
      f'<span style="--c:{C_CYAN};--b:rgba(56,217,255,.4)"><i></i>Your selection</span><span style="--c:#8FA3BC;--b:transparent"><i></i>Unavailable / restricted</span>'
      f'<span style="--c:{C_AMBER};--b:rgba(255,184,77,.2)"><i></i>Held / awaiting confirmation</span></div>')
    rows = {}
    for s in layout:
        rows.setdefault(s["row"], []).append(s)
    with st.container(key="rvgrid"):
        first_lane = True
        for r in sorted(rows):
            cols = st.columns([0.45] + [1] * len(rows[r]), gap="small")
            with cols[0]:
                H(f'<div class="rv-rowlab">{rows[r][0]["rowlabel"]}</div>')
            for col, s in zip(cols[1:], rows[r]):
                stt = states[s["id"]]
                with col:
                    with st.container(key=f"rvs{stt['state']}{s['id']}"):
                        st.button(s["id"], key=f"rvb{s['id']}", disabled=stt["state"] in ("red", "grey", "amber"),
                                  help=f"{s['id']} · {SLOT_TYPES[s['type']]} · {stt['reason']}", on_click=rv_select_slot, args=(s["id"],))
            if r % 2 == 0:
                H(f'<div class="rv-lane">{"<b>ENTRANCE ▸</b>" if first_lane else ""}<span>⟶ ⟶ ⟶ ⟶ ⟶ ⟶</span></div>')
                first_lane = False
            else:
                H('<div class="rv-gap"></div>')
    left, right = st.columns([1, 1])
    with left:
        if sel:
            s = get_slot(lot_id, sel)
            q = price_quote(lot, s, veh, arr, end)
            info = states[sel]
            fac = " · ".join(FACILITIES[s["type"]])
            H(f'<div class="glass flat"><div class="rv-chiprow">{_chip("Selected", "info", pulse=True)}{_chip(SLOT_TYPES[s["type"]], "vio")}</div>'
              f'<div class="rv-ref" style="margin:6px 0 4px">{sel}</div>'
              f'<div class="rv-kv"><div><span>Status</span><b>Available to reserve</b></div><div><span>Distance from entrance</span><b>{s["dist_m"]:.0f} m (~{s["dist_m"] / WALK_SPEED_MPS:.0f} s walk)</b></div>'
              f'<div><span>Facilities</span><b>{esc(fac)}</b></div><div><span>Estimated cost</span><b>{money(q["total"])} <small class="muted">(simulated)</small></b></div>'
              f'<div><span>Arrival</span><b>{fmt_dt(arr)}</b></div><div><span>Departure</span><b>{fmt_dt(end)}</b></div></div>'
              f'<div class="muted" style="font-size:12.5px;margin-top:10px">Held for you for {HOLD_SECONDS // 60} minutes. '
              f'{"Currently occupied, but your arrival is later so it may free up. " if info["occ_now"] else ""}</div></div>')
        else:
            H('<div class="glass flat"><h4>Pick a green slot</h4><p class="muted" style="font-size:14px;line-height:1.6">Click any green bay to see its details, cost and the walking route. '
              'Red, amber and grey bays cannot be reserved for this time window. Hover a bay to see why.</p></div>')
        if st.button("Next: review booking ▸", type="primary", disabled=not sel, key="rv_next"):
            err = rv_go_review()
            if err:
                st.warning(err)
            else:
                st.rerun()
    with right:
        show_plot(route_fig(lot_id, states, sel), "rv_route")


rv_map_static = _fragment(None)(_rv_map_impl)
rv_map_live = _fragment(5)(_rv_map_impl)       # live camera lots refresh their colours every 5 s


def rv_go_review():
    """Re-validate and refresh the hold, then move to the review step. Returns an error string or None."""
    ss = st.session_state
    sel, arr = ss.get("rv_sel"), ss.get("rv_p_arr")
    if not sel or arr is None:
        return "Select a slot first."
    try:
        place_hold(_uname(), ss["rv_lot"], sel, arr, arr + ss["rv_p_dur"] * 60, ss["rv_p_vehicle"])
    except BookingError as exc:
        ss["rv_sel"] = None
        return str(exc)
    ss["rv_step"] = "review"
    return None


def rv_pick() -> None:
    ss = st.session_state
    lot = LOT_BY_ID[ss["rv_lot"]]
    uname, now = _uname(), now_ts()
    sm = lot_summary(lot["id"], now)
    st.button("◂ All destinations", key="rv_back_browse", on_click=rv_back_to_browse)
    H(f'<div class="glass flat"><div class="rv-chiprow">{_chip(lot["cat"], "info")}{_src_badge(sm["src"])}</div>'
      f'<h3 style="margin:6px 0 2px">{esc(lot["name"])}</h3><div class="muted" style="font-size:13.5px">{esc(lot["addr"])} · '
      f'{sm["available"]} of {sm["usable"]} bays free right now · from {CURRENCY}{lot["rate"]:g}/h</div></div>')
    if not get_layout(lot["id"]):
        st.info("This lot has no layout yet. Calibrate the camera first.")
        return
    today = local_now().date()
    tomorrow = today + timedelta(days=1)
    pdate = ss.get("rv_p_date")
    if not isinstance(pdate, type(today)) or not (today <= pdate <= tomorrow):
        pdate = today
    if "rv_date_w" in ss and (not isinstance(ss["rv_date_w"], type(today)) or not (today <= ss["rv_date_w"] <= tomorrow)):
        del ss["rv_date_w"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d = st.date_input("Date", value=pdate, min_value=today, max_value=tomorrow, key="rv_date_w")
        ss["rv_p_date"] = d
    opts = arrival_options(d, now)
    with c2:
        if opts:
            cur = ss.get("rv_p_arr")
            arr = st.selectbox("Arrival time", opts, index=opts.index(cur) if cur in opts else 0, format_func=lambda t: fmt_clock(t), key=f"rv_arr_w{d}_{opts[0]}_{opts[-1]}")
            ss["rv_p_arr"] = arr
        else:
            arr = None
            st.selectbox("Arrival time", ["No valid time"], disabled=True, key="rv_arr_none")
    with c3:
        dur = _sel("Duration", DURATIONS, "rv_p_dur", 120, key="rv_dur_w", format_func=fmt_dur)
    with c4:
        veh = _sel("Vehicle type", list(VEHICLES), "rv_p_vehicle", "Car", key="rv_veh_w")
    if not opts or arr is None:
        st.warning(f"No arrival times are left on this date inside the {BOOKING_WINDOW_H}-hour booking window. Pick another date.")
        return
    end = arr + dur * 60
    errs = validate_window(arr, end, now)
    if errs:
        st.error(errs[0])
        return
    sig = (lot["id"], arr, dur, veh)
    if ss.get("rv_sig") != sig:                     # window or vehicle changed: the old hold no longer applies
        ss["rv_sig"], ss["rv_sel"] = sig, None
        release_holds(uname)
    H(f'<div class="muted" style="font-size:13px;margin:2px 0 6px">Stay: <b style="color:#fff">{fmt_dt(arr)}</b> to <b style="color:#fff">{fmt_dt(end)}</b> '
      f'({fmt_dur(dur)}) · times shown in {esc(rv_tzname())} · bookable up to {BOOKING_WINDOW_H} h ahead</div>')

    clash = rv_overlap_rows(uname, arr, end)
    if clash:
        c0 = clash[0]
        st.warning(f"You already have booking {c0['ref']} ({_lot_name(c0['lot_id'])}, slot {c0['slot_id']}, {fmt_dt(c0['start_ts'])} to {fmt_dt(c0['end_ts'])}) "
                   f"that overlaps this time. A new booking for this window will be rejected. Cancel it in My Bookings or pick a different time.")

    with st.expander("Smart slot recommendations", expanded=False):
        r1, r2 = st.columns([2, 1.4])
        must = r1.multiselect("Must have", ["EV charging", "Covered", "Accessible"], default=ss.get("rv_p_must", []), key="rv_must_w")
        ss["rv_p_must"] = must
        with r2:
            prio = _sel("Optimise for", ["Balanced", "Closest", "Cheapest"], "rv_p_prio", "Balanced", widget="radio", horizontal=True, key="rv_prio_w")
        recs = recommend_slots(lot["id"], arr, end, veh, uname, must, prio)
        if not recs:
            st.info("No free slot matches those needs for this window.")
        for i, r in enumerate(recs):
            s, q = r["slot"], r["quote"]
            a, b = st.columns([4, 1])
            a.markdown(f"**{s['id']}** · {SLOT_TYPES[s['type']]} · {s['dist_m']:.0f} m from the entrance · {money(q['total'])} (simulated)")
            b.button("Pick", key=f"rv_rec_{i}_{s['id']}", on_click=rv_pick_recommended, args=(s["id"],))

    (rv_map_live if lot_has_live(lot["id"]) else rv_map_static)()


def rv_review() -> None:
    ss = st.session_state
    uname, now = _uname(), now_ts()
    lot, sel, arr = LOT_BY_ID.get(ss.get("rv_lot")), ss.get("rv_sel"), ss.get("rv_p_arr")
    if not lot or not sel or arr is None:
        ss["rv_step"] = "pick"
        st.rerun()
    dur, veh = ss["rv_p_dur"], ss["rv_p_vehicle"]
    end = arr + dur * 60
    slot = get_slot(lot["id"], sel)
    st.button("◂ Back to slot map", key="rv_back_pick", on_click=lambda: st.session_state.update(rv_step="pick"))
    H('<div class="sect" style="margin-top:6px">Review your reservation</div>')
    problems = validate_window(arr, end, now) + ([e] if (e := slot_rule_error(slot, veh)) else [])
    if problems:
        st.error(problems[0] + " Go back and adjust your selection.")
        return
    q = price_quote(lot, slot, veh, arr, end)
    a, b = st.columns([1.2, 1])
    with a:
        H(f'<div class="glass flat"><div class="rv-chiprow">{_chip("Simulated payment", "warn")}{_chip(SLOT_TYPES[slot["type"]], "vio")}</div>'
          f'<h3 style="margin:6px 0 2px">{esc(lot["name"])}</h3><div class="muted" style="font-size:13px">{esc(lot["addr"])}</div>'
          f'<div class="rv-kv" style="margin-top:14px"><div><span>Slot</span><b>{sel}</b></div><div><span>Vehicle</span><b>{esc(veh)}</b></div>'
          f'<div><span>Date</span><b>{fmt_clock(arr, "%a %d %b %Y")}</b></div><div><span>Arrival</span><b>{fmt_clock(arr)}</b></div>'
          f'<div><span>Duration</span><b>{fmt_dur(dur)}</b></div><div><span>Departure</span><b>{fmt_dt(end)}</b></div></div></div>')
        H(f'<div class="rv-note" style="margin-top:10px"><b>Cancellation policy.</b> {esc(CANCELLATION_POLICY)}</div>')
    with b:
        rows = "".join(f"<tr><td>{esc(lbl)}</td><td>{money(v)}</td></tr>" for lbl, v in q["lines"])
        H(f'<div class="glass flat"><h4>Price breakdown</h4><table class="rv-price">{rows}<tr class="tot"><td>Total</td><td>{money(q["total"])}</td></tr></table>'
          f'<div class="muted" style="font-size:12.5px;margin-top:8px">Prototype rates, not real venue pricing. No money is charged.</div></div>')
    ack = st.checkbox("I understand payment is simulated and no real charge is made.", key="rv_ack")
    if st.button("Confirm reservation", type="primary", disabled=not ack, key="rv_confirm"):
        try:
            b_ = create_booking(uname, lot["id"], sel, veh, arr, end)
        except BookingError as exc:
            st.error(str(exc))
            ss["rv_sel"] = None
            st.button("Choose another slot", key="rv_choose_other", on_click=lambda: st.session_state.update(rv_step="pick"))
            return
        ss.update(rv_last_ref=b_["ref"], rv_step="done", rv_sel=None, rv_sig=None)
        st.rerun()


def rv_done() -> None:
    ss = st.session_state
    rows = db_q("SELECT * FROM bookings WHERE ref=? AND uname=?", (ss.get("rv_last_ref"), _uname()))
    if not rows:
        ss["rv_step"] = "browse"
        st.rerun()
    b = rows[0]
    lot = LOT_BY_ID[b["lot_id"]]
    qr = make_qr_png(pass_payload(b))
    img = f'<img src="data:image/png;base64,{base64.b64encode(qr).decode()}"/>' if qr else '<div class="rv-note">QR code unavailable on this system.</div>'
    H(f'<div class="glass flat"><div class="rv-pass"><div><div class="rv-chiprow">{_chip("Booking confirmed", "ok", pulse=True)}{_chip("Payment simulated", "warn")}</div>'
      f'<div class="muted" style="font-size:13px;margin-top:6px">Booking reference</div><div class="rv-ref">{esc(b["ref"])}</div>'
      f'<div class="rv-kv" style="margin-top:14px"><div><span>Destination</span><b>{esc(lot["name"])}</b></div><div><span>Slot</span><b>{b["slot_id"]} · {esc(b["vehicle"])}</b></div>'
      f'<div><span>Arrival</span><b>{fmt_dt(b["start_ts"])}</b></div><div><span>Departure</span><b>{fmt_dt(b["end_ts"])}</b></div>'
      f'<div><span>Total (simulated)</span><b>{money(b["total_cents"])}</b></div><div><span>Status</span><b>Confirmed</b></div></div></div>{img}</div></div>')
    c1, c2, c3 = st.columns(3)
    with c1:
        _stretch(st.download_button, "Download pass (PNG)", build_pass_png(b, lot, qr), file_name=f"{b['ref']}_pass.png", mime="image/png", key="rv_dl_pass")
    with c2:
        st.button("Book another space", key="rv_again", on_click=lambda: st.session_state.update(rv_step="browse"))
    with c3:
        st.button("Open My Bookings", key="rv_to_bookings", type="primary", on_click=rv_goto, args=("My Bookings",))


def rv_howto() -> None:
    """Short in-app guide, always at the top of the Reserve section."""
    with st.expander("How to reserve a parking space", expanded=not st.session_state.get("rv_seen_howto")):
        st.session_state["rv_seen_howto"] = True
        st.markdown(
            "1. **Choose a destination** from the cards (search, filter by category, or sort by availability).\n"
            "2. **Set your stay:** date, arrival time (within the next 24 hours), duration and vehicle type.\n"
            "3. **Pick a bay** on the map: green is free, cyan is yours, red is taken, amber is held by someone else, grey is not available. "
            "Hover a bay to see why. The details panel shows distance, facilities and cost, and the route map shows the way in.\n"
            "4. Press **Next**, check the price breakdown and cancellation policy, tick the simulated-payment box and **Confirm**.\n"
            "5. Download your **QR pass**. Find, cancel or check in to bookings under **My Bookings**.\n"
            "6. Want to know when a taken bay frees up? Use **Vacancy Alerts**.")
        st.warning("**One booking at a time:** a new booking is rejected if its time overlaps a booking you already have "
                   "(one vehicle cannot be in two places). Cancel the existing booking in My Bookings first, or choose a time that starts after it ends.")


def rv_overlap_rows(uname: str, start: int, end: int) -> list:
    return db_q("SELECT ref, lot_id, slot_id, start_ts, end_ts FROM bookings WHERE uname=? AND status='confirmed' AND start_ts<? AND end_ts>?",
                (uname, end, start))


def rv_reserve_tab() -> None:
    rv_howto()
    step = st.session_state.get("rv_step", "browse")
    if step == "pick" and st.session_state.get("rv_lot"):
        rv_pick()
    elif step == "review":
        rv_review()
    elif step == "done":
        rv_done()
    else:
        rv_browse()


# =====================================================================================
# MY BOOKINGS
# =====================================================================================
def rv_do_cancel(ref: str) -> None:
    try:
        refund, text = cancel_booking(_uname(), ref)
        st.session_state["rv_msg"] = ("ok", f"Booking {ref} cancelled. {text} Refund (simulated): {money(refund)}.")
    except BookingError as exc:
        st.session_state["rv_msg"] = ("err", str(exc))


def rv_do_checkin(ref: str) -> None:
    try:
        check_in(_uname(), ref)
        st.session_state["rv_msg"] = ("ok", f"Checked in to {ref}. Welcome!")
    except BookingError as exc:
        st.session_state["rv_msg"] = ("err", str(exc))


def rv_bookings_tab() -> None:
    ss = st.session_state
    uname, now = _uname(), now_ts()
    H('<div class="sect" style="margin-top:6px">My bookings</div>')
    msg = ss.pop("rv_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
    rows = db_q("SELECT * FROM bookings WHERE uname=? ORDER BY start_ts DESC LIMIT 30", (uname,))
    if not rows:
        empty_state("No bookings yet", "Reserve a space in the Reserve section and it will appear here.")
    else:
        flt = st.radio("Show", ["All", "Upcoming", "Active", "Completed", "Cancelled"], horizontal=True, key="rv_bk_filter", label_visibility="collapsed")
        kind = {"upcoming": "info", "active": "ok", "completed": "vio", "cancelled": "bad"}
        shown = 0
        for b in rows:
            stt = booking_status(b, now)
            if flt != "All" and stt != flt.lower():
                continue
            shown += 1
            lot = LOT_BY_ID.get(b["lot_id"], {"name": b["lot_id"]})
            extra = ""
            if b["checked_in_ts"]:
                extra += _chip("Checked in", "ok")
            if b["discrepancy_ts"]:
                extra += _chip("Occupancy discrepancy flagged", "bad")
            if stt == "cancelled" and b["refund_cents"] is not None:
                extra += _chip(f"Refund {money(b['refund_cents'])} (simulated)", "warn")
            H(f'<div class="glass flat"><div class="rv-chiprow">{_chip(stt.title(), kind[stt], pulse=stt == "active")}{extra}</div>'
              f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap"><div><div class="rv-ref" style="font-size:22px">{esc(b["ref"])}</div>'
              f'<div style="color:#fff;font-weight:600">{esc(lot["name"])} · slot {b["slot_id"]} · {esc(b["vehicle"])}</div>'
              f'<div class="muted" style="font-size:13px">{fmt_dt(b["start_ts"])} → {fmt_dt(b["end_ts"])}</div></div>'
              f'<div style="text-align:right"><div class="muted" style="font-size:12px">Total (simulated)</div><b style="font-size:20px;color:#fff">{money(b["total_cents"])}</b></div></div></div>')
            c1, c2, c3 = st.columns([1, 1, 2])
            if stt in ("upcoming", "active"):
                with c1:
                    refund, _ = refund_for(b, now)
                    st.button("Cancel booking", key=f"rv_can_{b['ref']}", on_click=rv_do_cancel, args=(b["ref"],), help=f"Simulated refund if cancelled now: {money(refund)}")
                if not b["checked_in_ts"] and b["start_ts"] - 15 * 60 <= now < b["end_ts"]:
                    with c2:
                        st.button("I've arrived: check in", key=f"rv_chk_{b['ref']}", type="primary", on_click=rv_do_checkin, args=(b["ref"],))
            if stt != "cancelled":
                with c3:
                    with st.expander("Show parking pass"):
                        qr = make_qr_png(pass_payload(b))
                        if qr:
                            st.image(qr, width=220)
                        st.caption("Pass payload: " + pass_payload(b))
        if flt != "All" and shown == 0:
            st.info(f"No {flt.lower()} bookings.")
        st.caption(CANCELLATION_POLICY)
    with st.expander("Verify a pass (paste the pass payload)"):
        txt = st.text_input("Pass payload", placeholder="PV1|PV-XXXXXXXX|...", key="rv_verify_txt")
        if txt:
            ok, text, _ = verify_payload(txt)
            (st.success if ok else st.error)(text)


# =====================================================================================
# VACANCY ALERTS TAB
# =====================================================================================
SOUND_NAMES = ["Chime", "Radar ping", "Siren", "Beep-beep"]


def _lots_with_layout() -> list:
    return [lot["id"] for lot in LOTS if get_layout(lot["id"])]


def _lot_name(lid) -> str:
    return LOT_BY_ID[lid]["name"] if lid in LOT_BY_ID else str(lid)


def _snd_cfg() -> dict:
    ss = st.session_state
    return {"sound": bool(ss.get("rv_snd_on", True)), "name": ss.get("rv_snd_name", "Chime"), "vol": float(ss.get("rv_snd_vol", 0.6)),
            "notify": bool(ss.get("rv_bn_on", False)), "repeat": 8}


def rv_watch_toggle(wid: int, active: int) -> None:
    db_x("UPDATE watches SET active=? WHERE id=?", (active, wid))


def rv_watch_delete(wid: int) -> None:
    db_x("DELETE FROM watches WHERE id=? AND uname=?", (wid, _uname()))


def rv_watch_snooze(wid: int) -> None:
    db_x("UPDATE watches SET snooze_until=? WHERE id=?", (now_ts() + int(st.session_state.get("rv_snooze_min", 10)) * 60, wid))


def rv_alerts_tab(model, info, dcfg) -> None:
    ss, uname, hub = st.session_state, _uname(), get_hub()
    H('<div class="sect" style="margin-top:6px">Vacancy alerts</div>')
    ms = monitor_state(hub)
    lot_txt = f" · {esc(_lot_name(ms['info'].get('lot_id')))}" if ms["code"] != "off" and ms["info"].get("lot_id") else ""
    H(f'<div class="glass flat"><div class="rv-chiprow">{_chip(ms["text"], ms["kind"], pulse=ms["code"] == "ok")}'
      f'{_chip("Camera: " + hub.status, {"connected": "ok", "connecting": "warn", "error": "bad"}.get(hub.status, "vio"))}</div>'
      f'<div class="muted" style="font-size:13.5px;margin-top:6px">Alerts are raised only from real detection updates of a connected camera{lot_txt}. '
      f'If nothing is being monitored, nothing can alert: start monitoring in <b>Live Camera &amp; Devices</b>. '
      f'The demo simulator at the bottom is separate and always labelled DEMO.</div></div>')
    unack = count_unacked(uname)
    if unack:
        st.warning(f"{unack} alert(s) are waiting for you at the top of the page. Acknowledge them there.")

    with st.expander("Sound and notifications", expanded=True):
        c1, c2, c3 = st.columns([1, 1.2, 1.4])
        ss["rv_snd_on"] = c1.toggle("Sound alarm", value=ss.get("rv_snd_on", True), key="rv_snd_on_w")
        with c2:
            _sel("Notification sound", SOUND_NAMES, "rv_snd_name", "Chime", key="rv_snd_name_w")
        ss["rv_snd_vol"] = c3.slider("Volume", 0.0, 1.0, float(ss.get("rv_snd_vol", 0.6)), 0.05, key="rv_snd_vol_w")
        c4, c5 = st.columns(2)
        ss["rv_bn_on"] = c4.toggle("Browser notification when an alert fires", value=ss.get("rv_bn_on", False), key="rv_bn_on_w")
        ss["rv_snooze_min"] = c5.select_slider("Snooze length (minutes)", options=[5, 10, 15, 30, 60], value=ss.get("rv_snooze_min", 10), key="rv_snooze_w")
        cfg = _snd_cfg()
        _embed(_SOUND_PANEL_HTML.replace("__CFG__", _json_js(cfg)).replace("__LIB__", _JS_LIB), height=92)
        st.caption("Browsers block audio until you click once on the page: press ENABLE SOUND (and, if you want pop-ups, the notifications button). "
                   "Volume and the chosen sound are applied by your browser. Alerts only sound while this app is open in a tab.")

    with st.expander("Watch slots for a vacancy", expanded=True):
        lots = _lots_with_layout()
        lot_id = st.selectbox("Parking lot", lots, format_func=_lot_name, key=f"rv_w_lot{len(lots)}")
        ids = [s["id"] for s in get_layout(lot_id) if not s["blocked"]]
        mode = st.radio("What to watch", ["One slot", "Selected slots", "Whole lot"], horizontal=True, key="rv_w_mode")
        if mode == "One slot":
            picked = [st.selectbox("Slot", ids, key=f"rv_w_one_{lot_id}")]
        elif mode == "Selected slots":
            picked = st.multiselect("Slots", ids, default=ids[:2], key=f"rv_w_many_{lot_id}")
        else:
            picked = "*"
        k1, k2, k3 = st.columns(3)
        c_obs = k1.slider("Confirm after N consecutive observations", 2, 10, 3, key="rv_w_obs",
                          help="A space must look empty this many observations in a row. One missed car never raises an alert.")
        c_sec = k2.slider("Minimum confirmation period (s)", 0, 60, 4, key="rv_w_sec")
        c_cool = k3.slider("Cooldown between alerts (s)", 30, 3600, 120, 30, key="rv_w_cool")
        if not hub.monitor_running() or ms["info"].get("lot_id") != lot_id:
            st.info("This lot is not being monitored right now. The watch is saved, but it can only alert while monitoring runs for this lot.")
        if st.button("Start watching", type="primary", key="rv_w_add", disabled=(picked != "*" and not picked)):
            add_watch(uname, "vacancy", lot_id, picked, c_obs, c_sec, c_cool)
            st.success("Watch created. A slot arms itself after it has been seen occupied, then alerts when it is confirmed vacant.")

    watches = list_watches(uname)
    H('<div class="sect">Your watches</div>')
    if not watches:
        st.caption("No watches yet.")
    snap = hub.tracks if ms["code"] == "ok" else {}
    for w in watches:
        slots = watch_slots(w)
        now = now_ts()
        flags = _chip("Waitlist" if w["kind"] == "waitlist" else "Vacancy watch", "vio")
        flags += _chip("Active" if w["active"] else "Paused", "ok" if w["active"] else "warn")
        if w["snooze_until"] > now:
            flags += _chip(f"Snoozed {(w['snooze_until'] - now) // 60 + 1} min", "warn")
        detail = ("whole lot" if w["slots"] == "*" else ", ".join(slots)) + f" · confirm {w['confirm_obs']} obs / {w['confirm_sec']:g}s · cooldown {w['cooldown_sec']:g}s"
        chips = ""
        if not w["active"]:
            chips = ""
        elif ms["code"] != "ok" or hub.monitor is None or ms["info"].get("lot_id") != w["lot_id"]:
            chips = _chip("Monitoring unavailable", "vio")
        else:
            tr = snap.get(w["id"], {})
            for sid in slots[:16]:
                t = tr.get(sid)
                lab, kind = TRACK_LABEL[t["state"]] if t else TRACK_LABEL["unknown"]
                chips += _chip(f"{sid}: {lab}", kind)
            if len(slots) > 16:
                chips += f'<span class="muted" style="font-size:12px">+{len(slots) - 16} more slots</span>'
        H(f'<div class="glass flat"><div class="rv-chiprow">{flags}</div><b style="color:#fff">{esc(_lot_name(w["lot_id"]))}</b>'
          f'<div class="muted" style="font-size:13px;margin:2px 0 8px">{esc(detail)}</div><div class="rv-chiprow">{chips}</div></div>')
        a, b, c, _ = st.columns([1, 1, 1, 2])
        a.button("Pause" if w["active"] else "Resume", key=f"rv_wt_{w['id']}", on_click=rv_watch_toggle, args=(w["id"], 0 if w["active"] else 1))
        b.button("Snooze", key=f"rv_ws_{w['id']}", on_click=rv_watch_snooze, args=(w["id"],))
        c.button("Remove", key=f"rv_wd_{w['id']}", on_click=rv_watch_delete, args=(w["id"],))

    with st.expander("Smart waitlist: tell me when a suitable space frees up"):
        lots = _lots_with_layout()
        wl_lot = st.selectbox("Parking lot", lots, format_func=_lot_name, key=f"rv_wl_lot{len(lots)}")
        wl_veh = st.selectbox("Vehicle type", list(VEHICLES), key="rv_wl_veh")
        sm = lot_summary(wl_lot)
        label, kind = availability_label(sm)
        H(f'<div class="rv-chiprow">{_chip(label, kind)}{_src_badge(sm["src"])}</div>'
          f'<div class="muted" style="font-size:13px">{sm["available"]} of {sm["usable"]} bays look free. When a bay is confirmed vacant, '
          f'it is checked against reservations and holds before you are told it is bookable.</div>')
        if st.button("Join waitlist", key="rv_wl_join"):
            add_watch(uname, "waitlist", wl_lot, "*", 3, 4, 60, wl_veh)
            st.success("Waitlist active. It is one-shot: you will be notified once, then it switches off.")

    H('<div class="sect">Alert history</div>')
    hist = list_alerts(uname, limit=50)
    if hist:
        df = pd.DataFrame([{"time": fmt_clock(a["ts"], "%d %b %H:%M:%S"), "type": a["kind"], "source": a["source"].upper(), "lot": _lot_name(a["lot_id"]),
                            "slot": a["slot_id"], "confidence": None if a["conf"] is None else round(a["conf"], 2),
                            "bookable": {1: "yes", 0: "no"}.get(a["bookable"], "-"), "status": a["ack_action"] or "waiting"} for a in hist])
        show_df(df)
    else:
        st.caption("No alerts yet.")

    with st.expander("DEMO simulator: try the alarm without a camera"):
        st.caption("This feeds made-up observations through the same vacancy state machine, using virtual time. Alerts it creates are marked DEMO.")
        lots = _lots_with_layout()
        s_lot = st.selectbox("Lot", lots, format_func=_lot_name, key=f"rv_sim_lot{len(lots)}")
        s_slot = st.selectbox("Slot", [s["id"] for s in get_layout(s_lot) if not s["blocked"]], key=f"rv_sim_slot_{s_lot}")
        n_obs, n_sec = ss.get("rv_w_obs", 3), ss.get("rv_w_sec", 4)
        c1, c2 = st.columns(2)
        run = None
        if c1.button("Simulate: vehicle really leaves", key="rv_sim_real"):
            run = "real"
        if c2.button("Simulate: one missed detection (must NOT alert)", key="rv_sim_blip"):
            run = "blip"
        if run:
            steps, ev = simulate_sequence(run, n_obs, n_sec)
            show_df(pd.DataFrame(steps))
            if ev:
                create_alert(uname, None, "vacancy", "demo", s_lot, s_slot, ev["conf"],
                             f"DEMO: vacancy confirmed after {ev['obs']} consecutive simulated observations over {ev['span']:.0f}s.",
                             slot_bookable_now(s_lot, s_slot, "Car"))
                st.success("The state machine confirmed the vacancy and raised an alert (see the banner at the top of the page).")
                st.rerun()
            else:
                st.info("No alert was raised, as expected: a single missing detection does not confirm a vacancy.")


# =====================================================================================
# LIVE CAMERA & DEVICES TAB
# =====================================================================================
CAM_KINDS = ["Built-in webcam", "External USB webcam", "IP camera (RTSP / HTTP)", "Uploaded image or video (test)", "Browser camera (snapshot)"]


def _frame_rgb(bgr, max_side=960):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = max_side / max(rgb.shape[:2])
    return cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb


def _rv_cam_status_impl(build_status: str) -> None:
    hub = get_hub()
    with hub.lock:
        status, err, label, last, fps = hub.status, hub.error, hub.label, hub.last_ok, hub.fps
        frame, preview, cfg = hub.frame, hub.preview, hub.cfg
        owner = hub.owner
    if status != build_status:
        st.rerun()                                      # state changed: re-evaluate polling and controls
    kind = {"connected": "ok", "connecting": "warn", "error": "bad"}.get(status, "vio")
    age = f"{time.time() - last:.1f}s ago" if last else "never"
    chips = _chip(f"Status: {status}", kind, pulse=status == "connected")
    if label:
        chips += _chip(f"Source: {label}", "info")
    if last:
        chips += _chip(f"Last frame: {fmt_clock(last, '%H:%M:%S')} ({age})", "ok" if time.time() - last < STALE_FRAME_SEC else "bad")
    if frame is not None:
        chips += _chip(f"{frame.shape[1]}x{frame.shape[0]}" + (f" · {fps:.0f} fps" if fps and cfg and cfg["kind"] not in ("image", "browser") else ""), "vio")
    if owner:
        chips += _chip(f"Owner: {owner}", "vio")
    H(f'<div class="glass flat"><div class="rv-chiprow">{chips}</div></div>')
    if err:
        st.error(err)
    img = preview if (preview is not None and get_hub().monitor_running()) else (None if frame is None else _frame_rgb(frame))
    if img is not None:
        show_image(img, caption="Live preview with detections" if preview is not None and get_hub().monitor_running() else "Latest frame")
    elif status == "connected":
        st.info("Connected, waiting for the first frame…")


rv_cam_status_static = _fragment(None)(_rv_cam_status_impl)
rv_cam_status_live = _fragment(1.5)(_rv_cam_status_impl)


def rv_calibration(model, info, dcfg) -> None:
    ss, hub = st.session_state, get_hub()
    H('<div class="sect">Slot mapping (calibration)</div>')
    st.caption("The detector finds parking spaces but does not know their names. Calibration stores where each named slot sits in the camera view, "
               "so a detection can be matched to slot B07 frame after frame. The detector itself is not modified.")
    lots_all = [lot["id"] for lot in LOTS]
    c1, c2 = st.columns([1.3, 1.5])
    lot_id = c1.selectbox("Map this camera to lot", lots_all, index=lots_all.index("LIVE-CAM"), format_func=_lot_name, key="rv_cal_lot")
    src = c2.radio("Reference image", ["Current camera frame", "Last Vision Lab analysis"], horizontal=True, key="rv_cal_src")
    if st.button("Detect slots in the reference image", key="rv_cal_run"):
        res = None
        if src.startswith("Current"):
            frame, _, _ = hub.get_frame()
            if frame is None:
                st.error("No camera frame yet. Connect a source first.")
            else:
                with st.spinner("Running the detector…"):
                    res = run_analysis(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "calibration", "cal", model, info, dcfg, _noop_cb)
        else:
            res = ss.get("analysis")
            if res is None:
                st.error("There is no Vision Lab analysis yet. Analyse an image there first.")
        if res is not None:
            dets = [d for d in res["dets"] if d["status"] in ("empty", "occupied")]
            if lot_id == "LIVE-CAM":
                mp = derive_live_slot_ids(dets)
                slot_of = [mp.get(d["id"], "(ignore)") for d in dets]
            else:
                ids = [s["id"] for s in get_layout(lot_id) if not s["blocked"]]
                slot_of = [ids[i] if i < len(ids) else "(ignore)" for i in range(len(dets))]
            table = pd.DataFrame({"Detection": [d["id"] for d in dets], "Detected state": [d["status"] for d in dets],
                                  "Confidence": [d["confidence"] for d in dets], "Layout slot": slot_of})
            ss["rv_calib"] = {"lot": lot_id, "size": res["size"], "dets": dets, "table": table,
                              "img": render_annotated(res["rgb"], res["dets"], {**_PREVIEW_RC, "ids": True})}
            ss.pop("rv_cal_editor", None)
    cal = ss.get("rv_calib")
    if cal and cal["lot"] == lot_id:
        show_image(cal["img"], caption=f"{len(cal['dets'])} detections. IDs (S01…) are matched to layout slots in reading order; fix the table if the order is wrong.")
        ids = [s["id"] for s in get_layout(lot_id) if not s["blocked"]]
        if lot_id != "LIVE-CAM" and len(cal["dets"]) != len(ids):
            st.warning(f"{len(cal['dets'])} detections vs {len(ids)} layout slots. Review the mapping below; unmatched detections are ignored.")
        live_lot = lot_id == "LIVE-CAM"
        cc = {} if live_lot else {"Layout slot": st.column_config.SelectboxColumn("Layout slot", options=ids + ["(ignore)"], required=True)}
        edited = st.data_editor(cal["table"], hide_index=True, key="rv_cal_editor", column_config=cc,
                                disabled=["Detection", "Detected state", "Confidence"] + (["Layout slot"] if live_lot else []))
        if st.button("Save slot mapping", type="primary", key="rv_cal_save"):
            by_id = {d["id"]: d for d in cal["dets"]}
            chosen = [(r["Layout slot"], r["Detection"]) for _, r in edited.iterrows() if r["Layout slot"] != "(ignore)"]
            names = [c[0] for c in chosen]
            if not chosen:
                st.error("Nothing to save: every detection is set to (ignore).")
            elif len(set(names)) != len(names):
                st.error("Each layout slot can be used only once. Fix the duplicates and save again.")
            else:
                save_anchors(lot_id, [(sid, (by_id[did]["x1"], by_id[did]["y1"], by_id[did]["x2"], by_id[did]["y2"])) for sid, did in chosen], cal["size"])
                st.success(f"Saved {len(chosen)} slot positions for {_lot_name(lot_id)}.")
    anchors = load_anchors(lot_id)
    st.caption(f"{_lot_name(lot_id)}: {len(anchors)} calibrated slots.")


def rv_camera_tab(model, info, dcfg) -> None:
    ss, hub, uname = st.session_state, get_hub(), _uname()
    H('<div class="sect" style="margin-top:6px">Live camera and devices</div>')
    H('<div class="rv-note" style="--rc:#38D9FF">The camera is opened by the <b>machine running this app</b>. On Streamlit Cloud that server has no webcam, '
      'so use an IP camera, a test video/image, or the browser camera (snapshots). To use a USB or built-in webcam, run the app locally. '
      'Frames stay in memory: nothing is uploaded or saved.</div>')
    blocked = bool(hub.owner and hub.owner != uname and hub.status != "disconnected")
    if blocked:
        st.warning(f"The camera is currently in use by '{hub.owner}'. Ask them to disconnect it first.")
    kind = _sel("Camera source", CAM_KINDS, "rv_p_cam", CAM_KINDS[0], widget="radio", key="rv_cam_w")
    idx, url, up, shot = 0, "", None, None
    if kind in CAM_KINDS[:2]:
        default_idx = 0 if kind == CAM_KINDS[0] else 1
        c1, c2 = st.columns([1, 2])
        c1.button("Scan for cameras", key="rv_scan", on_click=lambda: get_hub().discover())
        dev = hub.devices
        if dev["state"] == "scanning":
            c2.info("Scanning (bounded to a few seconds)… press any button to refresh the list.")
        if dev["items"]:
            nums = [i["index"] for i in dev["items"]]
            idx = c2.selectbox("Detected devices", nums, index=nums.index(default_idx) if default_idx in nums else 0,
                               format_func=lambda n: next(x["label"] for x in dev["items"] if x["index"] == n), key=f"rv_dev_{default_idx}_{'_'.join(map(str, nums))}")
        else:
            idx = c2.number_input("Device index (0 is usually the built-in camera)", 0, 9, default_idx, key=f"rv_idx_{default_idx}")
            if dev["note"]:
                c2.caption(dev["note"])
    elif kind == CAM_KINDS[2]:
        url = st.text_input("Stream URL (hidden while typing)", type="password", placeholder="rtsp://user:pass@192.168.1.20:554/stream", key="rv_ip_url",
                            help="rtsp://, rtsps://, http:// or https://. Credentials are never shown, logged or stored.")
        st.caption("Only the host is ever displayed. Credentials stay in memory for the connection and are never written to the log or database.")
    elif kind == CAM_KINDS[3]:
        up = st.file_uploader("Test image or video", type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv", "webm"], key="rv_cam_up")
        st.caption("A video is looped, so it behaves like a live camera. A single image gives only one observation, which cannot confirm a vacancy.")
    else:
        shot = st.camera_input("Take a snapshot with your browser camera", key="rv_cam_shot")
        st.caption("The browser camera gives snapshots, not a continuous stream. Each snapshot you push counts as one observation.")

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Connect", type="primary", disabled=blocked, key="rv_connect"):
        cfg, err = None, None
        if kind in CAM_KINDS[:2]:
            cfg = {"kind": "local", "index": int(idx), "label": f"Local camera {int(idx)}"}
        elif kind == CAM_KINDS[2]:
            ok, msg = validate_stream_url(url)
            if ok:
                cfg = {"kind": "ip", "url": url.strip(), "label": redact_text(url.strip())}
            else:
                err = msg
        elif kind == CAM_KINDS[3]:
            if up is None:
                err = "Choose a file first."
            else:
                data = up.getvalue()
                suffix = Path(up.name).suffix.lower() or ".bin"
                path = Path(tempfile.gettempdir()) / f"pv_cam_{hashlib.md5(data).hexdigest()[:10]}{suffix}"
                path.write_bytes(data)
                cfg = {"kind": "image" if suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else "video", "path": str(path), "label": f"Test file: {up.name}"}
        else:
            cfg = {"kind": "browser", "label": "Browser camera (snapshots)"}
        if err:
            st.error(err)
        elif cfg:
            hub.connect(cfg, uname)
            st.rerun()
    if b2.button("Disconnect", disabled=hub.status == "disconnected" or blocked, key="rv_disconnect"):
        hub.disconnect()
        st.rerun()
    b3.button("Refresh preview", key="rv_refresh")
    if kind == CAM_KINDS[4]:
        if b4.button("Use this snapshot", disabled=shot is None or hub.status != "connected", key="rv_push_shot",
                     help="Connect first, then push the snapshot as an observation."):
            img = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if img is None or not hub.push_frame(img):
                st.error("Could not use that snapshot. Make sure the browser camera source is connected.")
            else:
                st.success("Snapshot received.")

    (rv_cam_status_live if hub.status in ("connecting", "connected") else rv_cam_status_static)(hub.status)

    rv_calibration(model, info, dcfg)

    H('<div class="sect">Monitoring</div>')
    ms = monitor_state(hub)
    H(f'<div class="rv-chiprow">{_chip(ms["text"], ms["kind"], pulse=ms["code"] == "ok")}</div>')
    cal_lots = [lot["id"] for lot in LOTS if load_anchors(lot["id"])]
    if not cal_lots:
        st.info("Calibrate at least one lot above before starting monitoring.")
        return
    m1, m2, m3 = st.columns(3)
    mon_lot = m1.selectbox("Lot to monitor", cal_lots, format_func=_lot_name, key="rv_mon_lot_" + "_".join(cal_lots))
    interval = m2.slider("Seconds between detections", 1.0, 10.0, 2.0, 0.5, key="rv_mon_int")
    mode = m3.selectbox("Detection mode while monitoring", ["Fast", "Accurate"], key="rv_mon_mode",
                        help="Uses your sidebar settings with this mode. Fast is one pass per frame.")
    s1, s2, _ = st.columns([1, 1, 2])
    if s1.button("Start monitoring", type="primary", disabled=hub.monitor_running() or hub.status != "connected" or blocked, key="rv_mon_start"):
        try:
            hub.start_monitor(mon_lot, model, info, {**dcfg, "mode": mode}, interval, uname)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if s2.button("Stop monitoring", disabled=not hub.monitor_running() or blocked, key="rv_mon_stop"):
        hub.stop_monitor()
        st.rerun()
    if hub.monitor_running():
        mi = hub.mon_info
        st.caption(f"{mi.get('cycles', 0)} detection cycles · last cycle {mi.get('ms', 0)} ms" + (f" · {mi['error']}" if mi.get("error") else ""))


# =====================================================================================
# INSIGHTS, HEATMAP, ACTIVITY
# =====================================================================================
def heat_matrix(lot_id: str):
    """Returns (rows, cols, matrix of occupancy 0..1, text, is_demo, n_obs)."""
    layout = get_layout(lot_id)
    stats = {r["slot_id"]: r for r in db_q("SELECT * FROM slot_stats WHERE lot_id=?", (lot_id,))}
    n_obs = sum(r["occ"] + r["emp"] for r in stats.values())
    demo = n_obs < 20
    nr, nc = max(s["row"] for s in layout) + 1, max(s["col"] for s in layout) + 1
    z = [[None] * nc for _ in range(nr)]
    txt = [[""] * nc for _ in range(nr)]
    rnd = random.Random(f"heat{lot_id}")
    max_d = max(s["dist_m"] for s in layout) or 1.0
    base = random.Random(lot_id).uniform(0.40, 0.93)
    for s in layout:
        if s["blocked"]:
            continue
        r = stats.get(s["id"])
        if not demo and r and (r["occ"] + r["emp"]) > 0:
            v = r["occ"] / (r["occ"] + r["emp"])
        elif demo:
            v = min(0.98, max(0.05, base + 0.25 * (1 - s["dist_m"] / max_d) - 0.1 + rnd.uniform(-0.12, 0.12)))
        else:
            continue
        z[s["row"]][s["col"]] = v
        txt[s["row"]][s["col"]] = s["id"]
    return nr, nc, z, txt, demo, n_obs


def rv_insights_tab() -> None:
    H('<div class="sect" style="margin-top:6px">Occupancy insights</div>')
    lots = _lots_with_layout()
    lot_id = st.selectbox("Parking lot", lots, format_func=_lot_name, key=f"rv_in_lot{len(lots)}")
    sm = lot_summary(lot_id)
    c = st.columns(4)
    for col, (lab, v, note, colr, suf, dec) in zip(c, [
            ("Total spaces", sm["capacity"], f"{sm['blocked']} closed or restricted", C_CYAN, "", 0),
            ("Occupied now", sm["occupied"], "Observed (live or demo)", C_RED, "", 0),
            ("Available now", sm["available"], f"{sm['reserved']} reserved right now", C_GREEN, "", 0),
            ("Occupancy", sm["occ_pct"], "Occupied + reserved of usable", C_VIOLET, "%", 1)]):
        with col:
            H(kpi(lab, v, note, colr, suf, dec))
    H(f'<div style="margin-top:8px">{_src_badge(sm["src"])}</div>')
    a, b = st.columns([1.5, 1])
    with a:
        nr, nc, z, txt, demo, n_obs = heat_matrix(lot_id)
        fig = go.Figure(go.Heatmap(z=z, text=txt, texttemplate="%{text}", x=list(range(1, nc + 1)), y=[chr(65 + (i % 26)) for i in range(nr)],
                                   zmin=0, zmax=1, colorscale=[[0, "#1f8f66"], [0.5, "#FFB84D"], [1, "#FF526B"]], colorbar=dict(title="Occupied", tickformat=".0%"),
                                   hovertemplate="%{text}: %{z:.0%} occupied<extra></extra>", xgap=3, ygap=3))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Parking heatmap: how often each bay is occupied")
        show_plot(style_fig(fig, 150 + 46 * nr), "rv_heat")
        H(f'<div class="rv-note">{"<b>DEMO DATA.</b> Not enough real observations yet (" + str(n_obs) + "), so this is a simulated pattern." if demo else f"<b>Real observations:</b> built from {n_obs} slot observations recorded by live monitoring."}</div>')
    with b:
        obs = observed_map(lot_id)
        rows = {}
        for s in get_layout(lot_id):
            if s["blocked"] or s["id"] not in obs:
                continue
            r = rows.setdefault(s["rowlabel"], [0, 0])
            r[0 if obs[s["id"]]["status"] == "empty" else 1] += 1
        if rows:
            labels = [f"Row {k}" for k in rows]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=labels, x=[v[0] for v in rows.values()], name="Free", orientation="h", marker_color=C_GREEN))
            fig.add_trace(go.Bar(y=labels, x=[v[1] for v in rows.values()], name="Occupied", orientation="h", marker_color=C_RED))
            fig.update_layout(barmode="stack", title="Availability by row now")
            fig.update_yaxes(autorange="reversed")
            show_plot(style_fig(fig, max(240, 30 * len(rows) + 120)), "rv_rows")
    hist = db_q("SELECT ts, total, occupied FROM observations WHERE lot_id=? ORDER BY ts DESC LIMIT 400", (lot_id,))[::-1]
    if len(hist) >= 2:
        fig = go.Figure(go.Scatter(x=[datetime.fromtimestamp(h["ts"], get_tz(rv_tzname())) for h in hist],
                                   y=[h["occupied"] / max(h["total"], 1) * 100 for h in hist], mode="lines", line=dict(color=C_CYAN, width=3), fill="tozeroy",
                                   fillcolor="rgba(56,217,255,.12)"))
        fig.update_layout(title="Occupancy trend from live observations", yaxis=dict(range=[0, 100], title="Occupancy %"))
        show_plot(style_fig(fig, 300), "rv_trend")
    else:
        hours = list(range(24))
        base = random.Random(lot_id).uniform(0.4, 0.9)
        y = [round(100 * min(0.98, max(0.1, base * (0.55 + 0.45 * np.sin((h - 7) / 24 * 2 * np.pi * 1.0) ** 2))), 1) for h in hours]
        fig = go.Figure(go.Scatter(x=hours, y=y, mode="lines", line=dict(color=C_AMBER, width=3, dash="dot")))
        fig.update_layout(title="Typical day (DEMO DATA)", xaxis_title="Hour of day", yaxis=dict(range=[0, 100], title="Occupancy %"))
        show_plot(style_fig(fig, 280), "rv_trend_demo")
        st.caption("No live observations recorded for this lot yet, so the typical-day curve above is simulated and labelled as demo data.")


def rv_activity_tab() -> None:
    H('<div class="sect" style="margin-top:6px">Activity log</div>')
    st.caption("Bookings, cancellations, alerts and camera events. No footage and no personal data beyond your username is stored. Camera URLs are redacted.")
    kinds = ["booking", "alert", "camera", "monitor", "watch", "calibration"]
    sel = st.multiselect("Event types", kinds, default=kinds, key="rv_act_kinds")
    rows = db_q("SELECT ts, uname, kind, msg FROM activity ORDER BY id DESC LIMIT 300")
    rows = [r for r in rows if r["kind"] in sel and (not r["uname"] or r["uname"] == _uname() or r["kind"] in ("camera", "monitor", "calibration"))]
    if not rows:
        st.info("No matching events yet.")
        return
    df = pd.DataFrame([{"time": fmt_clock(r["ts"], "%d %b %H:%M:%S"), "user": r["uname"], "type": r["kind"], "event": r["msg"]} for r in rows])
    show_df(df)
    _stretch(st.download_button, "Download log (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="parkvision_activity.csv", mime="text/csv", key="rv_act_dl")


# =====================================================================================
# GLOBAL ALERT CENTER (rendered above every page while watches are active or alerts wait)
# =====================================================================================
def rv_alert_action(alert_id: int, action: str, watch_id) -> None:
    ack_alert(alert_id, action)
    if watch_id and action == "snooze":
        db_x("UPDATE watches SET snooze_until=? WHERE id=?", (now_ts() + int(st.session_state.get("rv_snooze_min", 10)) * 60, watch_id))
    if watch_id and action == "disable":
        db_x("UPDATE watches SET active=0 WHERE id=?", (watch_id,))


def _alert_titles(a: dict):
    if a["kind"] == "discrepancy":
        return "RESERVATION DISCREPANCY", "Discrepancy"
    if a["kind"] == "waitlist":
        return "WAITLIST: A SPACE JUST FREED UP", "Waitlist"
    return "VACANCY CONFIRMED", "Vacancy"


def _rv_alert_center_impl() -> None:
    ss, uname = st.session_state, _uname()
    alerts = list_alerts(uname, unacked_only=True, limit=3)
    snd = _snd_cfg()
    if alerts:
        ss["rv_had_alert"] = True
        rv_css()
        for a in alerts:
            title, short = _alert_titles(a)
            bk = {1: "Bookable right now: yes", 0: "Bookable right now: no (reserved or restricted)"}.get(a["bookable"], "")
            conf = f" · confidence {a['conf']:.2f}" if a["conf"] is not None else ""
            demo = '<span class="rv-src demo" style="margin-left:8px">DEMO</span>' if a["source"] == "demo" else ""
            H(f'<div class="rv-alert {"disc" if a["kind"] == "discrepancy" else ""}"><div class="rv-radar"></div><div style="flex:1;min-width:0">'
              f'<h3>{title}{demo}</h3><div class="slot">{esc(a["slot_id"])} <span style="font-size:15px;font-weight:500;color:#cfe3f7">· {esc(_lot_name(a["lot_id"]))}</span></div>'
              f'<div style="color:#cfe3f7;font-size:14px;margin-top:2px">Detected at {fmt_clock(a["ts"], "%H:%M:%S")}{conf} {("· " + bk) if bk else ""}</div>'
              f'<div class="muted" style="font-size:13px;margin-top:4px">{esc(a["detail"])}</div></div></div>')
            c1, c2, c3, c4 = st.columns(4)
            c1.button("Acknowledge", key=f"al_ack_{a['id']}", type="primary", on_click=rv_alert_action, args=(a["id"], "acknowledged", None))
            c2.button(f"Snooze {ss.get('rv_snooze_min', 10)} min", key=f"al_snz_{a['id']}", on_click=rv_alert_action, args=(a["id"], "snoozed", a["watch_id"]),
                      disabled=not a["watch_id"])
            c3.button("Dismiss", key=f"al_dis_{a['id']}", on_click=rv_alert_action, args=(a["id"], "dismissed", None))
            c4.button("Disable alerts", key=f"al_off_{a['id']}", on_click=rv_alert_action, args=(a["id"], "disabled", a["watch_id"]), disabled=not a["watch_id"])
        payload = [{"id": a["id"], "slot": a["slot_id"], "title": _alert_titles(a)[1], "text": f"{_lot_name(a['lot_id'])} at {fmt_clock(a['ts'], '%H:%M:%S')}"} for a in alerts]
        _embed(_ALARM_HTML.replace("__CFG__", _json_js(snd)).replace("__ALERTS__", _json_js(payload)).replace("__LIB__", _JS_LIB),
                        height=52 if snd["sound"] else 0)
    elif ss.pop("rv_had_alert", False):
        _embed(_ALARM_HTML.replace("__CFG__", _json_js(snd)).replace("__ALERTS__", "[]").replace("__LIB__", _JS_LIB), height=0)


rv_alert_static = _fragment(None)(_rv_alert_center_impl)
rv_alert_live = _fragment(3)(_rv_alert_center_impl)


def rv_alert_center() -> None:
    uname = _uname()
    if count_active_watches(uname) > 0:
        rv_alert_live()
    elif count_unacked(uname) > 0 or st.session_state.get("rv_had_alert"):
        rv_alert_static()


# =====================================================================================
# PAGE
# =====================================================================================
def page_reserve(model, info, dcfg, rcfg) -> None:
    rv_css()
    H('<div class="page-title">ParkVision Reserve</div>'
      '<div class="page-sub">Pick a destination, choose a bay on the live-style map, and get a pass. Vacancy alerts and the camera layer reuse your existing detector.</div>')
    tzs = TZ_CHOICES if DEFAULT_TZ in TZ_CHOICES else [DEFAULT_TZ] + TZ_CHOICES
    c1, c2 = st.columns([3, 1.2])
    with c2:
        _sel("Time zone", tzs, "rv_tz", DEFAULT_TZ, key="rv_tz_w")
    with c1:
        hub = get_hub()
        H(f'<div class="rv-chiprow" style="margin-top:6px">{_chip("Local time " + local_now().strftime("%a %d %b, %H:%M"), "info")}'
          f'{_chip("Camera: " + hub.status, {"connected": "ok", "connecting": "warn", "error": "bad"}.get(hub.status, "vio"))}'
          f'{_chip("Booking window: " + str(BOOKING_WINDOW_H) + " h", "vio")}{_chip("Payment simulated", "warn")}</div>')
    tab = _sel("Section", RV_TABS, "rv_tab", "Reserve", widget="radio", horizontal=True, key="rv_tab_w", label_visibility="collapsed")
    if tab == "Reserve":
        rv_reserve_tab()
    elif tab == "My Bookings":
        rv_bookings_tab()
    elif tab == "Vacancy Alerts":
        rv_alerts_tab(model, info, dcfg)
    elif tab == "Live Camera & Devices":
        rv_camera_tab(model, info, dcfg)
    elif tab == "Insights & Heatmap":
        rv_insights_tab()
    else:
        rv_activity_tab()


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

    try:  # [RESERVE] database + vacancy alert banner (shown above every page while alerts are pending)
        rv_init()
        rv_alert_center()
    except Exception as exc:
        log.exception("Reserve module failed to initialise")
        st.warning(f"ParkVision Reserve could not start ({type(exc).__name__}). The rest of the app is unaffected.")

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard(info, path.name, rcfg)
    elif page == "Vision Lab":
        page_lab(model, info, dcfg, rcfg, ui)
    elif page == "Analytics":
        page_analytics(rcfg)
    elif page == "Model Lab":
        page_model(model, info, path.name)
    elif page == "ParkVision Reserve":  # [RESERVE]
        page_reserve(model, info, dcfg, rcfg)
    else:
        page_export(rcfg, path.name)
    st.markdown("")
    st.caption("ParkVision AI estimates availability from the uploaded image only. It is not a live parking feed.")


if os.environ.get("PARKVISION_TEST") != "1":
    main()
