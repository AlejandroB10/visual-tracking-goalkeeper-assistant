"""Helper functions for initial ball detection and visualization."""

import cv2
import numpy as np

DEFAULT_BALL_CFG = {
    "mog2_history": 220,
    "mog2_varThreshold": 28,
    "mog2_detectShadows": False,
    "prewarm_seconds": 0.60,
    "prewarm_lr": 0.08,
    "warmup_frames": 8,
    "warmup_lr": 0.02,
    "run_lr": 0.001,
    "kernel_size": 5,
    "min_area": 90,
    "max_area": 1800,
    "aspect_min": 0.60,
    "aspect_max": 1.65,
    "min_circularity": 0.45,
    "min_extent": 0.40,
    "min_solidity": 0.78,
    "white_sat_max": 110,
    "white_val_min": 125,
    "green_h_min": 30,
    "green_h_max": 95,
    "green_s_min": 35,
    "green_v_min": 35,
    "content_black_thresh": 18,
    "x_center_band_ratio": 0.20,
    "y_min_ratio": 0.48,
    "y_max_ratio": 0.92,
    "min_fg_support": 0.0,
    "min_score": 0.58,
    "min_hits": 2,
    "max_jump": 60,
    "max_fg_density": 0.12,
}


def build_detection_config(user_cfg=None):
    """
    Merge user overrides with internal defaults.

    This keeps notebook config compact while preserving full helper behavior.
    """
    return {**DEFAULT_BALL_CFG, **(user_cfg or {})}


def build_clean_foreground_mask(mog2, frame_bgr, kernel, learning_rate):
    """Apply MOG2 and morphology to obtain a clean binary foreground mask."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    fg = mog2.apply(blur, learningRate=learning_rate)
    fg[fg == 127] = 0
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fg


def get_active_video_bounds(frame_bgr, black_thresh=18):
    """Return left/right x-bounds of non-black content."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = np.mean(gray, axis=0)
    idx = np.where(col_mean > black_thresh)[0]
    if len(idx) == 0:
        return 0, frame_bgr.shape[1] - 1
    return int(idx[0]), int(idx[-1])


def select_best_ball_candidate(frame_bgr, fg_mask, cfg):
    """Return best ball candidate dict, or None if no contour passes all filters."""
    h, w = frame_bgr.shape[:2]
    left, right = get_active_video_bounds(frame_bgr, black_thresh=cfg["content_black_thresh"])
    center_x = 0.5 * (left + right)
    center_band = cfg["x_center_band_ratio"] * (right - left)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        (0, 0, cfg["white_val_min"]),
        (179, cfg["white_sat_max"], 255),
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg["min_area"] <= area <= cfg["max_area"]):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue

        aspect = bw / float(bh)
        if not (cfg["aspect_min"] <= aspect <= cfg["aspect_max"]):
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        if not (cfg["y_min_ratio"] * h <= cy <= cfg["y_max_ratio"] * h):
            continue
        if abs(cx - center_x) > center_band:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < cfg["min_circularity"]:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < cfg["min_solidity"]:
            continue

        extent = area / float(max(bw * bh, 1))
        if extent < cfg["min_extent"]:
            continue

        pad = max(12, int(max(bw, bh) * 1.6))
        x0, x1 = max(0, cx - pad), min(w, cx + pad)
        y0, y1 = max(0, cy - pad), min(h, cy + pad)
        patch_hsv = hsv[y0:y1, x0:x1]
        if patch_hsv.size == 0:
            continue

        green_ratio = float(
            np.mean(
                (patch_hsv[..., 0] >= cfg["green_h_min"])
                & (patch_hsv[..., 0] <= cfg["green_h_max"])
                & (patch_hsv[..., 1] >= cfg["green_s_min"])
                & (patch_hsv[..., 2] >= cfg["green_v_min"])
            )
        )

        fg_support = float(np.mean(fg_mask[y0:y1, x0:x1] > 0))
        if fg_support < cfg["min_fg_support"]:
            continue

        center_score = 1.0 - min(abs(cx - center_x) / max(center_band, 1.0), 1.0)
        y_norm = (cy / float(h) - cfg["y_min_ratio"]) / max(cfg["y_max_ratio"] - cfg["y_min_ratio"], 1e-6)
        y_score = 1.0 - min(abs(y_norm - 0.30), 1.0)

        score = (
            0.24 * circularity
            + 0.12 * solidity
            + 0.10 * extent
            + 0.24 * green_ratio
            + 0.18 * center_score
            + 0.08 * y_score
            + 0.04 * min(fg_support / 0.02, 1.0)
        )

        candidate = {
            "x": int(cx),
            "y": int(cy),
            "score": float(score),
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "radius": int(max(bw, bh) / 2),
            "area": float(area),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "extent": float(extent),
            "green_ratio": float(green_ratio),
            "fg_support": float(fg_support),
            "active_bounds": (int(left), int(right)),
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def draw_ball_marker(vis_rgb, x, y, bbox, label, color=(0, 255, 255)):
    """Draw a large ball marker, crosshair and a text label."""
    x = int(x)
    y = int(y)
    _, _, bw, bh = bbox
    marker_r = max(24, int(max(bw, bh) * 2.0))

    cv2.circle(vis_rgb, (x, y), marker_r, color, 3)
    cv2.drawMarker(
        vis_rgb,
        (x, y),
        (255, 64, 64),
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_r + 20,
        thickness=2,
    )
    cv2.circle(vis_rgb, (x, y), 5, (255, 64, 64), -1)

    txt = f"{label}: ({x}, {y})"
    tx = max(10, x - 140)
    ty = max(25, y - marker_r - 12)
    cv2.rectangle(vis_rgb, (tx - 4, ty - 18), (tx + 250, ty + 6), (0, 0, 0), -1)
    cv2.putText(vis_rgb, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
