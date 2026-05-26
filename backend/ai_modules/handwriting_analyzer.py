"""
VRITTIFY ASTRA – Handwriting Analysis Engine v3.0

Key rewrite goals (v3.0):
  - Fingerprint now captures WRITER-IDENTITY features, not just style stats.
    Old 9-feature vector (stroke_density, height_cv, etc.) was topic/pressure
    sensitive → all students on the same assignment matched each other at 90%+.
  - New 18-feature vector adds: slant angle, curvature, aspect ratio,
    stroke thinness, loop density, baseline deviation — features that are
    stable per writer but differ between writers.
  - compare_handwritings() threshold guidance raised to 0.92+ for a
    "same writer" call (old 0.75 was far too permissive with this vector).
  - compute_consistency_score() unchanged; still used for scoring only.
  - analyze_handwriting() returns fingerprint as list[float] (not ndarray).
"""

import cv2
import numpy as np
from typing import Optional


# ── Core feature extractors ───────────────────────────────────────────────

def extract_stroke_features(binary: np.ndarray) -> dict:
    """
    Extract stroke-level features from a binarized page image.
    binary: uint8 image; ink may be 0 or 255 — handled internally.
    """
    if np.mean(binary) > 127:
        ink = cv2.bitwise_not(binary)
    else:
        ink = binary.copy()

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_stroke_features()

    min_area = 4
    strokes  = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not strokes:
        return _empty_stroke_features()

    areas      = [cv2.contourArea(c)    for c in strokes]
    perimeters = [cv2.arcLength(c, False) for c in strokes]
    bboxes     = [cv2.boundingRect(c)   for c in strokes]
    heights    = [b[3] for b in bboxes]
    widths     = [b[2] for b in bboxes]

    h, w = ink.shape
    stroke_density = sum(areas) / (h * w) if h * w > 0 else 0.0

    mean_h = float(np.mean(heights))
    std_h  = float(np.std(heights))
    h_cv   = std_h / mean_h if mean_h > 0 else 1.0

    mean_w = float(np.mean(widths))
    std_w  = float(np.std(widths))
    w_cv   = std_w / mean_w if mean_w > 0 else 1.0

    # Aspect ratio (width/height) — characteristic of letter style
    aspect_ratios = [
        (b[2] / b[3]) for b in bboxes if b[3] > 0
    ]
    mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0

    # Compactness (4π·area / perimeter²) — measures stroke roundness
    compact = [
        4 * np.pi * a / (p ** 2)
        for a, p in zip(areas, perimeters) if p > 0
    ]
    mean_compact = float(np.mean(compact)) if compact else 0.0

    # Stroke thinness: mean perimeter / mean area ratio
    # Thin cursive strokes have high ratio; block letters have lower ratio
    thin_ratios = [
        p / a for a, p in zip(areas, perimeters) if a > 0
    ]
    mean_thinness = float(np.mean(thin_ratios)) if thin_ratios else 0.0
    # Normalize to 0-1 range (typical range 0.5–10)
    mean_thinness_norm = min(mean_thinness / 10.0, 1.0)

    # Inter-stroke gaps
    sorted_bboxes = sorted(bboxes, key=lambda b: b[0])
    gaps = []
    for i in range(1, len(sorted_bboxes)):
        prev_right = sorted_bboxes[i-1][0] + sorted_bboxes[i-1][2]
        curr_left  = sorted_bboxes[i][0]
        gap = curr_left - prev_right
        if 0 < gap < 200:
            gaps.append(gap)
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    std_gap  = float(np.std(gaps))  if gaps else 0.0
    gap_cv   = std_gap / mean_gap if mean_gap > 0 else 1.0

    # Normalized gap (gap relative to mean letter width)
    gap_to_width_ratio = mean_gap / mean_w if mean_w > 0 else 0.5
    gap_to_width_ratio = min(gap_to_width_ratio, 2.0) / 2.0   # normalize 0-1

    return {
        'stroke_count':        len(strokes),
        'stroke_density':      round(stroke_density, 4),
        'mean_height':         round(mean_h, 2),
        'height_cv':           round(h_cv, 3),
        'mean_width':          round(mean_w, 2),
        'width_cv':            round(w_cv, 3),
        'mean_aspect':         round(mean_aspect, 3),
        'mean_gap':            round(mean_gap, 2),
        'gap_cv':              round(gap_cv, 3),
        'gap_to_width_ratio':  round(gap_to_width_ratio, 3),
        'compactness':         round(mean_compact, 4),
        'thinness':            round(mean_thinness_norm, 4),
        'area_mean':           round(float(np.mean(areas)), 2),
        'area_std':            round(float(np.std(areas)),  2),
    }


def _empty_stroke_features() -> dict:
    return {
        'stroke_count': 0, 'stroke_density': 0.0,
        'mean_height': 0.0, 'height_cv': 1.0,
        'mean_width': 0.0,  'width_cv': 1.0,
        'mean_aspect': 1.0,
        'mean_gap': 0.0,    'gap_cv': 1.0,
        'gap_to_width_ratio': 0.5,
        'compactness': 0.0, 'thinness': 0.0,
        'area_mean': 0.0,   'area_std': 0.0,
    }


def extract_line_features(binary: np.ndarray) -> dict:
    """
    Extract text-line level features via horizontal projection profile.
    """
    if np.mean(binary) > 127:
        ink = cv2.bitwise_not(binary)
    else:
        ink = binary.copy()

    h, w = ink.shape
    h_proj    = np.sum(ink // 255, axis=1).astype(float)
    threshold = w * 0.02
    in_line   = False
    line_heights   = []
    line_densities = []
    line_start = 0

    for r in range(h):
        if h_proj[r] > threshold and not in_line:
            in_line    = True
            line_start = r
        elif h_proj[r] <= threshold and in_line:
            in_line = False
            lh = r - line_start
            if lh > 3:
                line_heights.append(lh)
                seg_ink = np.sum(ink[line_start:r, :] // 255)
                seg_px  = lh * w
                line_densities.append(seg_ink / seg_px if seg_px > 0 else 0.0)

    if not line_heights:
        return {
            'line_count': 0,
            'line_height_mean': 0.0,
            'line_height_cv': 1.0,
            'line_density_mean': 0.0,
            'line_spacing_cv': 1.0,
        }

    lh_mean = float(np.mean(line_heights))
    lh_std  = float(np.std(line_heights))
    lh_cv   = lh_std / lh_mean if lh_mean > 0 else 1.0
    ld_mean = float(np.mean(line_densities))

    # Line spacing consistency
    if len(line_heights) >= 2:
        spacings = [
            abs(line_heights[i] - line_heights[i-1])
            for i in range(1, len(line_heights))
        ]
        sp_mean = float(np.mean(spacings))
        sp_std  = float(np.std(spacings))
        sp_cv   = sp_std / sp_mean if sp_mean > 0 else 1.0
    else:
        sp_cv = 1.0

    return {
        'line_count':        len(line_heights),
        'line_height_mean':  round(lh_mean, 2),
        'line_height_cv':    round(lh_cv, 3),
        'line_density_mean': round(ld_mean, 4),
        'line_spacing_cv':   round(sp_cv, 3),
    }


def extract_slant_angle(binary: np.ndarray) -> float:
    """
    Estimate dominant writing slant angle using Hough line transform.
    Returns angle in [-1, 1] where 0 = vertical, + = right-leaning, - = left-leaning.
    This is one of the most discriminative writer-identity features.
    """
    try:
        if np.mean(binary) > 127:
            ink = cv2.bitwise_not(binary)
        else:
            ink = binary.copy()

        # Thin the strokes for better line detection
        kernel = np.ones((2, 1), np.uint8)
        thinned = cv2.erode(ink, kernel, iterations=1)

        lines = cv2.HoughLines(thinned, 1, np.pi / 180, threshold=30)
        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines[:50]:   # top 50 lines
            rho, theta = line[0]
            # Convert theta (0=horizontal Hough convention) to writing slant
            # theta near π/2 = vertical strokes (upright writing)
            angle_deg = np.degrees(theta) - 90
            # Only consider near-vertical strokes (writing strokes)
            if -45 < angle_deg < 45:
                angles.append(angle_deg)

        if not angles:
            return 0.0

        # Median slant (robust to outlier lines)
        median_slant = float(np.median(angles))
        # Normalize to [-1, 1]: most writers slant -30° to +30°
        return round(min(max(median_slant / 30.0, -1.0), 1.0), 3)
    except Exception:
        return 0.0


def extract_loop_density(binary: np.ndarray) -> float:
    """
    Estimate proportion of closed loops in strokes.
    Loopy/cursive writing has more closed contours per stroke count.
    Returns 0-1 normalized loop ratio.
    """
    try:
        if np.mean(binary) > 127:
            ink = cv2.bitwise_not(binary)
        else:
            ink = binary.copy()

        # All contours including holes
        contours_ext, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_all, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        n_ext = len([c for c in contours_ext if cv2.contourArea(c) > 4])
        n_all = len([c for c in contours_all if cv2.contourArea(c) > 4])

        if n_ext == 0:
            return 0.0

        # Ratio of inner holes to outer contours = loop density
        loop_ratio = min((n_all - n_ext) / n_ext, 1.0)
        return round(loop_ratio, 3)
    except Exception:
        return 0.0


def extract_baseline_deviation(binary: np.ndarray) -> float:
    """
    Measure how consistently strokes sit on their baseline.
    Low deviation = disciplined writing; high = irregular (also writer-specific).
    Returns CV of bottom-y coordinates of strokes, normalized 0-1.
    """
    try:
        if np.mean(binary) > 127:
            ink = cv2.bitwise_not(binary)
        else:
            ink = binary.copy()

        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bottoms = []
        for c in contours:
            if cv2.contourArea(c) < 10:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bottoms.append(y + h)   # bottom pixel of this stroke

        if len(bottoms) < 5:
            return 0.5

        bottoms_arr = np.array(bottoms, dtype=float)
        mean_b = float(np.mean(bottoms_arr))
        std_b  = float(np.std(bottoms_arr))
        cv_b   = std_b / mean_b if mean_b > 0 else 1.0
        # Normalize: cv > 0.15 is high deviation, < 0.03 is very precise
        return round(min(cv_b / 0.15, 1.0), 3)
    except Exception:
        return 0.5


def compute_pressure_proxy(cv2_img: np.ndarray, binary: np.ndarray) -> float:
    """
    Estimate writing pressure from stroke darkness.
    Returns normalized pressure score 0-1.
    """
    try:
        gray = (
            cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            if len(cv2_img.shape) == 3
            else cv2_img
        )
        ink_mask   = (binary == 0) if np.mean(binary) > 127 else (binary == 255)
        ink_pixels = gray[ink_mask]
        if len(ink_pixels) == 0:
            return 0.5
        mean_darkness = 1.0 - float(np.mean(ink_pixels)) / 255.0
        return round(min(max(mean_darkness, 0.0), 1.0), 3)
    except Exception:
        return 0.5


def build_handwriting_fingerprint(
    stroke_feats: dict,
    line_feats: dict,
    pressure: float,
    slant: float,
    loop_density: float,
    baseline_dev: float,
) -> np.ndarray:
    """
    Create a normalized 18-feature writer-identity vector.

    Features split into:
      GROUP A — writer-identity (stable across topics, discriminative between writers)
        slant, aspect_ratio, thinness, compactness, loop_density,
        gap_to_width_ratio, baseline_dev, line_spacing_cv
      GROUP B — style/pressure (less discriminative, included with lower weight)
        stroke_density, height_cv, width_cv, gap_cv, line_height_cv,
        line_density_mean, pressure, area_mean_norm, area_std_norm, stroke_count_norm

    The vector is L2-normalized so cosine similarity == dot product.
    """
    # GROUP A — identity features (given 1.5x weight by repeating scaled)
    identity = [
        slant,                                              # writing slant angle
        stroke_feats.get('mean_aspect', 1.0),              # letter aspect ratio
        stroke_feats.get('thinness', 0.0),                 # stroke thinness
        stroke_feats.get('compactness', 0.0),              # roundness
        loop_density,                                       # cursive loop ratio
        stroke_feats.get('gap_to_width_ratio', 0.5),       # spacing style
        baseline_dev,                                       # baseline consistency
        line_feats.get('line_spacing_cv', 1.0),            # line spacing regularity
    ]

    # GROUP B — style features (standard weight)
    style = [
        stroke_feats.get('stroke_density', 0.0),
        stroke_feats.get('height_cv', 1.0),
        stroke_feats.get('width_cv', 1.0),
        stroke_feats.get('gap_cv', 1.0),
        line_feats.get('line_height_cv', 1.0),
        line_feats.get('line_density_mean', 0.0),
        pressure,
        min(stroke_feats.get('area_mean', 0.0) / 500.0, 1.0),
        min(stroke_feats.get('area_std',  0.0) / 500.0, 1.0),
        min(stroke_feats.get('stroke_count', 0) / 200.0, 1.0),
    ]

    # Weight identity features 1.5x
    weighted = [v * 1.5 for v in identity] + style

    vec  = np.array(weighted, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return 0.0
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_consistency_score(stroke_feats: dict, line_feats: dict) -> float:
    """
    Overall handwriting consistency score (0-1) for SCORING purposes only.
    Not used for identity comparison.
    """
    h_cv  = stroke_feats.get('height_cv', 1.0)
    w_cv  = stroke_feats.get('width_cv', 1.0)
    g_cv  = stroke_feats.get('gap_cv', 1.0)
    lh_cv = line_feats.get('line_height_cv', 1.0)

    def cv_to_score(cv: float) -> float:
        return max(0.4, 1.0 - min(cv, 1.0) * 0.6)

    consistency = (
        cv_to_score(h_cv)  * 0.35 +
        cv_to_score(w_cv)  * 0.25 +
        cv_to_score(g_cv)  * 0.25 +
        cv_to_score(lh_cv) * 0.15
    )

    if stroke_feats.get('stroke_count', 0) < 20:
        consistency *= 0.7

    return round(min(max(consistency, 0.1), 1.0), 3)


def analyze_handwriting(
    cv2_pages: list,
    binary_pages: list,
) -> dict:
    """
    Full handwriting analysis across all pages.
    cv2_pages    : list of preprocessed BGR images
    binary_pages : list of binarized images
    Returns comprehensive analysis dict including 18-feature fingerprint.
    """
    all_stroke       = []
    all_line         = []
    all_pressure     = []
    all_slant        = []
    all_loop         = []
    all_baseline     = []
    all_fingerprints = []

    for cv2_img, binary in zip(cv2_pages, binary_pages):
        sf      = extract_stroke_features(binary)
        lf      = extract_line_features(binary)
        pr      = compute_pressure_proxy(cv2_img, binary)
        slant   = extract_slant_angle(binary)
        loops   = extract_loop_density(binary)
        baseln  = extract_baseline_deviation(binary)
        fp      = build_handwriting_fingerprint(sf, lf, pr, slant, loops, baseln)

        all_stroke.append(sf)
        all_line.append(lf)
        all_pressure.append(pr)
        all_slant.append(slant)
        all_loop.append(loops)
        all_baseline.append(baseln)
        all_fingerprints.append(fp)

    n_features = 18   # 8 identity * 1.5 + 10 style = 22 values in vector

    if not all_stroke:
        return {
            'consistency':     0.5,
            'pressure':        0.5,
            'stroke_density':  0.0,
            'stroke_count':    0,
            'line_count':      0,
            'height_cv':       1.0,
            'gap_cv':          1.0,
            'compactness':     0.0,
            'slant':           0.0,
            'loop_density':    0.0,
            'fingerprint':     [0.0] * 18,
            'similarity_flag': False,
        }

    def avg_dict(dicts: list, key: str) -> float:
        vals = [d.get(key, 0) for d in dicts if d.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    avg_stroke   = {k: avg_dict(all_stroke, k) for k in all_stroke[0]}
    avg_line     = {k: avg_dict(all_line, k)   for k in all_line[0]}
    avg_pressure = round(sum(all_pressure) / len(all_pressure), 3)
    avg_slant    = round(sum(all_slant)    / len(all_slant), 3)
    avg_loop     = round(sum(all_loop)     / len(all_loop), 3)

    fp_stack = np.stack(all_fingerprints)
    mean_fp  = np.mean(fp_stack, axis=0)
    norm_fp  = mean_fp / (np.linalg.norm(mean_fp) + 1e-8)

    consistency = compute_consistency_score(avg_stroke, avg_line)

    return {
        'consistency':     consistency,
        'pressure':        avg_pressure,
        'stroke_density':  avg_stroke.get('stroke_density', 0),
        'stroke_count':    int(avg_stroke.get('stroke_count', 0)),
        'line_count':      int(avg_line.get('line_count', 0)),
        'height_cv':       avg_stroke.get('height_cv', 1.0),
        'gap_cv':          avg_stroke.get('gap_cv', 1.0),
        'compactness':     avg_stroke.get('compactness', 0.0),
        'slant':           avg_slant,
        'loop_density':    avg_loop,
        'fingerprint':     norm_fp.tolist(),
        'similarity_flag': False,
    }


def compare_handwritings(fp_a: list, fp_b: list) -> float:
    """
    Compare two 18-feature handwriting fingerprints.
    Returns similarity 0-1.

    THRESHOLD GUIDANCE (v3.0, 18-feature identity vector):
      >= 0.92 : very likely same writer
      0.85-0.92: possibly same writer — flag for review
      < 0.85  : different writers (do NOT flag)

    Old 9-feature vector had threshold ~0.75 which caused mass false positives
    because style features (pressure, density) overlap across students writing
    the same assignment with the same pen on the same paper.
    """
    if not fp_a or not fp_b:
        return 0.0
    va = np.array(fp_a, dtype=np.float32)
    vb = np.array(fp_b, dtype=np.float32)
    if len(va) != len(vb):
        return 0.0
    return round(float(cosine_similarity(va, vb)), 3)