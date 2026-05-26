"""
VRITTIFY ASTRA – AI Pipeline Orchestrator v3.3
Ties all modules together: preprocess → OCR → stylometry → handwriting → AI detect → copy → score

Fixes v3.3 (over v3.2):
  - detect_copies_batch() now imported from ai_modules.copy_detector (not
    ai_modules.ai_detector).  copy_detector is the single source of truth
    for all copy / ghostwriter / HW-match logic.
  - 'student_id' field is forwarded through other_assignments dicts so
    copy_detector can block same-student self-comparisons.
  - analysis_summary exposes 'max_text_similarity' and 'max_hw_similarity'
    separately so app.py can apply independent thresholds.
  - build_flags() in scorer receives copy_score=0.0 and
    hw_similarity_flag=False — copy / ghostwriter / HW flags are built
    entirely in app.py with real peer names (v3.2 design retained).
"""

import os
import sys

_AI_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
if _AI_MODULES_DIR not in sys.path:
    sys.path.insert(0, _AI_MODULES_DIR)

import json
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('VSTRA.pipeline')


def run_full_pipeline(
    pdf_path: str,
    assignment_id: str,
    other_assignments: list = None,
    processed_dir: str = None,
) -> dict:
    """
    Full AI analysis pipeline for one assignment PDF.

    Args:
        pdf_path:          Absolute path to the uploaded PDF.
        assignment_id:     DB ID of this assignment.
        other_assignments: List of peer dicts for copy detection:
                           [{'id', 'student_id', 'student_name', 'text',
                             'hw_fingerprint'}, ...]
                           'student_id' is required for same-student exclusion.
        processed_dir:     Optional dir to save preprocessed page images.

    Returns:
        {
          'ocr':          dict
          'stylometry':   dict
          'handwriting':  dict
          'ai_detection': dict
          'copy':         dict
          'analysis':     dict  ← includes max_text_similarity, max_hw_similarity
          'flags':        list  ← NO copy/ghostwriter/HW flags (built in app.py)
          'score':        float
          'feedback':     str
          'grade':        str
        }
    """
    other_assignments = other_assignments or []

    # ── STEP 1: Preprocess ──────────────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 1: Preprocessing PDF")
    try:
        from ai_modules.preprocessor import preprocess_pdf, get_page_metrics
        pages        = preprocess_pdf(pdf_path, output_dir=processed_dir)
        cv2_pages    = [p[0] for p in pages]
        pil_pages    = [p[1] for p in pages]
        binary_pages = [p[2] for p in pages]
        page_metrics = [get_page_metrics(cv, bi) for cv, bi in zip(cv2_pages, binary_pages)]
        log.info(f"[{assignment_id}] Preprocessed {len(pages)} pages")
    except Exception as e:
        log.warning(f"[{assignment_id}] Preprocessing failed: {e}. Using fallback.")
        cv2_pages = binary_pages = pil_pages = []
        page_metrics = []

    # ── STEP 2: OCR ─────────────────────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 2: OCR extraction")
    try:
        from ai_modules.ocr_engine import extract_from_all_pages, clean_ocr_text, get_text_stats
        ocr_result   = extract_from_all_pages(pil_pages) if pil_pages else {
            'text': '', 'avg_confidence': 0.0, 'word_count': 0, 'page_count': 0, 'words': []
        }
        cleaned_text = clean_ocr_text(ocr_result['text'])
        text_stats   = get_text_stats(cleaned_text)
        ocr_result['cleaned_text'] = cleaned_text
        ocr_result['stats']        = text_stats
        log.info(f"[{assignment_id}] OCR: {ocr_result.get('word_count', 0)} words, "
                 f"conf={ocr_result.get('avg_confidence', 0):.2f}")
    except Exception as e:
        log.warning(f"[{assignment_id}] OCR failed: {e}")
        cleaned_text = ""
        ocr_result = {
            'text': '', 'cleaned_text': '', 'avg_confidence': 0.0,
            'word_count': 0, 'stats': {}
        }

    # ── STEP 3: Stylometry ──────────────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 3: Stylometry analysis")
    try:
        from ai_modules.stylometry import analyze_stylometry
        style_result = analyze_stylometry(cleaned_text)
        log.info(f"[{assignment_id}] Style score: {style_result.get('style_score', 0):.3f}")
    except Exception as e:
        log.warning(f"[{assignment_id}] Stylometry failed: {e}")
        style_result = {'style_score': 0.5, 'ai_markers': {}}

    # ── STEP 4: Handwriting Analysis ────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 4: Handwriting analysis")
    try:
        from ai_modules.handwriting_analyzer import analyze_handwriting
        hw_result = analyze_handwriting(cv2_pages, binary_pages) if cv2_pages else {
            'consistency': 0.6, 'pressure': 0.5, 'stroke_density': 0.0,
            'stroke_count': 0, 'line_count': 0, 'fingerprint': None,
            'similarity_flag': False,
        }
        log.info(f"[{assignment_id}] HW consistency: {hw_result.get('consistency', 0):.3f}")
    except Exception as e:
        log.warning(f"[{assignment_id}] Handwriting analysis failed: {e}")
        hw_result = {
            'consistency': 0.6, 'pressure': 0.5, 'fingerprint': None,
            'similarity_flag': False
        }

    # ── STEP 5: AI Detection ────────────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 5: AI content detection")
    try:
        from ai_modules.ai_detector import detect_ai_probability
        ai_result = detect_ai_probability(cleaned_text, style_result)
        log.info(f"[{assignment_id}] AI probability: {ai_result.get('probability', 0):.3f}")
    except Exception as e:
        log.warning(f"[{assignment_id}] AI detection failed: {e}")
        ai_result = {'probability': 0.1, 'signals': {}, 'confidence': 'low'}

    # ── STEP 6: Copy Detection ───────────────────────────────────────────────
    # *** FIX v3.3: import from copy_detector, not ai_detector ***
    log.info(f"[{assignment_id}] Step 6: Copy detection against "
             f"{len(other_assignments)} assignments")
    try:
        from ai_modules.copy_detector import detect_copies_batch
        copy_result = detect_copies_batch(
            current_text   = cleaned_text,
            current_hw_fp  = hw_result.get('fingerprint'),
            other_assignments = other_assignments,
            # each dict must contain 'student_id' so same-student exclusion works
        )
        log.info(
            f"[{assignment_id}] Copy score: "
            f"{copy_result.get('combined_copy_score', 0):.3f} "
            f"| text={copy_result.get('max_text_similarity', 0):.3f} "
            f"| hw={copy_result.get('max_hw_similarity', 0):.3f}"
        )
    except Exception as e:
        log.warning(f"[{assignment_id}] Copy detection failed: {e}")
        copy_result = {
            'max_text_similarity':  0.0,
            'max_hw_similarity':    0.0,
            'combined_copy_score':  0.0,
            'matches':              [],
            'copy_flag':            False,
        }

    # ── STEP 7: Scoring ──────────────────────────────────────────────────────
    log.info(f"[{assignment_id}] Step 7: Score generation")
    from ai_modules.scorer import (
        compute_originality, compute_final_score,
        build_flags, generate_feedback, get_grade_label,
    )

    ai_prob  = ai_result.get('probability', 0.1)
    copy_sim = copy_result.get('combined_copy_score', 0.0)
    hw_cons  = hw_result.get('consistency', 0.6)
    style_sc = style_result.get('style_score', 0.5)
    ocr_conf = ocr_result.get('avg_confidence', 0.5)

    originality = compute_originality(ai_prob, copy_sim)
    score       = compute_final_score(
        originality, ai_prob, copy_sim, hw_cons, style_sc, ocr_conf
    )

    # build_flags() emits AI, style, and legibility flags only.
    # Copy / ghostwriter / HW-match flags are built in app.py with real peer
    # names. Pass copy_score=0.0 and hw_similarity_flag=False here.
    flags = build_flags(
        ai_probability     = ai_prob,
        copy_score         = 0.0,    # intentional — handled in app.py
        hw_similarity_flag = False,  # intentional — handled in app.py
        style_score        = style_sc,
        ocr_confidence     = ocr_conf,
    )

    analysis_summary = {
        'originality':         originality,
        'ai_probability':      ai_prob,
        'copy_similarity':     copy_sim,
        'max_text_similarity': copy_result.get('max_text_similarity', 0.0),
        'max_hw_similarity':   copy_result.get('max_hw_similarity',   0.0),
        'hw_consistency':      hw_cons,
        'style_score':         style_sc,
        'ocr_confidence':      ocr_conf,
        'word_count':          ocr_result.get('word_count', 0),
        'hw_pressure':         hw_result.get('pressure', 0.5),
        'hw_stroke_density':   hw_result.get('stroke_density', 0.0),
        'hw_fingerprint':      hw_result.get('fingerprint'),
        'copy_matches':        copy_result.get('matches', []),
        'ai_signals':          ai_result.get('signals', {}),
    }

    feedback = generate_feedback(score, flags, analysis_summary)
    grade    = get_grade_label(score)

    log.info(
        f"[{assignment_id}] FINAL SCORE: {score}/10 | "
        f"Grade: {grade} | Flags: {len(flags)}"
    )

    return {
        'ocr':          ocr_result,
        'stylometry':   style_result,
        'handwriting':  hw_result,
        'ai_detection': ai_result,
        'copy':         copy_result,
        'analysis':     analysis_summary,
        'flags':        flags,
        'score':        score,
        'grade':        grade,
        'feedback':     feedback,
    }