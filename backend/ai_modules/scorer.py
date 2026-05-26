"""
VRITTIFY ASTRA – Scoring Engine
Computes final score (0-10) and generates human-readable feedback.

v3.2 fixes:
  - build_flags() no longer emits copy-detection flags.
    Copy flags are now built exclusively in app.py where actual peer
    names, scores, and directional logic are available.
    This eliminates generic "Copied from another student" strings and
    the fragile post-hoc string-replacement that caused random peer names.
"""


def compute_final_score(
    originality: float,
    ai_probability: float,
    copy_similarity: float,
    hw_consistency: float,
    style_score: float,
    ocr_confidence: float = 0.7,
) -> float:
    """
    Weighted scoring formula. All inputs are 0-1.
    Returns score 0.0 - 10.0 (rounded to 1 decimal).

    Weights:
        originality    35% – core academic integrity
        ai safety      25% – penalize AI-generated content
        copy safety    20% – penalize copied content
        hw consistency 10% – penalize inconsistent handwriting
        style score    10% – authenticity of writing style
    """
    raw = (
        originality              * 0.35 +
        (1 - ai_probability)     * 0.25 +
        (1 - copy_similarity)    * 0.20 +
        hw_consistency           * 0.10 +
        style_score              * 0.10
    )
    # Small bonus for high OCR confidence (legible handwriting)
    legibility_bonus = max(0.0, (ocr_confidence - 0.5) * 0.04)
    raw += legibility_bonus

    score = round(raw * 10, 1)
    return max(0.0, min(10.0, score))


def compute_originality(ai_prob: float, copy_sim: float) -> float:
    """Derive originality from AI and copy penalties."""
    orig = 1.0 - (copy_sim * 0.65 + ai_prob * 0.35)
    return round(max(0.0, min(1.0, orig)), 3)


def build_flags(
    ai_probability: float,
    copy_score: float,          # kept for signature compatibility; NOT used for copy flags
    hw_similarity_flag: bool,
    style_score: float,
    ocr_confidence: float,
) -> list:
    """
    Generate flag strings based on detection thresholds.

    NOTE: Copy-detection flags are intentionally NOT emitted here.
    They are built in app.py (scan_assignments) where the real peer
    names, per-pair similarity scores, and directional logic are known.
    Emitting generic copy strings here and patching them later was
    fragile and produced wrong peer names when scores were close.
    """
    flags = []

    # ── AI detection ────────────────────────────────────────────────────────
    if ai_probability >= 0.40:
        flags.append("AI-generated content detected")
    elif ai_probability >= 0.28:
        flags.append("Possible AI assistance detected")

    # ── Handwriting identity ─────────────────────────────────────────────────
    if hw_similarity_flag:
        flags.append("Handwriting matches another student's submission")

    # ── Writing style ────────────────────────────────────────────────────────
    if style_score < 0.25:
        flags.append("Highly inconsistent writing style detected")

    # ── Legibility ───────────────────────────────────────────────────────────
    if 0 < ocr_confidence < 0.30:
        flags.append("Low legibility – assignment hard to read")

    return flags


def generate_feedback(
    score: float,
    flags: list,
    analysis: dict,
) -> str:
    """
    Generate detailed, personalised feedback text.
    """
    parts = []

    # Overall assessment
    if score >= 9.0:
        parts.append("Outstanding work! This assignment demonstrates exceptional originality, "
                     "consistent handwriting, and authentic student effort.")
    elif score >= 8.0:
        parts.append("Excellent submission. The work shows strong originality and consistent writing quality.")
    elif score >= 7.0:
        parts.append("Good work overall. The assignment demonstrates genuine effort with minor areas for improvement.")
    elif score >= 6.0:
        parts.append("Satisfactory submission. The work shows effort, but there are notable areas requiring attention.")
    elif score >= 5.0:
        parts.append("Average performance. The assignment has several concerns that need to be addressed.")
    elif score >= 3.0:
        parts.append("Below average. This submission has significant issues that affect its integrity score.")
    else:
        parts.append("This submission requires serious review. Multiple integrity concerns have been detected.")

    # AI detection feedback
    ai_prob = analysis.get('ai_probability', 0)
    if ai_prob >= 0.40:
        parts.append(
            f"⚠️ AI Detection: High likelihood ({round(ai_prob*100)}%) of AI-generated content. "
            "Writing patterns strongly match AI-generated text. Please ensure all work is original."
        )
    elif ai_prob >= 0.28:
        parts.append(
            f"⚠️ AI Detection: Possible AI assistance detected ({round(ai_prob*100)}%). "
            "Some phrasing appears AI-generated. Please review and rewrite in your own words."
        )
    elif ai_prob < 0.20:
        parts.append("✅ Writing appears genuinely student-authored with natural human patterns.")

    # Copy detection feedback — uses named flag strings built in app.py
    copy_sim = analysis.get('copy_similarity', 0)
    if copy_sim >= 0.55:
        parts.append(
            f"🚩 Copy Detection: High similarity ({round(copy_sim*100)}%) found with another submission. "
            "Academic dishonesty policy applies."
        )
    elif copy_sim >= 0.40:
        parts.append(
            f"⚠️ Copy Detection: Moderate similarity ({round(copy_sim*100)}%) with another submission. "
            "Please review and ensure independent work."
        )
    elif copy_sim >= 0.28:
        parts.append(
            f"📋 Copy Detection: Some similarity ({round(copy_sim*100)}%) detected. "
            "This may be coincidental for assignments on the same topic."
        )

    # Handwriting consistency
    hw_cons = analysis.get('hw_consistency', 0)
    if hw_cons >= 0.85:
        parts.append("✅ Handwriting is very consistent and legible throughout.")
    elif hw_cons >= 0.65:
        parts.append("📝 Handwriting shows reasonable consistency with minor variations.")
    elif hw_cons < 0.50:
        parts.append(
            "⚠️ Handwriting consistency is low. Significant style variations detected across pages "
            "which may indicate multiple authors."
        )

    # Originality
    orig = analysis.get('originality', 0)
    if orig >= 0.80:
        parts.append("✅ High originality score – work appears genuinely independent.")
    elif orig < 0.50:
        parts.append("⚠️ Originality score is below acceptable threshold.")

    # OCR / legibility
    ocr_conf = analysis.get('ocr_confidence', 0)
    if ocr_conf > 0:
        if ocr_conf < 0.35:
            parts.append("📄 Note: Handwriting legibility is low. Consider writing more clearly for better OCR accuracy.")
        elif ocr_conf > 0.75:
            parts.append("✅ Handwriting is clear and easily readable.")

    return " ".join(parts)


def get_grade_label(score: float) -> str:
    """Map score to grade label."""
    if score >= 9.0: return "A+"
    if score >= 8.0: return "A"
    if score >= 7.0: return "B+"
    if score >= 6.0: return "B"
    if score >= 5.0: return "C"
    if score >= 4.0: return "D"
    return "F"


def get_score_label(score: float) -> str:
    if score >= 8: return "Excellent"
    if score >= 6: return "Good"
    if score >= 5: return "Average"
    if score >= 3: return "Below Average"
    return "Needs Improvement"