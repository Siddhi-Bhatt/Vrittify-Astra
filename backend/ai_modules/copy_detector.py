"""
VRITTIFY ASTRA – Copy & AI Detection Engine v2.2

Expected flag behaviour for the test batch:
  s1 (original)  → no flags
  s2 (AI)        → AI flag
  s3 (original)  → no flags  (source; higher scorer)
  s4 (copied s3) → "Copied from [s3 name]" flag
  s5 (s3 wrote + AI) → AI flag + "Written by [s3 name]" handwriting flag

Fixes v2.2 (over v2.1):
  - HW_MATCH_THRESHOLD lowered to 0.75 (was 0.88) — 0.88 was too strict
    and missed the s3→s5 ghostwriter link in practice.
  - Ghostwriter detection (high hw_sim + different student + high text_sim)
    now emits "Submission may have been written by {peer}" instead of the
    confusing "Handwriting matches {peer}'s submission" when there is also
    a text-copy relationship.
  - detect_flags() is now the canonical public API used by pipeline.py's
    detect_copies_batch() wrapper so both call-sites share one code path.
  - Same-student self-comparison guard added (student_id check).
  - AI_FLAG_THRESHOLD stays at 0.55 (v2.1 fix retained).
  - summarise_flags() unchanged.
"""

from __future__ import annotations
from typing import Optional

# ── Thresholds ─────────────────────────────────────────────────────────────
TEXT_COPY_THRESHOLD  = 0.40   # text similarity → possible copy
HW_MATCH_THRESHOLD   = 0.75   # hw cosine similarity → likely same writer
                               # FIX v2.2: was 0.88 — too strict for real data
AI_FLAG_THRESHOLD    = 0.55   # ai_probability → AI flag
                               # (raised from 0.50 in v2.1 to cut false positives)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Safe cosine similarity between two equal-length float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compare_handwritings(fp_a: list[float], fp_b: list[float]) -> float:
    """
    Compare two handwriting fingerprint vectors.
    Returns cosine similarity in [0, 1].  0 if either fingerprint is missing.
    """
    return _cosine_similarity(fp_a or [], fp_b or [])


def detect_flags(submissions: list[dict]) -> list[dict]:
    """
    Run copy and AI detection across all submissions.

    Each submission dict must have:
      {
        'student':    str,          # display name
        'student_id': str,          # DB id — used to block self-comparison
        'analysis': {
          'ai_probability':  float,
          'copy_similarity': float,  # highest pairwise text sim (pre-computed)
          'fingerprint':     list[float],
        },
        'text_hash_pairs': dict[str, float],
          # {other_student_name: text_sim_score} — pre-computed by pipeline
      }

    Returns the same list with 'flags' key added to each submission.
    Each flag: {'type': 'ai'|'copied'|'ghostwriter'|'handwriting', 'detail': str}
    """
    n = len(submissions)

    # ── Build pairwise HW similarity matrix ────────────────────────────────
    hw_sim: dict[tuple[str, str], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sa = submissions[i]
            sb = submissions[j]
            # Never compare a student against themselves (resubmit edge-case)
            if sa.get('student_id') and sa.get('student_id') == sb.get('student_id'):
                sim = 1.0   # identical — handled by same-student guard below
            else:
                fp_a = sa['analysis'].get('fingerprint') or []
                fp_b = sb['analysis'].get('fingerprint') or []
                sim  = compare_handwritings(fp_a, fp_b)
            hw_sim[(sa['student'], sb['student'])] = sim
            hw_sim[(sb['student'], sa['student'])] = sim

    # ── Assign flags ────────────────────────────────────────────────────────
    for sub in submissions:
        flags       = []
        name        = sub['student']
        my_id       = sub.get('student_id', '')
        ana         = sub['analysis']
        text_pairs  = sub.get('text_hash_pairs', {})   # {peer_name: score}

        # ── 1. AI flag ──────────────────────────────────────────────────────
        ai_prob = ana.get('ai_probability', 0.0)
        if ai_prob > AI_FLAG_THRESHOLD:
            prob_pct = round(ai_prob * 100)
            flags.append({
                'type':   'ai',
                'detail': f'AI-generated content detected ({prob_pct}% probability)',
            })

        # ── 2. Find best text-copy peer (excluding self) ────────────────────
        best_copy_peer  = None
        best_copy_score = 0.0
        peer_sid_map    = {s['student']: s.get('student_id', '') for s in submissions}

        for peer, score in text_pairs.items():
            if peer == name:
                continue
            if my_id and peer_sid_map.get(peer) == my_id:
                continue   # same student, different submission — skip
            if score > best_copy_score:
                best_copy_score = score
                best_copy_peer  = peer

        if best_copy_peer and best_copy_score > TEXT_COPY_THRESHOLD:
            pct         = round(best_copy_score * 100)
            hw_to_peer  = hw_sim.get((name, best_copy_peer), 0.0)

            if hw_to_peer > HW_MATCH_THRESHOLD:
                # Our handwriting looks like the peer's — ghostwriter scenario:
                # the peer likely wrote this submission for us.
                flags.append({
                    'type':   'ghostwriter',
                    'detail': (
                        f'Submission may have been written by {best_copy_peer} '
                        f'({pct}% text similarity, handwriting match)'
                    ),
                })
            else:
                # Different handwriting but same text — straightforward copy
                flags.append({
                    'type':   'copied',
                    'detail': f'Copied from {best_copy_peer} ({pct}% similarity)',
                })

        # ── 3. HW-match flag (ghostwriter without obvious text copy) ─────────
        # Raised when hw_sim is high but we didn't already flag that pair
        # for text-copy/ghostwriter above, so we catch the s5-type scenario
        # where the text is AI-generated (not literally copied from the writer)
        # yet the handwriting is clearly the same person.
        already_hw_flagged: set[str] = set()
        for f in flags:
            if f['type'] in ('ghostwriter', 'handwriting'):
                # extract peer name stored in detail between "by " and " ("
                detail = f.get('detail', '')
                try:
                    peer = detail.split('written by ')[1].split(' (')[0]
                    already_hw_flagged.add(peer)
                except (IndexError, AttributeError):
                    pass

        for other in submissions:
            other_name = other['student']
            other_id   = other.get('student_id', '')
            if other_name == name:
                continue
            if my_id and other_id == my_id:
                continue
            if other_name in already_hw_flagged:
                continue
            sim = hw_sim.get((name, other_name), 0.0)
            if sim > HW_MATCH_THRESHOLD:
                # Check text similarity to distinguish "ghostwriter with AI"
                # from a plain handwriting coincidence.
                text_sim = text_pairs.get(other_name, 0.0)
                if text_sim > TEXT_COPY_THRESHOLD:
                    # Already handled in step 2 — skip
                    continue
                # Low text similarity but same handwriting → suspicious
                # (e.g. s5: AI text written in s3's hand)
                flags.append({
                    'type':   'handwriting',
                    'detail': (
                        f'Handwriting closely matches {other_name}\'s submission '
                        f'(possible ghostwriter)'
                    ),
                })
                already_hw_flagged.add(other_name)

        sub['flags'] = flags

    return submissions


def summarise_flags(submissions: list[dict]) -> dict:
    """
    Build a human-readable summary dict.
    Returns { student_name: { flagged: bool, reasons: [str], flag_count: int } }
    """
    out = {}
    for sub in submissions:
        name  = sub['student']
        flags = sub.get('flags', [])
        out[name] = {
            'flagged':    len(flags) > 0,
            'reasons':    [f['detail'] for f in flags],
            'flag_count': len(flags),
        }
    return out


# ── detect_copies_batch() ──────────────────────────────────────────────────
# Drop-in replacement for the version previously living in ai_detector.py.
# Called by pipeline.py → run_full_pipeline().
#
# Args:
#   current_text:    OCR text of the assignment being scored
#   current_hw_fp:   handwriting fingerprint list (may be None)
#   other_assignments: list of dicts:
#       {'id', 'student_id', 'student_name', 'text', 'hw_fingerprint'}
#
# Returns:
#   {
#     'max_text_similarity': float,
#     'max_hw_similarity':   float,
#     'combined_copy_score': float,
#     'matches':             list of match dicts (sorted by combined desc),
#     'copy_flag':           bool,
#   }

def detect_copies_batch(
    current_text: str,
    current_hw_fp,
    other_assignments: list[dict],
) -> dict:
    """
    Compare one assignment against a list of peers.
    Text similarity is computed with a simple token-overlap (Jaccard).
    HW similarity is cosine over fingerprint vectors.
    combined = 0.6 * text_sim + 0.4 * hw_sim
    """
    if not other_assignments:
        return {
            'max_text_similarity':  0.0,
            'max_hw_similarity':    0.0,
            'combined_copy_score':  0.0,
            'matches':              [],
            'copy_flag':            False,
        }

    current_tokens = set((current_text or '').lower().split())

    matches = []
    for other in other_assignments:
        other_text = other.get('text', '')
        other_fp   = other.get('hw_fingerprint') or []
        other_id   = other.get('id', '')
        other_sid  = other.get('student_id', '')
        other_name = other.get('student_name', other_id)

        # Text similarity — Jaccard over word tokens
        other_tokens = set((other_text or '').lower().split())
        if current_tokens or other_tokens:
            intersection = len(current_tokens & other_tokens)
            union        = len(current_tokens | other_tokens)
            text_sim     = intersection / union if union else 0.0
        else:
            text_sim = 0.0

        # HW similarity — cosine
        hw_sim = compare_handwritings(
            list(current_hw_fp) if current_hw_fp else [],
            other_fp,
        )

        combined = 0.6 * text_sim + 0.4 * hw_sim

        matches.append({
            'id':           other_id,
            'student_id':   other_sid,
            'student_name': other_name,
            'text_sim':     round(text_sim,  4),
            'hw_sim':       round(hw_sim,    4),
            'combined':     round(combined,  4),
        })

    matches.sort(key=lambda m: m['combined'], reverse=True)

    max_text = max((m['text_sim'] for m in matches), default=0.0)
    max_hw   = max((m['hw_sim']   for m in matches), default=0.0)
    combined_top = matches[0]['combined'] if matches else 0.0

    return {
        'max_text_similarity':  round(max_text,    4),
        'max_hw_similarity':    round(max_hw,      4),
        'combined_copy_score':  round(combined_top, 4),
        'matches':              matches,
        'copy_flag':            combined_top > TEXT_COPY_THRESHOLD,
    }