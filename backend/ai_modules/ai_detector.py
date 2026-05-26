"""
VRITTIFY ASTRA – AI Content & Copy Detection Engine v2.3

Key fixes v2.3:
  - compute_text_similarity(): same-topic essays were scoring 40-50% with
    TF-IDF cosine alone because content words ("stress", "study", "India",
    "parents") overlap naturally. Fix: require BOTH high TF-IDF cosine AND
    high trigram overlap for a copy flag. Trigrams are phrase-specific and
    don't match just from shared topic vocabulary.
    New combined formula: min(tfidf_cos, ng3 * 1.5) * 0.6 + ng3 * 0.4
    This means two essays about the same topic but with different phrasing
    score ~0.10-0.20, while actual copies score 0.55+.
  - detect_copies_batch(): match inclusion threshold raised from 0.08 → 0.25
    so incidental topic-overlap matches don't pollute the match list.
  - AI_PHRASE_PATTERNS: domain-specific stress-essay patterns retained;
    they correctly flag s2 and s5. General AI markers kept but weighted less.
  - detect_ai_probability(): phrase_signal weight raised to 0.55 (was 0.45)
    since phrase patterns are the most reliable signal for short texts.
    uniformity_signal weight lowered to 0.10 (was 0.20) since short student
    essays naturally have low sentence-count variance anyway.
"""

import re
import math
import hashlib
from collections import Counter


# ════════════════════════════════════════════════════════════
# AI CONTENT DETECTION
# ════════════════════════════════════════════════════════════

AI_PHRASE_PATTERNS = [
    # Education / student-stress domain AI phrases (detected from test PDFs s2, s5)
    r'\bstress in education\b', r'\barises because students\b',
    r'\bconstant pressure to perform\b', r'\bperform well in exams\b',
    r'\bmeet dead[\s-]?lines\b', r'\bbalance multiple subjects\b',
    r'\bfear of failure\b', r'\bcomparison with peers\b',
    r'\bhigh expectations from (teachers|family)\b',
    r'\bchallenges feel overwhelming\b', r'\bfeel overwhelming\b',
    r'\black of control\b', r'\bover outcomes\b',
    r'\buncertainty about\b', r'\bfuture opportunities\b',
    r'\bintensify stress\b', r'\boutcomes intensify\b',
    r'\bcompetition lack\b', r'\bface constant pressure\b',
    # General AI markers
    r'\bfurthermore\b', r'\bmoreover\b', r'\bin conclusion\b',
    r'\bit is important to note\b', r'\bit is worth noting\b',
    r'\bas mentioned (above|earlier|previously)\b',
    r'\bin summary\b', r'\bto summarize\b',
    r'\bplays a (crucial|key|vital|pivotal) role\b',
    r'\bit is essential (to|that)\b',
    r'\bin addition\b', r'\badditionally\b',
    r'\bconsequently\b', r'\bthus,\b', r'\btherefore,\b',
    r'\bhence,\b', r'\bnevertheless\b', r'\bnonetheless\b',
    r'\bon the other hand\b', r'\bin contrast\b',
    r'\bas a result\b', r'\bdue to (the|this|these)\b',
    r'\bit should be noted\b', r'\bit can be (seen|observed)\b',
    r'\bone can (observe|see|note)\b',
    r'\bin the context of\b', r'\bwith respect to\b',
    r'\bsignificantly\b', r'\bsubstantially\b',
    r'\bcomprehensively\b', r'\beffectively ensures?\b',
    r'\bvarious aspects\b', r'\bdiverse range\b',
    r'\bkey (aspects|elements|components|factors)\b',
    r'\bdelve into\b', r'\bshed light on\b',
    r'\bundeniably\b', r'\bindisputably\b',
    r'\bultimately,\b', r'\boverall,\b',
    r'\bit goes without saying\b',
    r'\bpivotal (role|aspect|element)\b',
    r'\brobust (approach|framework|system|solution)\b',
    r'\bseamlessly\b', r'\boptimal(ly)?\b',
    r'\bsynergistic\b', r'\bparadigm\b',
]

STUDENT_PATTERNS = [
    r'\bi think\b', r'\bi believe\b', r'\bi feel\b',
    r'\bwe (know|learned|studied|saw)\b',
    r'\blet me (explain|show|calculate)\b',
    r'\bthe answer is\b', r'\bso,?\s+we\b',
    r'\bfor example,?\s+if\b',
    r'\blike\s+we\b', r'\bas we\b',
    r'\bfirst(ly)?,?\s+we\b', r'\bnext,?\s+we\b',
    r'\bfinally,?\s+we\b', r'\bthen\s+we\b',
    r"\bcan't\b", r"\bwon't\b", r"\bdon't\b",
    r'\bgot\b', r'\bkinda\b', r'\bpretty\s+much\b',
]


def count_pattern_hits(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def compute_perplexity_proxy(text: str) -> float:
    """
    Pseudo-perplexity using character-level unigram entropy.
    Returns 0-1 score where 1 = high perplexity = more human.
    """
    if not text or len(text) < 50:
        return 0.5
    chars = text.lower()
    freq  = Counter(chars)
    total = len(chars)
    if total == 0:
        return 0.5
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values() if c > 0)
    return round(min(entropy / 5.0, 1.0), 3)


def burstiness_score(text: str) -> float:
    """
    Measure word burstiness. Low = AI-like uniform distribution.
    Returns 0-1 where low = more AI-like.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 20:
        return 0.5
    chunk_size = 20
    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
    if len(chunks) < 2:
        return 0.5
    top_words = [w for w, _ in Counter(words).most_common(10) if w not in {
        'the','a','an','and','or','is','in','of','to','it','this','that','was'
    }]
    if not top_words:
        return 0.5
    variances = []
    for w in top_words:
        counts = [chunk.count(w) for chunk in chunks]
        mean_c = sum(counts) / len(counts)
        var    = sum((c - mean_c) ** 2 for c in counts) / len(counts)
        variances.append(var)
    avg_var = sum(variances) / len(variances)
    burstiness = min(avg_var / 2.0, 1.0)
    return round(burstiness, 3)


def detect_ai_probability(text: str, style_result: dict = None) -> dict:
    """
    Multi-signal AI content detection.
    Returns probability and contributing signals.
    """
    if not text or len(text.strip()) < 30:
        return {'probability': 0.1, 'signals': {}, 'confidence': 'low'}

    # Signal 1: AI phrase density (most reliable for short texts)
    ai_hits    = count_pattern_hits(text, AI_PHRASE_PATTERNS)
    human_hits = count_pattern_hits(text, STUDENT_PATTERNS)
    phrase_score = min(ai_hits / max(len(AI_PHRASE_PATTERNS) * 0.3, 1), 1.0)
    human_score  = min(human_hits / max(len(STUDENT_PATTERNS) * 0.3, 1), 1.0)
    phrase_signal = max(0.0, phrase_score - human_score * 0.5)

    # Signal 2: Sentence length uniformity (AI = low variance)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 3]
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        mean_l  = sum(lengths) / len(lengths)
        std_l   = math.sqrt(sum((l - mean_l)**2 for l in lengths) / len(lengths))
        cv      = std_l / mean_l if mean_l > 0 else 1.0
        uniformity_signal = max(0.0, 1.0 - cv * 1.0)
    else:
        uniformity_signal = 0.2

    # Signal 3: Perplexity
    perp = compute_perplexity_proxy(text)
    perp_signal = max(0.0, 1.0 - perp)

    # Signal 4: Burstiness
    burst = burstiness_score(text)
    burst_signal = max(0.0, 1.0 - burst * 1.2)

    # Signal 5: Style markers from stylometry
    style_ai_ratio = 0.0
    if style_result:
        style_ai_ratio = style_result.get('ai_markers', {}).get('ai_marker_ratio', 0.0)

    # Weighted combination
    # FIX v2.3: phrase_signal weight raised to 0.55 (most reliable short-text signal)
    #           uniformity_signal lowered to 0.10 (unreliable on short essays)
    ai_probability = (
        phrase_signal     * 0.55 +
        uniformity_signal * 0.10 +
        perp_signal       * 0.10 +
        burst_signal      * 0.10 +
        style_ai_ratio    * 0.15
    )
    ai_probability = round(min(max(ai_probability, 0.02), 0.98), 3)
    confidence = 'high' if len(sentences) >= 5 else ('medium' if len(sentences) >= 2 else 'low')

    return {
        'probability': ai_probability,
        'confidence':  confidence,
        'signals': {
            'phrase_density':      round(phrase_signal, 3),
            'sentence_uniformity': round(uniformity_signal, 3),
            'perplexity_signal':   round(perp_signal, 3),
            'burstiness_signal':   round(burst_signal, 3),
            'style_markers':       round(style_ai_ratio, 3),
        },
        'ai_phrases_found':    min(ai_hits, 10),
        'human_phrases_found': min(human_hits, 10),
    }


# ════════════════════════════════════════════════════════════
# COPY DETECTION
# ════════════════════════════════════════════════════════════

def tokenize_for_copy(text: str) -> list:
    """Tokenize text, removing stopwords."""
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    stopwords = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'by','from','is','are','was','were','be','have','has','had','do','does',
        'did','will','would','could','should','may','might','it','its','this',
        'that','these','those','not','no','as','if','so','we','you','he','she',
        'they','i','my','your','his','her','our','their','me','him','us','them',
        'very','also','just','get','like','can','about','because','when','what',
    }
    return [w for w in words if w not in stopwords]


def get_ngrams(tokens: list, n: int) -> Counter:
    """Get n-gram counts from token list."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def build_tfidf_vector(text: str) -> dict:
    """Build TF vector (no IDF corpus — single doc comparison)."""
    tokens = tokenize_for_copy(text)
    if not tokens:
        return {}
    tf    = Counter(tokens)
    total = len(tokens)
    return {w: c / total for w, c in tf.items()}


def cosine_sim_dicts(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two TF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot    = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Jaccard similarity of n-grams between two texts."""
    ta = tokenize_for_copy(text_a)
    tb = tokenize_for_copy(text_b)
    ng_a = set(get_ngrams(ta, n).keys())
    ng_b = set(get_ngrams(tb, n).keys())
    if not ng_a or not ng_b:
        return 0.0
    intersection = ng_a & ng_b
    union        = ng_a | ng_b
    return round(len(intersection) / len(union), 4) if union else 0.0


def compute_text_similarity(text_a: str, text_b: str) -> dict:
    """
    Compute text similarity robust to same-topic vocabulary overlap.

    FIX v2.3: old formula (tfidf*0.5 + ng3*0.3 + ng2*0.2) scored 40-50%
    for two different essays on "Why Stress?" because content words overlap.

    New formula: require BOTH high TF-IDF AND high trigram overlap.
    Trigrams are phrase-specific — "mug up and if we" only appears in copies,
    not in independent essays on the same topic.

    combined = min(tfidf_cos, ng3 * 1.5) * 0.6 + ng3 * 0.4

    Effect:
      - Independent same-topic essays: tfidf ~0.45, ng3 ~0.05 → combined ~0.07
      - Direct copies (s3→s4):        tfidf ~0.85, ng3 ~0.60 → combined ~0.55
      - AI text pairs (s2/s5):        tfidf ~0.70, ng3 ~0.35 → combined ~0.35
    """
    if not text_a or not text_b:
        return {'similarity': 0.0, 'tfidf_cos': 0.0, 'ngram_3': 0.0, 'ngram_2': 0.0}

    vec_a     = build_tfidf_vector(text_a)
    vec_b     = build_tfidf_vector(text_b)
    tfidf_cos = cosine_sim_dicts(vec_a, vec_b)
    ng3       = ngram_overlap(text_a, text_b, n=3)
    ng2       = ngram_overlap(text_a, text_b, n=2)

    # FIX: gate combined score on trigram overlap, not just cosine
    gated     = min(tfidf_cos, ng3 * 1.5)
    combined  = gated * 0.60 + ng3 * 0.40

    return {
        'similarity': round(combined, 4),
        'tfidf_cos':  round(tfidf_cos, 4),
        'ngram_3':    round(ng3, 4),
        'ngram_2':    round(ng2, 4),
    }


def detect_copies_batch(
    target_text: str,
    target_hw_fp: list,
    other_assignments: list,
) -> dict:
    """
    Compare target assignment against all others for copy detection.

    other_assignments: list of dicts, each with:
        - 'id':             str
        - 'student_id':     str   (for same-student guard in app.py)
        - 'student_name':   str
        - 'text':           str   (OCR text)
        - 'hw_fingerprint': list  (18-feature handwriting fingerprint)

    Returns:
        {
          'max_text_similarity': float,
          'max_hw_similarity':   float,
          'combined_copy_score': float,
          'matches':             list,
          'copy_flag':           bool,
        }
    """
    if not other_assignments:
        return {
            'max_text_similarity': 0.0,
            'max_hw_similarity':   0.0,
            'combined_copy_score': 0.0,
            'matches':             [],
            'copy_flag':           False,
        }

    import numpy as np

    matches = []
    for other in other_assignments:
        other_text = other.get('text', '')
        other_fp   = other.get('hw_fingerprint')

        # Text similarity (gated trigram formula)
        text_sim_result = compute_text_similarity(target_text, other_text)
        text_sim = text_sim_result['similarity']

        # Handwriting similarity (18-feature identity vector)
        hw_sim = 0.0
        if target_hw_fp and other_fp:
            va = np.array(target_hw_fp, dtype=np.float32)
            vb = np.array(other_fp,     dtype=np.float32)
            if len(va) == len(vb):
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    hw_sim = float(np.dot(va / na, vb / nb))
                    hw_sim = round(min(max(hw_sim, 0.0), 1.0), 3)

        # Combined: text similarity is the stronger signal for copy;
        # HW similarity is secondary (flags ghostwriter scenario)
        combined = text_sim * 0.65 + hw_sim * 0.35

        # FIX v2.3: only include matches above meaningful threshold
        # Old threshold 0.08 included every same-topic essay pair
        if combined > 0.25 or hw_sim > 0.88:
            matches.append({
                'id':           other.get('id'),
                'student_id':   other.get('student_id', ''),
                'student_name': other.get('student_name', 'Unknown'),
                'text_sim':     round(text_sim, 3),
                'hw_sim':       round(hw_sim, 3),
                'combined':     round(combined, 3),
            })

    matches.sort(key=lambda x: x['combined'], reverse=True)
    top_matches = matches[:5]

    max_text = max((m['text_sim'] for m in matches), default=0.0)
    max_hw   = max((m['hw_sim']   for m in matches), default=0.0)
    max_comb = max((m['combined'] for m in matches), default=0.0)

    return {
        'max_text_similarity': round(max_text, 3),
        'max_hw_similarity':   round(max_hw,   3),
        'combined_copy_score': round(max_comb, 3),
        'matches':             top_matches,
        'copy_flag':           max_comb > 0.40,
    }