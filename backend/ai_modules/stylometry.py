"""
VRITTIFY ASTRA – Stylometry Engine
NLP-based writing style analysis: sentence structure, linguistic features,
vocabulary richness, and style consistency scoring.
"""

import re
import math
from collections import Counter


# ── Linguistic feature extraction ────────────────────────────

STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'it','its','this','that','these','those','i','we','you','he','she','they',
    'my','your','his','her','our','their','me','him','us','them','not','no',
}

# Common AI writing markers
AI_MARKERS = [
    'furthermore', 'moreover', 'in conclusion', 'it is important to note',
    'it is worth noting', 'as mentioned', 'in summary', 'to summarize',
    'plays a crucial role', 'it is essential', 'in addition', 'additionally',
    'consequently', 'thus', 'therefore', 'hence', 'nevertheless', 'nonetheless',
    'on the other hand', 'in contrast', 'as a result', 'due to this',
    'it should be noted', 'it can be seen', 'one can observe',
    'in the context of', 'with respect to', 'in terms of',
    'significantly', 'substantially', 'comprehensively', 'effectively',
    'various aspects', 'multifaceted', 'diverse range', 'key aspects',
    'delve into', 'shed light on', 'pivotal role', 'undeniable',
    'at the end of the day', 'it goes without saying',
]

# Human handwriting markers (informal, natural language patterns)
HUMAN_MARKERS = [
    "i think", "i believe", "i feel", "we know", "we can see",
    "as we learned", "like we studied", "the answer is", "because",
    "so", "then", "also", "but", "since", "when", "if", "let me",
    "first", "next", "finally", "another", "example",
]


def tokenize(text: str) -> list:
    """Basic word tokenizer."""
    text = text.lower()
    words = re.findall(r"\b[a-z']+\b", text)
    return words


def get_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def lexical_richness(words: list) -> float:
    """
    Type-Token Ratio (TTR): unique words / total words.
    Corrected TTR using moving window to reduce length bias.
    """
    if not words:
        return 0.0
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not content_words:
        return 0.0
    total  = len(content_words)
    unique = len(set(content_words))
    # MATTR: Moving Average TTR with window 50
    if total < 50:
        return round(unique / total, 3)
    window = 50
    ttrs = []
    for i in range(total - window + 1):
        chunk = content_words[i:i+window]
        ttrs.append(len(set(chunk)) / window)
    return round(sum(ttrs) / len(ttrs), 3)


def sentence_length_variance(sentences: list) -> dict:
    """
    Compute sentence length statistics.
    High variance = more human-like. Low variance = more AI-like.
    """
    if not sentences:
        return {'mean': 0.0, 'variance': 0.0, 'std': 0.0, 'cv': 0.0}
    lengths = [len(s.split()) for s in sentences]
    mean    = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std      = math.sqrt(variance)
    cv       = round(std / mean, 3) if mean > 0 else 0.0  # Coefficient of variation
    return {
        'mean':     round(mean, 2),
        'variance': round(variance, 2),
        'std':      round(std, 2),
        'cv':       cv,
    }


def detect_ai_markers(text: str) -> dict:
    """Count AI writing pattern markers and compute probability."""
    text_lower = text.lower()
    hits = []
    for marker in AI_MARKERS:
        if marker in text_lower:
            hits.append(marker)
    human_hits = []
    for marker in HUMAN_MARKERS:
        if marker in text_lower:
            human_hits.append(marker)
    ai_ratio    = len(hits) / len(AI_MARKERS)
    human_ratio = len(human_hits) / len(HUMAN_MARKERS)
    return {
        'ai_marker_count':    len(hits),
        'ai_marker_ratio':    round(ai_ratio, 3),
        'human_marker_count': len(human_hits),
        'human_marker_ratio': round(human_ratio, 3),
        'ai_markers_found':   hits[:10],
    }


def punctuation_profile(text: str) -> dict:
    """Analyze punctuation usage (human writers use varied punctuation)."""
    punct_counts = Counter(c for c in text if c in '.,;:!?-–—()"\'')
    total_chars  = len(text)
    punct_density = sum(punct_counts.values()) / total_chars if total_chars else 0
    return {
        'density':  round(punct_density, 4),
        'counts':   dict(punct_counts.most_common(8)),
    }


def paragraph_structure(text: str) -> dict:
    """Analyze paragraph lengths and structure."""
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paras:
        paras = [text]
    lengths = [len(p.split()) for p in paras]
    avg = sum(lengths) / len(lengths) if lengths else 0
    return {
        'count':      len(paras),
        'avg_length': round(avg, 1),
        'lengths':    lengths,
    }


def compute_style_score(
    lexical: float,
    sent_stats: dict,
    ai_markers: dict,
    punct: dict,
) -> float:
    """
    Compute overall writing style authenticity score (0-1).
    Higher = more authentic human writing.
    """
    # Lexical richness: 0.15-0.70 is typical for student writing
    lr_score = min(lexical / 0.5, 1.0)

    # Sentence variation: more variation = more human
    cv = sent_stats.get('cv', 0.3)
    sv_score = min(cv / 0.5, 1.0)

    # AI markers: fewer = better
    ai_penalty = ai_markers.get('ai_marker_ratio', 0)
    human_bonus = ai_markers.get('human_marker_ratio', 0)
    marker_score = max(0.0, 1.0 - ai_penalty * 2 + human_bonus * 0.5)
    marker_score = min(marker_score, 1.0)

    # Punctuation: moderate density is human-like
    pd = punct.get('density', 0.02)
    punct_score = 1.0 if 0.01 <= pd <= 0.06 else max(0.3, 1 - abs(pd - 0.035) * 20)

    style = (
        lr_score     * 0.30 +
        sv_score     * 0.30 +
        marker_score * 0.30 +
        punct_score  * 0.10
    )
    return round(min(max(style, 0.05), 1.0), 3)


def analyze_stylometry(text: str) -> dict:
    """
    Full stylometry analysis pipeline.
    Returns comprehensive style features dict.
    """
    if not text or len(text.strip()) < 20:
        return {
            'style_score':   0.5,
            'lexical_richness': 0.0,
            'sentence_stats': {},
            'ai_markers': {},
            'punctuation': {},
            'paragraph_stats': {},
            'word_count': 0,
        }

    words     = tokenize(text)
    sentences = get_sentences(text)
    lexical   = lexical_richness(words)
    sent_stats = sentence_length_variance(sentences)
    ai_markers = detect_ai_markers(text)
    punct      = punctuation_profile(text)
    para_stats = paragraph_structure(text)
    style_score = compute_style_score(lexical, sent_stats, ai_markers, punct)

    return {
        'style_score':      style_score,
        'lexical_richness': lexical,
        'sentence_stats':   sent_stats,
        'ai_markers':       ai_markers,
        'punctuation':      punct,
        'paragraph_stats':  para_stats,
        'word_count':       len(words),
        'sentence_count':   len(sentences),
    }


def compare_styles(text_a: str, text_b: str) -> float:
    """
    Compare stylometric fingerprints of two texts.
    Returns similarity score 0-1.
    """
    if not text_a or not text_b:
        return 0.0

    def fingerprint(text):
        words = tokenize(text)
        sents = get_sentences(text)
        return {
            'lr':  lexical_richness(words),
            'cv':  sentence_length_variance(sents).get('cv', 0),
            'ai':  detect_ai_markers(text).get('ai_marker_ratio', 0),
            'pd':  punctuation_profile(text).get('density', 0),
            'awl': sum(len(w) for w in words) / len(words) if words else 0,
        }

    fa = fingerprint(text_a)
    fb = fingerprint(text_b)

    diffs = []
    for key in fa:
        va, vb = fa[key], fb[key]
        maxv = max(abs(va), abs(vb), 0.001)
        diffs.append(abs(va - vb) / maxv)

    avg_diff = sum(diffs) / len(diffs) if diffs else 1.0
    similarity = round(1.0 - min(avg_diff, 1.0), 3)
    return similarity