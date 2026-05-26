"""
VRITTIFY ASTRA – OCR Module
Extracts text from preprocessed handwritten assignment images using Tesseract OCR.
"""

import re
import pytesseract
from PIL import Image
import numpy as np

import platform, shutil
if platform.system() == 'Windows':
    _tess = shutil.which('tesseract') or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = _tess
# On Linux/Mac, pytesseract finds it from PATH automatically — no line needed


# Tesseract config optimized for handwriting
TESS_CONFIG = r'--oem 3 --psm 6 -l eng'


def extract_text_from_image(pil_img: Image.Image) -> str:
    """Run Tesseract OCR on a single PIL image."""
    try:
        text = pytesseract.image_to_string(pil_img, config=TESS_CONFIG)
        return text.strip()
    except Exception as e:
        return ""


def extract_text_with_confidence(pil_img: Image.Image) -> dict:
    """
    Run Tesseract and return text + per-word confidence.
    Returns dict: { 'text': str, 'avg_confidence': float, 'words': list }
    """
    try:
        data = pytesseract.image_to_data(
            pil_img, config=TESS_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        words = []
        confs = []
        for i, word in enumerate(data['text']):
            conf = int(data['conf'][i])
            if conf > 0 and word.strip():
                words.append({'word': word.strip(), 'conf': conf})
                confs.append(conf)
        avg_conf = round(sum(confs) / len(confs) / 100, 3) if confs else 0.0
        full_text = ' '.join(w['word'] for w in words)
        return {
            'text':           full_text,
            'avg_confidence': avg_conf,
            'word_count':     len(words),
            'words':          words,
        }
    except Exception:
        return {'text': '', 'avg_confidence': 0.0, 'word_count': 0, 'words': []}


def extract_from_all_pages(page_pil_images: list) -> dict:
    """
    Run OCR on all pages and combine results.
    page_pil_images: list of PIL Images (preprocessed)
    Returns combined OCR result dict.
    """
    all_text = []
    all_words = []
    all_conf  = []

    for pil_img in page_pil_images:
        result = extract_text_with_confidence(pil_img)
        if result['text']:
            all_text.append(result['text'])
            all_words.extend(result['words'])
            all_conf.append(result['avg_confidence'])

    combined_text = '\n'.join(all_text)
    avg_conf = round(sum(all_conf) / len(all_conf), 3) if all_conf else 0.0

    return {
        'text':           combined_text,
        'avg_confidence': avg_conf,
        'word_count':     len(all_words),
        'page_count':     len(page_pil_images),
        'words':          all_words,
    }


def clean_ocr_text(text: str) -> str:
    """Clean OCR output: remove junk characters, normalize whitespace."""
    if not text:
        return ""
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace per line
    lines = [l.strip() for l in text.split('\n')]
    lines = [l for l in lines if l]
    return '\n'.join(lines)


def get_text_stats(text: str) -> dict:
    """Compute basic text statistics for downstream analysis."""
    if not text:
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0.0, 'avg_sentence_length': 0.0,
            'unique_word_ratio': 0.0, 'paragraph_count': 0,
        }
    words     = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    paras     = [p.strip() for p in text.split('\n\n') if p.strip()]
    word_count    = len(words)
    sent_count    = len(sentences)
    unique_words  = set(w.lower() for w in words)
    avg_wl = round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    avg_sl = round(word_count / sent_count, 2) if sent_count else 0.0
    unique_ratio = round(len(unique_words) / word_count, 3) if word_count else 0.0

    return {
        'char_count':         len(text),
        'word_count':         word_count,
        'sentence_count':     sent_count,
        'avg_word_length':    avg_wl,
        'avg_sentence_length': avg_sl,
        'unique_word_ratio':  unique_ratio,
        'paragraph_count':    len(paras),
    }