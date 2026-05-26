"""
VRITTIFY ASTRA – Preprocessor Module
Handles PDF to image conversion, noise removal, skew correction, binarization
"""

import os
import cv2
import numpy as np
from PIL import Image

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF pages to PIL images."""
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image not installed. Run: pip install pdf2image")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=dpi, fmt='PNG')
    return pages


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def remove_noise(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur + morphological opening to remove noise."""
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    kernel  = np.ones((2, 2), np.uint8)
    opened  = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    return opened


def binarize(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply Otsu binarization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Adaptive threshold for uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=4
    )
    return binary


def correct_skew(img: np.ndarray) -> np.ndarray:
    """Detect and correct page skew using Hough lines."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return img
        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return img
        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return img  # Skip tiny corrections
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    except Exception:
        return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Denoise + CLAHE contrast enhancement optimized for handwriting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Fast non-local means denoising — critical for handwritten pages
    denoised = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)
    # Higher CLAHE clip for handwriting contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def preprocess_image(pil_img: Image.Image) -> tuple:
    """
    Full preprocessing pipeline for a single page.
    Returns (preprocessed_cv2_img, preprocessed_pil_img, binary_img)
    """
    cv2_img   = pil_to_cv2(pil_img)
    denoised  = remove_noise(cv2_img)
    deskewed  = correct_skew(denoised)
    enhanced  = enhance_contrast(deskewed)
    binary    = binarize(enhanced)
    pil_out   = cv2_to_pil(enhanced)
    return enhanced, pil_out, binary


def preprocess_pdf(pdf_path: str, output_dir: str = None) -> list:
    """
    Full pipeline: PDF → pages → preprocessed images.
    Returns list of (cv2_img, pil_img, binary_img) per page.
    """
    pages = pdf_to_images(pdf_path)
    results = []
    for i, page in enumerate(pages):
        cv2_img, pil_img, binary = preprocess_image(page)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pil_img.save(os.path.join(output_dir, f"page_{i+1}.png"))
        results.append((cv2_img, pil_img, binary))
    return results


def get_page_metrics(cv2_img: np.ndarray, binary: np.ndarray) -> dict:
    """Extract basic page metrics used downstream."""
    h, w = cv2_img.shape[:2]
    ink_pixels  = int(np.sum(binary == 0))   # black pixels
    total_pixels = h * w
    ink_ratio   = ink_pixels / total_pixels if total_pixels else 0

    # Contour count (proxy for character/stroke count)
    contours, _ = cv2.findContours(
        cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    char_count = len(contours)

    # Horizontal projection profile (line detection)
    inv = cv2.bitwise_not(binary)
    h_proj = np.sum(inv, axis=1)
    text_lines = int(np.sum(h_proj > w * 0.02))  # rows with >2% ink

    return {
        'width':       w,
        'height':      h,
        'ink_ratio':   round(ink_ratio, 4),
        'char_count':  char_count,
        'text_lines':  text_lines,
    }