import os
import logging
from typing import Optional
import pandas as pd
from pypdf import PdfReader
import docx
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_from_pdf(file_path: str) -> str:
    """Extracts text from PDF. Tries standard extraction first. If text is insufficient, falls back to OCR."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # Check if text is sufficient (heuristic: < 50 chars might be an image scan)
        if len(text.strip()) < 50:
            logger.info(f"PDF text content low ({len(text)} chars). Attempting OCR for {file_path}")
            images = convert_from_path(file_path)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img)
            text = ocr_text
            
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {e}")
        return ""
        
    return text

def extract_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX {file_path}: {e}")
        return ""

def extract_from_csv(file_path: str) -> str:
    try:
        df = pd.read_csv(file_path)
        # Convert each row to a string representation
        text_parts = []
        for _, row in df.iterrows():
            row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            text_parts.append(row_str)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting CSV {file_path}: {e}")
        return ""

def extract_from_txt(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error extracting TXT {file_path}: {e}")
        return ""

def extract_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using OCR.
    """
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"Error extracting Image {file_path}: {e}")
        return ""

def extract_content(file_path: str) -> str:
    """
    Main entry point for extraction. Determines file type and calls appropriate extractor.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return ""

    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_from_pdf(file_path)
    elif ext == '.docx':
        return extract_from_docx(file_path)
    elif ext == '.csv':
        return extract_from_csv(file_path)
    elif ext in ['.txt', '.md', '.log']:
        return extract_from_txt(file_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return extract_from_image(file_path)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""
