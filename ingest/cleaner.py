import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Normalizing Unicode characters.
    2. Removing excessive whitespace.
    3. Removing non-printable characters.
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)

    # Replace multiple spaces/newlines with a single space (optional, depending on chunking strategy)
    # For now, we'll keep newlines but collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove non-printable characters (except newlines)
    text = "".join(ch for ch in text if ch.isprintable() or ch == '\n')
    
    # Strip leading/trailing whitespace
    return text.strip()
