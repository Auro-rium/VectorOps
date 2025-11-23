from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of a specified size with overlap.
    Uses a simple character-based sliding window.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    # If text is shorter than chunk size, return it as one chunk
    if text_len <= chunk_size:
        return [text]
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to find a natural break point (newline or space) if we are not at the end
        if end < text_len:
            # Look back for the last newline within the last 10% of the chunk
            last_newline = text.rfind('\n', start, end)
            if last_newline != -1 and last_newline > start + (chunk_size * 0.9):
                end = last_newline + 1 # Include the newline
            else:
                # Look for the last space
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + (chunk_size * 0.9):
                    end = last_space + 1
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        # If we reached the end, break
        if end == text_len:
            break
            
        start += chunk_size - overlap
        
        # Ensure we don't get stuck if overlap >= chunk_size (shouldn't happen with valid inputs)
        if start >= end:
            start = end
            
    return chunks
