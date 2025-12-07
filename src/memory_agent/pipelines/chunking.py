def split_long_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        part = " ".join(words[start:end])
        chunks.append(part)
        start = end - overlap

    return chunks
