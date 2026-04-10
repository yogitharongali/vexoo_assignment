import re
from collections import defaultdict
from difflib import SequenceMatcher

# -----------------------------
# Sliding Window Function
# -----------------------------
def sliding_window(text, window_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + window_size]
        chunks.append(chunk)
        i += (window_size - overlap)
    return chunks

# -----------------------------
# Placeholder Summary
# -----------------------------
def summarize(text):
    return text[:100]  # simple mock summary

# -----------------------------
# Category Label (Rule-based)
# -----------------------------
def categorize(text):
    if "math" in text.lower():
        return "Math"
    elif "law" in text.lower():
        return "Legal"
    else:
        return "General"

# -----------------------------
# Distilled Knowledge
# -----------------------------
def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    return list(set(words[:10]))

# -----------------------------
# Build Knowledge Pyramid
# -----------------------------
def build_pyramid(chunks):
    pyramid = []

    for chunk in chunks:
        entry = {
            "raw": chunk,
            "summary": summarize(chunk),
            "category": categorize(chunk),
            "keywords": extract_keywords(chunk)
        }
        pyramid.append(entry)

    return pyramid

# -----------------------------
# Similarity Function
# -----------------------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# -----------------------------
# Query Retrieval
# -----------------------------
def retrieve(query, pyramid):
    best_score = 0
    best_result = None

    for entry in pyramid:
        for level in entry.values():
            text = " ".join(level) if isinstance(level, list) else level
            score = similarity(query, text)

            if score > best_score:
                best_score = score
                best_result = text

    return best_result

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    text = open("sample.txt").read()

    chunks = sliding_window(text)
    pyramid = build_pyramid(chunks)

    query = input("Enter your query: ")
    result = retrieve(query, pyramid)

    print("\nBest Match:\n", result)
