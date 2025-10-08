from pathlib import Path
import re

def read_texts(raw_dir="data/raw"):
    texts = []
    for p in Path(raw_dir).glob("*"):
        if p.suffix.lower() in {".md", ".txt", ".pdf"}:
            try:
                if p.suffix.lower() == ".pdf":
                    import pypdf
                    pdf = pypdf.PdfReader(str(p))
                    content = "\n".join([(page.extract_text() or "") for page in pdf.pages])
                else:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                content = re.sub(r"\s+", " ", content).strip()
                texts.append({"id": p.name, "content": content, "path": str(p)})
            except Exception as e:
                print("Skipping", p, "->", e)
    return texts

def chunk_text(text, size=800, overlap=150):
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(text):
        chunks.append(text[i:i+size])
        i += step
    return chunks
