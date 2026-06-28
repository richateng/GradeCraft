import re
import json
from pathlib import Path
from PyPDF2 import PdfReader
from ocr import ocr_pdf_gcv


QUESTION_SPLIT_RE = re.compile(r"(?m)^(?:Q(?:uestion)?\s*|Q)\s*(\d+)[:\.\)]?", re.IGNORECASE)
MARKS_RE = re.compile(r"Marks?[:\s]*([0-9]+)", re.IGNORECASE)


def read_text_file(path: Path):
    return path.read_text(encoding="utf-8")


def extract_text_from_pdf(path: Path, poppler_path=None):
    # try digital PDF extraction first
    try:
        reader = PdfReader(str(path))
        texts = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                texts.append(txt)
        joined = "\n\n".join(texts).strip()
        if joined:
            return joined
    except Exception:
        pass
    # fallback to OCR for scanned PDF
    res = ocr_pdf_gcv(str(path), poppler_path=poppler_path)
    return "\n\n".join(res.get("pages", [])).strip()


def parse_solution_file(path, poppler_path=None):
    p = Path(path)
    if p.suffix.lower() == ".txt":
        text = read_text_file(p)
    elif p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    else:
        text = extract_text_from_pdf(p, poppler_path=poppler_path)

    # split into questions
    parts = QUESTION_SPLIT_RE.split(text)
    if len(parts) <= 1:
        # no question headings found, return whole as single entry
        marks = None
        m = MARKS_RE.search(text)
        if m:
            marks = int(m.group(1))
        return [{"qnum": 1, "text": text, "max_marks": marks}]

    # parts layout: [prefix, qnum1, body1, qnum2, body2, ...]
    result = []
    prefix = parts[0].strip()
    i = 1
    while i < len(parts):
        qnum = parts[i]
        body = parts[i + 1].strip()
        m = MARKS_RE.search(body)
        marks = int(m.group(1)) if m else None
        result.append({"qnum": int(qnum), "text": body, "max_marks": marks})
        i += 2
    # if there was a prefix before Q1, attach as Q0
    if prefix:
        result.insert(0, {"qnum": 0, "text": prefix, "max_marks": None})
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data = parse_solution_file(sys.argv[1])
        import pprint
        pprint.pprint(data)