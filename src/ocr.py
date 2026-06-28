import io
import os
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from google.cloud import vision
from PIL import Image


def ocr_pdf_gcv(pdf_path, poppler_path=None):
    """Convert PDF pages to images and run Google Vision document OCR.

    Returns: dict {"pages": [text_per_page], "raw_responses": [response_dicts]}
    """
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    except PDFInfoNotInstalledError as e:
        hint = (
            f"Poppler not found. Install Poppler and ensure 'pdftoppm' is on PATH, "
            f"or pass poppler_path='{poppler_path or os.environ.get('POPPLER_PATH')}'. "
            f"Detected poppler_path: {poppler_path!r}"
        )
        raise RuntimeError(hint) from e
    client = vision.ImageAnnotatorClient()
    pages_text = []
    raw = []
    for img in images:
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            content = output.getvalue()
        image = vision.Image(content=content)
        resp = client.document_text_detection(image=image)
        text = ""
        if resp.full_text_annotation:
            text = resp.full_text_annotation.text
        pages_text.append(text)
        # convert to serializable form (minimal)
        raw.append({"text": text})
    return {"pages": pages_text, "raw_responses": raw}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        res = ocr_pdf_gcv(sys.argv[1])
        print(res["pages"][0] if res["pages"] else "(no text)")