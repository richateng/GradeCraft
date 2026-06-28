# Handwritten Answer-Sheet Auto-Grader

Small prototype to OCR scanned handwritten answer sheets using Google Vision, then evaluate against a provided solution using a local Ollama LLM.

Highlights
- OCR: Google Cloud Vision (document_text_detection) applied to PDF pages via `pdf2image`.
- Evaluation: Local Ollama model (recommended: Llama-2-13b-chat or Mistral-7B-instruct) called via HTTP API.
- Frontend: Streamlit app with upload + evaluate buttons.

Setup
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Google Cloud Vision: set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON.
3. Install Poppler (required by `pdf2image`) and ensure `pdftoppm` is on PATH.
	- On Windows, Poppler binaries may be located under a path like `D:\Tools\poppler-26.02.0\Library\bin`.
	 - On Windows, Poppler binaries may be located under a path like `D:\Tools\poppler-26.02.0\Library\bin`.
	- You can either add that folder to your `PATH`, or set an environment variable `POPPLER_PATH` to that folder, or paste the path into the Streamlit app's "poppler path" field.
	- Example (PowerShell temporary):
	```powershell
	$env:Path += ";D:\Tools\poppler-26.02.0\Library\bin"
	```
	- Persistent (requires restart):
	```powershell
	setx PATH "$($env:PATH);D:\Tools\poppler-26.02.0\Library\bin"
	setx POPPLER_PATH "D:\Tools\poppler-26.02.0\Library\bin"
	```

	4. Google Cloud credentials
	- Create a service account with the Vision API enabled and download the JSON key.
	- For Streamlit you can either set the environment variable before running the app:
	```powershell
	$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
	``` 
	or upload the JSON in the app UI (temporary, session-only) or paste the path into the app's credential field.
4. Install and run Ollama locally and pull a model (e.g., `llama2-13b-chat` or `mistral-7b-instruct`). Ollama listens on `http://127.0.0.1:11434` by default.

Run
```
streamlit run app.py
```

Notes
- The solution file can be a plain text file, a digital PDF (text-extractable), or a scanned PDF (will be OCRed).
- The parser looks for question headings like `Q1`/`Q 1`/`Question 1` and an optional `Marks:` line per question. If not found, the whole solution is treated as one item.
- The grader asks the LLM to return structured JSON per question; if parsing fails, raw LLM output is shown.
