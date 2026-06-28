import os
import tempfile
import json
import streamlit as st
from pathlib import Path
import os
import shutil

# ensure src is importable
import sys
ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ocr import ocr_pdf_gcv
from solution_parser import parse_solution_file
from evaluator import evaluate, get_ollama_models, detect_ollama_endpoint


st.title("Handwritten Answer Sheet Auto-Grader")

st.sidebar.markdown("Upload student scan and provided solution, then evaluate with OpenAI or local Ollama.")

student_file = st.file_uploader("Upload handwritten answer sheet (PDF)", type=["pdf"]) 
solution_file = st.file_uploader("Upload solution (PDF / TXT / JSON)", type=["pdf", "txt", "json"]) 

provider = st.sidebar.selectbox("Language model provider", ["openai", "ollama"], index=0)
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if provider == "openai":
    st.sidebar.markdown("Use OpenAI for grading. Set `OPENAI_API_KEY` in the environment or paste it here.")
    openai_api_key = st.sidebar.text_input("OpenAI API key", value=openai_api_key, type="password", key="openai_api_key")
    model_name = st.sidebar.text_input("OpenAI model", value="gpt-3.5-turbo", key="openai_model_name")
    st.sidebar.info("Recommended: gpt-3.5-turbo or gpt-3.5-turbo-16k for broad availability. If you see model errors, try a different model here.")
    per_question = st.sidebar.checkbox("Grade per question (safer, more API calls)", value=False, key="per_question")
else:
    st.sidebar.markdown("Use a local Ollama model. If none appear, ensure Ollama is running and reachable.")
    try:
        _models = get_ollama_models()
    except Exception:
        _models = []

    if _models:
        st.sidebar.success("Local Ollama models found")
        model_name = st.sidebar.selectbox("Choose Ollama model (exact id)", _models, index=0, key="model_select")
        if st.sidebar.button("Refresh local models"):
            st.experimental_rerun()
    else:
        st.sidebar.warning("No local Ollama models found; enter exact model id below")
        model_name = st.sidebar.text_input("Ollama model name (exact id)", value="qwen3:8b", key="model_text")

    if st.sidebar.button("Auto-detect Ollama server"):
        try:
            det = detect_ollama_endpoint()
            st.success(f"Detected endpoint {det['endpoint']} with model {det['model']}")
            if det.get("models"):
                _models = det.get("models")
                model_name = st.selectbox("Choose model", _models, index=0)
        except Exception as e:
            st.error(f"Auto-detect failed: {e}")

poppler_path = None  # Poppler path input removed — auto-detect from POPPLER_PATH env or PATH
creds_path = st.text_input("(Optional) Google credentials JSON path (or leave empty to use env var)", value="")
creds_file = st.file_uploader("(Optional) Upload Google service-account JSON (will be used for this session)", type=["json"])

if st.button("Evaluate"):
    if not student_file or not solution_file:
        st.error("Please upload both student sheet and solution file.")
    else:
        with tempfile.TemporaryDirectory() as td:
            student_path = Path(td) / "student.pdf"
            solution_path = Path(td) / (solution_file.name)
            student_path.write_bytes(student_file.getvalue())
            solution_path.write_bytes(solution_file.getvalue())

            st.info("Running OCR on student sheet (Google Vision)...")
            # handle Google credentials: uploaded file or explicit path
            if creds_file:
                # write uploaded credentials to a temp file and point GOOGLE_APPLICATION_CREDENTIALS to it
                creds_tmp = Path(td) / "gcloud_credentials.json"
                creds_tmp.write_bytes(creds_file.getvalue())
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_tmp)
            elif creds_path and creds_path.strip():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path.strip()

            # Auto-detect Poppler: prefer POPPLER_PATH env var, else rely on PATH
            poppler_used = os.environ.get("POPPLER_PATH")
            # validate pdftoppm availability
            pdftoppm_ok = False
            if poppler_used:
                candidate = os.path.join(poppler_used, "pdftoppm.exe") if os.name == "nt" else os.path.join(poppler_used, "pdftoppm")
                pdftoppm_ok = os.path.exists(candidate)
            else:
                # check if pdftoppm is on PATH
                pdftoppm_ok = shutil.which("pdftoppm") is not None

            if not pdftoppm_ok:
                st.error(
                    "Poppler 'pdftoppm' not found. Either add Poppler to PATH or set POPPLER_PATH or provide the Poppler path above."
                )
                st.stop()

            try:
                ocr_res = ocr_pdf_gcv(str(student_path), poppler_path=poppler_used or None)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()
            student_text = "\n\n".join(ocr_res.get("pages", []))

            st.info("OCR output from the student answer sheet:")
            st.text_area("Extracted OCR text", value=student_text, height=280)
            st.info("Parsing solution file...")
            solution_questions = parse_solution_file(str(solution_path), poppler_path=poppler_used or None)

            st.info("Calling the selected model provider to evaluate answers...")

            if provider == "openai":
                if not openai_api_key or not openai_api_key.strip():
                    st.error("OpenAI API key is required for OpenAI evaluation. Set OPENAI_API_KEY or enter it above.")
                    st.stop()
                if not model_name or not model_name.strip():
                    model_name = "gpt-4o-mini-0613"
                chosen_provider = "openai"
            else:
                if isinstance(model_name, str):
                    model_name = model_name.strip()
                if _models and model_name not in _models:
                    st.error(f"Selected model '{model_name}' is not among local models: {', '.join(_models)}")
                    st.stop()
                chosen_provider = "ollama"

            try:
                with st.spinner("Evaluating..."):
                    results = evaluate(
                        student_text,
                        solution_questions,
                        provider=chosen_provider,
                        model=model_name,
                        api_key=openai_api_key if provider == "openai" else None,
                        per_question=per_question if provider == "openai" else False,
                    )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.stop()

            st.success("Evaluation complete")
            st.header("Question-wise results")
            total_score = 0
            total_max = 0
            for r in results:
                q = r.get("qnum")
                res = r.get("result")
                st.subheader(f"Question {q}")
                if "score" in res:
                    st.write(f"Score: **{res['score']}** / {res.get('max_score')}")
                    total_score += float(res['score'])
                    total_max += float(res.get('max_score') or 0)
                else:
                    st.write("Score: (not provided by model)")
                st.write("**Feedback:**")
                st.write(res.get("feedback") or res.get("raw") or json.dumps(res, indent=2))
                st.write("**Strengths:**")
                st.write(res.get("strengths") or "-")
                st.write("**Weaknesses:**")
                st.write(res.get("weaknesses") or "-")

            st.markdown("---")
            st.write(f"**Total:** {total_score} / {total_max}")

            st.download_button("Download full results (JSON)", data=json.dumps(results, indent=2), file_name="results.json")
