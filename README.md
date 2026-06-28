# GradeCraft 📝
**Automated Answer Sheet Grading Powered by an Async Multi-Agent VLM Swarm**

GradeCraft has been completely redesigned to leverage a specialized **Multi-Agent Vision-Language Swarm Architecture** powered by the ultra-fast Groq inference engine. It seamlessly converts raw, noisy handwritten exam booklets into accurate, mathematically verified objective scores.

## 🚀 Architecture Highlights

1. **Vision Transcriber (`meta-llama/llama-4-scout-17b-16e-instruct`)**
   - Ingests high-resolution images of handwritten pages.
   - Accurately recovers layout schemas, flattening 2D matrices and diagrams into structured Markdown and LaTeX.
   - Built-in chunking logic respects strict vision API constraints.

2. **Math Verifier (`groq/compound-mini`)**
   - Ingests the unified transcribed document alongside the solution criteria.
   - Evaluates complex calculation paths and checks for mathematical equivalence.

3. **Rubric Evaluator (`llama-3.3-70b-versatile`)**
   - Parses the master solution key PDF into programmatic JSON rubrics.
   - Synthesizes the transcribed exam and verification logs to output an array of scored evaluations with detailed deduction rationales.

4. **Async Swarm Orchestrator (`src/core/throttler.py`)**
   - Custom `LeakyBucketThrottler` paces concurrent API requests based on predefined rate limit profiles (RPM/TPM).
   - High-performance, non-blocking execution pipeline seamlessly bridged into the Streamlit frontend.

5. **Dependency-Free PDF Processing**
   - Upgraded to `PyMuPDF` (`fitz`) to extract base64 images from PDFs instantly, completely eliminating tricky external dependencies like Poppler/pdf2image.

## 🛠️ Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory (or just use the sidebar in the app):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage
1. Open the Streamlit dashboard on `http://localhost:8501`.
2. Upload the **Solution Key PDF** (digital or scanned).
3. Upload the **Student Answer Booklet PDF**.
4. Click **Evaluate Student (Swarm Pipeline)**.
5. Review the auto-generated scores and deduction rationales!
