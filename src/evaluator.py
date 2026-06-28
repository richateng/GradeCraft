import json
import os
import random
import time
import requests
import subprocess
from typing import List, Optional
from pathlib import Path
import datetime


OLLAMA_BASE = "http://127.0.0.1:11434"
OLLAMA_REQUEST_TIMEOUT = 120
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo-0613"
OPENAI_FALLBACK_MODELS = [
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-4o-mini-0613",
    "gpt-4o-mini",
    "gpt-4o",
]
OPENAI_DEFAULT_TEMPERATURE = 0.0
OPENAI_DEFAULT_MAX_TOKENS = 1024

# Logs file for Ollama diagnostics
_OLLAMA_LOG = Path(__file__).resolve().parent / "logs" / "ollama_errors.log"


def _log_diagnostic(msg: str) -> None:
    try:
        _OLLAMA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _OLLAMA_LOG.open("a", encoding="utf-8") as fh:
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            fh.write(f"{ts} {msg}\n\n")
    except Exception:
        # Best-effort logging; don't let logging failures crash the app
        pass


def _post_with_backoff(url: str, headers: dict, json_payload: dict, timeout: int, max_retries: int = 8):
    retry_statuses = {429, 502, 503, 504}
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            wait_seconds = min(30, (2 ** attempt) + random.random())
            _log_diagnostic(
                f"OpenAI request exception {exc}; retrying in {wait_seconds:.1f}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_seconds)
            attempt += 1
            continue

        if resp.status_code not in retry_statuses:
            resp.raise_for_status()
            return resp

        if attempt >= max_retries:
            if resp.status_code == 429:
                raise RuntimeError(
                    "OpenAI rate limit reached after multiple retries. Please wait a minute, reduce the number of questions, or try a lower-rate model."
                )
            resp.raise_for_status()

        retry_after = resp.headers.get("Retry-After")
        wait_seconds = None
        if retry_after:
            try:
                wait_seconds = float(retry_after)
            except ValueError:
                pass
        if wait_seconds is None:
            wait_seconds = min(30, (2 ** attempt) + random.random())

        _log_diagnostic(
            f"OpenAI request throttled with status {resp.status_code}, retrying in {wait_seconds:.1f}s (attempt {attempt + 1}/{max_retries})"
        )
        time.sleep(wait_seconds)
        attempt += 1


def get_ollama_models(timeout: int = 5) -> List[str]:
    """Return a list of models available in the local Ollama server.

    Tries both `/v1/models` and `/api/models` endpoints.
    """
    endpoints = ["/v1/models", "/api/models"]
    for endpoint in endpoints:
        try:
            resp = requests.get(f"{OLLAMA_BASE}{endpoint}", timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            models = []
            if isinstance(data, dict):
                if endpoint == "/v1/models" and "data" in data:
                    for m in data["data"]:
                        if isinstance(m, dict) and "id" in m:
                            models.append(m["id"])
                elif "models" in data:
                    for m in data["models"]:
                        if isinstance(m, dict) and "name" in m:
                            models.append(m["name"])
            elif isinstance(data, list):
                for m in data:
                    if isinstance(m, dict) and "name" in m:
                        models.append(m["name"])
                    elif isinstance(m, str):
                        models.append(m)
            if models:
                return models
        except Exception:
            continue
    return []


def detect_ollama_endpoint(timeout: int = 5) -> dict:
    """Auto-detect a working Ollama endpoint and model.

    Returns a dict with keys: 'endpoint' (string), 'model' (string or None), 'models' (list).
    Raises RuntimeError if detection fails.
    """
    models = get_ollama_models(timeout=timeout)
    endpoints = ["/v1/completions", "/v1/chat/completions", "/api/generate", "/api/completions", "/api/chat/completions"]
    test_prompt = "Say hello"
    last_exc = None
    for ep in endpoints:
        candidates = models or [None]
        for m in candidates:
            try:
                if ep.endswith("/chat/completions"):
                    payload = {"messages": [{"role": "user", "content": test_prompt}], "max_tokens": 5}
                else:
                    payload = {"prompt": test_prompt, "max_tokens": 5}
                if m:
                    payload["model"] = m
                resp = _try_post(ep, payload, timeout=timeout)
                resp.raise_for_status()
                return {"endpoint": ep, "model": m, "models": models}
            except Exception as e:
                last_exc = e
                continue

    raise RuntimeError(f"Could not detect a working Ollama endpoint. Last error: {last_exc}")


def _try_post(endpoint: str, payload: dict, timeout: int = OLLAMA_REQUEST_TIMEOUT) -> requests.Response:
    url = f"{OLLAMA_BASE}{endpoint}"
    return requests.post(url, json=payload, timeout=timeout)


def _extract_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.dumps(json.loads(candidate))
            except json.JSONDecodeError:
                pass
    return text


def call_openai(prompt: str, model: str = OPENAI_DEFAULT_MODEL, api_key: Optional[str] = None, max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS, temperature: float = OPENAI_DEFAULT_TEMPERATURE, timeout: int = OLLAMA_REQUEST_TIMEOUT) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is required for OpenAI evaluation.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_message = {
        "role": "system",
        "content": (
            "You are an expert grader for university-level answers. "
            "Compare the student's answers with the provided solution and return only valid JSON. "
            "The JSON object must contain: score (number), max_score (number), feedback (string), strengths (array of strings), weaknesses (array of strings)."
        ),
    }
    user_message = {"role": "user", "content": prompt}
    payload = {
        "model": model,
        "messages": [system_message, user_message],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = _post_with_backoff(OPENAI_API_URL, headers=headers, json_payload=payload, timeout=timeout)
        data = resp.json()
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else None
        body = response.text if response is not None else str(exc)
        # If endpoint missing, try Responses API
        if status == 404:
            _log_diagnostic(f"chat/completions 404, falling back to responses: {body}")
            return _call_openai_responses(prompt, model=model, headers=headers, max_tokens=max_tokens, temperature=temperature, timeout=timeout)

        # If model not found, attempt fallback models
        if status == 400:
            try:
                err = response.json()
                err_code = err.get("error", {}).get("code")
                err_msg = err.get("error", {}).get("message")
            except Exception:
                err_code = None
                err_msg = body

            if err_code == "model_not_found" or (isinstance(err_msg, str) and "does not exist" in err_msg):
                _log_diagnostic(f"Model not found ({model}), trying fallbacks. Server message: {err_msg}")
                last_exc = None
                for fb in OPENAI_FALLBACK_MODELS:
                    if fb == model:
                        continue
                    try:
                        payload["model"] = fb
                        resp = _post_with_backoff(OPENAI_API_URL, headers=headers, json_payload=payload, timeout=timeout)
                        data = resp.json()
                        _log_diagnostic(f"Fallback to model {fb} succeeded")
                        break
                    except Exception as e:
                        last_exc = e
                        _log_diagnostic(f"Fallback model {fb} failed: {e}")
                        continue

                else:
                    # try Responses endpoint with first fallback
                    for fb in OPENAI_FALLBACK_MODELS:
                        try:
                            return _call_openai_responses(prompt, model=fb, headers=headers, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                        except Exception as e:
                            last_exc = e
                            continue
                    raise RuntimeError(f"All OpenAI model fallbacks failed. Last error: {last_exc}") from exc
            # not model not found -> surface error
            raise RuntimeError(f"OpenAI chat completions returned 400: {body}") from exc

        # other statuses: re-raise
        raise
    if not isinstance(data, dict):
        return json.dumps(data)

    choices = data.get("choices") or []
    if not choices:
        return json.dumps(data)

    first = choices[0]
    message = first.get("message", {})
    if isinstance(message, dict):
        function_call = message.get("function_call")
        if function_call and isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                return _extract_json(arguments)
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return _extract_json(content.strip())

    return json.dumps(data)


def _extract_response_output(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        if "output" in data:
            output = data["output"]
            if isinstance(output, str):
                return output
            if isinstance(output, list):
                texts = []
                for item in output:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        if "text" in item and isinstance(item["text"], str):
                            texts.append(item["text"])
                        elif "content" in item and isinstance(item["content"], list):
                            for chunk in item["content"]:
                                if isinstance(chunk, str):
                                    texts.append(chunk)
                                elif isinstance(chunk, dict) and "text" in chunk and isinstance(chunk["text"], str):
                                    texts.append(chunk["text"])
                return "\n".join(texts)
    return json.dumps(data)


def _call_openai_responses(prompt: str, model: str, headers: dict, max_tokens: int, temperature: float, timeout: int):
    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    # Try the requested model first, if it fails with model_not_found try fallbacks
    try_models = [model]
    # also try a stripped version (drop trailing version suffix like '-0613')
    if "-" in model:
        base = "-".join(model.split("-")[:-1])
        if base and base not in try_models:
            try_models.append(base)

    # append configured fallbacks
    for fb in OPENAI_FALLBACK_MODELS:
        if fb not in try_models:
            try_models.append(fb)

    last_exc = None
    for m in try_models:
        payload["model"] = m
        try:
            resp = _post_with_backoff(OPENAI_RESPONSES_URL, headers=headers, json_payload=payload, timeout=timeout)
            data = resp.json()
            return _extract_response_output(data)
        except requests.HTTPError as exc:
            response = exc.response
            body = response.text if response is not None else str(exc)
            status = response.status_code if response is not None else "unknown"
            _log_diagnostic(f"OpenAI /responses model {m} failed {status}: {body}")
            # If model not found, try next fallback; otherwise stop and raise
            try:
                err_json = response.json() if response is not None else {}
                code = err_json.get("error", {}).get("code")
            except Exception:
                code = None

            if code == "model_not_found" or (isinstance(body, str) and "does not exist" in body):
                last_exc = exc
                continue
            # other error -> surface
            raise RuntimeError(f"OpenAI responses endpoint error {status}: {body}") from exc

    # All fallbacks failed
    raise RuntimeError(f"All OpenAI response model fallbacks failed. Last error: {last_exc}") from last_exc


def call_llm(prompt: str, provider: str = "openai", model: str = OPENAI_DEFAULT_MODEL, api_key: Optional[str] = None, max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS, timeout: int = OLLAMA_REQUEST_TIMEOUT) -> str:
    provider = provider.lower().strip()
    if provider == "openai":
        return call_openai(prompt, model=model, api_key=api_key, max_tokens=max_tokens, timeout=timeout)
    if provider == "ollama":
        return call_ollama(prompt, model=model, max_tokens=max_tokens, timeout=timeout)
    raise ValueError(f"Unsupported provider: {provider}")


def call_ollama_cli(prompt: str, model: str, timeout: int = 180) -> str:
    # Ensure we have a model name; if not, pick the first available local model
    if not model:
        models = get_ollama_models(timeout=5)
        if models:
            model = models[0]
        else:
            raise RuntimeError("No Ollama model specified and no local models found")

    cmd = ["ollama", "run", model, prompt, "--format", "json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=True,
        )
        stdout = proc.stdout.strip()
        if not stdout:
            stderr = proc.stderr.strip() if proc.stderr else ""
            if stderr:
                raise RuntimeError(f"Ollama CLI returned no stdout. stderr: {stderr}")
            return ""

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout

        # Common output shapes: {status,message,data}, or nested data fields
        if isinstance(data, dict):
            if "response" in data:
                return data["response"]
            if "message" in data and isinstance(data["message"], str):
                return data["message"]
            if "content" in data:
                return data["content"]
            if "data" in data:
                d = data["data"]
                if isinstance(d, str):
                    return d
                if isinstance(d, dict):
                    for k in ("response", "message", "text"):
                        if k in d:
                            return d[k]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                if isinstance(first, dict) and "message" in first:
                    return first["message"].get("content") or stdout
        return stdout
    except subprocess.CalledProcessError as e:
        # Include both stdout and stderr in the raised error for diagnostics
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        _log_diagnostic(f"Ollama CLI CalledProcessError cmd={cmd} exit={e.returncode} stdout={out!r} stderr={err!r}")
        raise RuntimeError(f"Ollama CLI failed (cmd={cmd}) exit={e.returncode} stdout={out!r} stderr={err!r}") from e
    except Exception as e:
        _log_diagnostic(f"Ollama CLI error (cmd={cmd}): {e}")
        raise RuntimeError(f"Ollama CLI error (cmd={cmd}): {e}") from e


def call_ollama(prompt: str, model: str = "llama2-13b-chat", max_tokens: int = 512, timeout: int = OLLAMA_REQUEST_TIMEOUT) -> str:
    """Call the local Ollama server using a best-effort set of endpoints.

    Tries both v1 and legacy api endpoints until one succeeds.
    """
    endpoints = ["/v1/completions", "/v1/chat/completions", "/api/generate", "/api/completions", "/api/chat/completions"]
    payloads = []
    for ep in endpoints:
        if ep.endswith("/chat/completions"):
            payloads.append({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens})
        else:
            payloads.append({"model": model, "prompt": prompt, "max_tokens": max_tokens})

    last_exc: Optional[Exception] = None
    for ep, payload in zip(endpoints, payloads):
        try:
            resp = _try_post(ep, payload, timeout=timeout)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                return resp.text

            if isinstance(data, dict):
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    first = data["choices"][0]
                    if isinstance(first, dict) and "message" in first:
                        return first["message"].get("content") or json.dumps(data)
                    if isinstance(first, dict) and "text" in first:
                        return first.get("text")
                if "text" in data and isinstance(data["text"], str):
                    return data["text"]
            return json.dumps(data)
        except requests.HTTPError as e:
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            continue

    # Fallback to Ollama CLI if HTTP endpoints fail
    try:
        return call_ollama_cli(prompt, model=model, timeout=timeout)
    except Exception as e:
        diag = (
            f"All Ollama endpoints failed. Tried endpoints: {', '.join(endpoints)}. Last error: {last_exc}. CLI fallback error: {e}"
        )
        _log_diagnostic(diag)
        raise RuntimeError(diag) from e


def make_grade_prompt(question_text: str, solution_text: str, student_text: str, max_marks=None):
    instructions = [
        "You are an expert grader for university-level answers.",
        "Compare the student's answer with the solution and assign a numeric score.",
        "Return a JSON object with keys: score (number), max_score (number), feedback (short), strengths (list), weaknesses (list).",
    ]
    if max_marks:
        instructions.append(f"Max marks for this question: {max_marks}.")
    prompt = "\n\n".join(instructions) + "\n\n"
    prompt += "QUESTION:\n" + question_text + "\n\n"
    prompt += "MODEL SOLUTION:\n" + solution_text + "\n\n"
    prompt += "STUDENT ANSWER (extracted OCR):\n" + student_text + "\n\n"
    prompt += "Please produce only valid JSON as specified."
    return prompt


def make_grade_prompt_batch(student_text: str, solution_questions: List[dict]) -> str:
    instructions = [
        "You are an expert grader for university-level answers.",
        "Compare each student's answer with the provided solution and assign a numeric score.",
        "Return a single valid JSON object with a 'results' array.",
        "Each item in results must contain: qnum (number or string), score (number), max_score (number), feedback (string), strengths (array), weaknesses (array).",
        "Do not include any text outside the JSON object."
    ]
    prompt = "\n\n".join(instructions) + "\n\n"
    prompt += "STUDENT ANSWER (extracted OCR):\n" + student_text + "\n\n"
    prompt += "QUESTION SET:\n"
    for q in solution_questions:
        qnum = q.get("qnum")
        qtext = q.get("text", "")
        max_marks = q.get("max_marks") or 10
        prompt += f"Question {qnum}:\n{qtext}\nMax marks: {max_marks}\n\n"
    prompt += "Return only valid JSON."
    return prompt


def evaluate(student_text: str, solution_questions: List[dict], provider: str = "openai", model: str = OPENAI_DEFAULT_MODEL, api_key: Optional[str] = None, per_question: bool = False):
    """Evaluate either in a single batched request (default) or per-question.

    Set `per_question=True` to send one request per question (safer for large prompts or rate limits).
    """
    provider = provider.lower().strip()

    # Batched OpenAI path
    if provider == "openai" and not per_question:
        prompt = make_grade_prompt_batch(student_text, solution_questions)
        raw = call_llm(prompt, provider=provider, model=model, api_key=api_key)
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}

        results = []
        if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
            for item in parsed["results"]:
                if not isinstance(item, dict):
                    continue
                qnum = item.get("qnum")
                max_marks = item.get("max_score") or next((q.get("max_marks") for q in solution_questions if q.get("qnum") == qnum), None) or 10
                item.setdefault("max_score", max_marks)
                results.append({"qnum": qnum, "result": item})
        else:
            # Fallback if the model did not return a structured results array
            for q in solution_questions:
                max_marks = q.get("max_marks") or 10
                results.append({"qnum": q.get("qnum"), "result": {"raw": raw, "max_score": max_marks}})
        return results

    # Per-question path (works for any provider)
    results = []
    for q in solution_questions:
        qtext = q.get("text", "")
        max_marks = q.get("max_marks") or 10
        prompt = make_grade_prompt(f"Question {q.get('qnum')}", qtext, student_text, max_marks=max_marks)
        raw = call_llm(prompt, provider=provider, model=model, api_key=api_key)

        # try parse JSON
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        parsed.setdefault("max_score", max_marks)
        results.append({"qnum": q.get("qnum"), "result": parsed})

    return results


if __name__ == "__main__":
    # quick smoke test (requires OpenAI key or running Ollama)
    prompt = make_grade_prompt("Q1 sample", "Solution: A=1", "Student wrote A=1", max_marks=5)
    print(call_llm(prompt, provider="openai", model=OPENAI_DEFAULT_MODEL))