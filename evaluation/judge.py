"""Score completions through an already-running `ollama serve`."""
import json
import re
import urllib.error
import urllib.request

import config

OLLAMA_TIMEOUT_S = 600


def parse_score(text: str):
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|?/?think\|?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("`").strip()
    if not cleaned:
        return None
    first = cleaned.split()[0].rstrip(".,:;").upper()
    if first in {"CODE", "REFUSAL"}:
        return None
    match = re.search(r"\b(100|\d{1,2})\b", cleaned)
    return int(match.group(1)) if match else None


class OllamaJudge:
    def __init__(self, model: str, prompt_template: str, host: str = config.OLLAMA_HOST):
        self.model = model
        self.prompt_template = prompt_template
        self.host = host.rstrip("/")

    def judge(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 32},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_S) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach ollama serve at {self.host}. "
                "Leave `ollama serve` running; this script does not start it."
            ) from exc
        content = payload.get("message", {}).get("content", "") or ""
        return parse_score(content)

    def __call__(self, **kwargs):
        return self.judge(**kwargs)


def unload_ollama(model: str | None = None, host: str = config.OLLAMA_HOST) -> None:
    """Drop the judge weights from GPU so Unsloth can load the 3B subject."""
    body = {"model": model or config.JUDGE_MODEL, "keep_alive": 0}
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except urllib.error.URLError:
        pass
