import json
import time
import httpx

class VLLMOpenAIAdapter:
    def __init__(self, base_url, model, api_key, timeout_s=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def generate(self, client, prompt, max_tokens, temperature=0.0):
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": "Bearer " + self.api_key}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        first_text_at = None
        usage = None
        async with client.stream("POST", url, headers=headers, json=payload,
                                 timeout=self.timeout_s) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if content and first_text_at is None:
                        first_text_at = time.perf_counter()
        end = time.perf_counter()
        latency = end-start
        ttft = first_text_at-start if first_text_at else None
        completion_tokens = usage.get("completion_tokens") if usage else None
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        tpot = None
        if ttft is not None and completion_tokens and completion_tokens > 1:
            tpot = max(0.0, latency-ttft)/(completion_tokens-1)
        return {
            "ok": True, "latency_s": latency, "ttft_s": ttft, "tpot_s": tpot,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        }
