"""Multi-model LLM client for Bailian DashScope.

Three clients:
- TextClient:    qwen3.6-plus for content generation (chat_text / chat_json)
- VisionClient:  qwen3-vl-plus for visual decision making
- ImageGenClient: wan2.6-t2i for AI image generation

All share the same API key and base_url (OpenAI-compatible mode for text/vision,
dedicated multimodal-generation endpoint for image gen).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    raw: Any = None

    def as_json(self) -> Any:
        return _safe_json(self.text)


def _safe_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        j = t.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(t[i: j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Failed to parse JSON from LLM output: {text[:200]!r}")


# ---------------------------------------------------------------------------
# TextClient  (qwen3.6-plus)
# ---------------------------------------------------------------------------

class TextClient:
    def __init__(self) -> None:
        self.model = settings.text_model
        self._openai = None

    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            if not settings.dashscope_api_key:
                raise LLMError("DASHSCOPE_API_KEY not set")
            self._openai = OpenAI(
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )
        return self._openai

    def chat_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        client = self._ensure_openai()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise LLMError(f"Text chat failed: {e}") from e
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, raw=resp)

    def chat_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> Any:
        client = self._ensure_openai()
        messages: list[dict[str, str]] = []
        sys_prompt = (
            (system or "")
            + "\n\nReturn ONLY a valid JSON object or array. No markdown, no commentary."
        ).strip()
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        text = resp.choices[0].message.content or ""
        return _safe_json(text)


# ---------------------------------------------------------------------------
# VisionClient  (qwen3-vl-plus)
# ---------------------------------------------------------------------------

class VisionClient:
    def __init__(self) -> None:
        self.model = settings.vision_model
        self._openai = None

    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            if not settings.dashscope_api_key:
                raise LLMError("DASHSCOPE_API_KEY not set")
            self._openai = OpenAI(
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )
        return self._openai

    def analyze(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> LLMResponse:
        client = self._ensure_openai()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise LLMError(f"Vision chat failed: {e}") from e
        text = resp.choices[0].message.content or ""
        return LLMResponse(text=text, raw=resp)

    def analyze_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> Any:
        client = self._ensure_openai()
        messages: list[dict[str, str]] = []
        sys_prompt = (
            (system or "")
            + "\n\nReturn ONLY a valid JSON object. No markdown, no commentary."
        ).strip()
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        text = resp.choices[0].message.content or ""
        return _safe_json(text)


# ---------------------------------------------------------------------------
# ImageGenClient  (wan2.6-t2i)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedImage:
    url: str
    prompt: str
    local_path: str | None = None


class ImageGenClient:
    def __init__(self) -> None:
        self.model = settings.image_gen_model
        self._session = None

    def _ensure_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.dashscope_api_key}",
            })
        return self._session

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1920*1080",
        negative_prompt: str | None = None,
        timeout: int = 120,
    ) -> GeneratedImage:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": {
                "size": size,
                "prompt_extend": True,
                "watermark": False,
            },
        }
        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        session = self._ensure_session()
        try:
            resp = session.post(
                settings.image_gen_base_url,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"Image gen request failed: {e}") from e

        if resp.status_code != 200:
            raise LLMError(
                f"Image gen HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        try:
            choices = data["output"]["choices"]
            image_url = choices[0]["message"]["content"][0]["image"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected image gen response: {data}") from e

        return GeneratedImage(url=image_url, prompt=prompt)

    def generate_and_download(
        self,
        prompt: str,
        output_dir: str | Path,
        *,
        size: str = "1920*1080",
        filename: str | None = None,
    ) -> GeneratedImage:
        gen = self.generate(prompt, size=size)
        img = self._download_image(gen, output_dir, filename)
        return img

    def _download_image(
        self,
        gen: GeneratedImage,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> GeneratedImage:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"ai_img_{int(time.time() * 1000)}.png"
        local = out / filename

        try:
            r = requests.get(gen.url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Image download failed: {e}") from e

        local.write_bytes(r.content)
        logger.info("AI image saved: %s", local)
        return GeneratedImage(url=gen.url, prompt=gen.prompt, local_path=str(local))


# ---------------------------------------------------------------------------
# Convenience: BailianClient bundles all three
# ---------------------------------------------------------------------------

class BailianClient:
    def __init__(self) -> None:
        self.text = TextClient()
        self.vision = VisionClient()
        self.image_gen = ImageGenClient()


_default_client: BailianClient | None = None


def get_client() -> BailianClient:
    global _default_client
    if _default_client is None:
        _default_client = BailianClient()
    return _default_client
