"""
Qwen-Image-Edit Client for NVIDIA NIM, DashScope and Hugging Face
===================================================================
Поддержка модели редактирования изображений Qwen-Image-Edit
через локальные microservice контейнеры NVIDIA NIM, QwenCloud/DashScope
и Hugging Face Inference Providers.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from PIL import Image

# Загружаем переменные из .env при наличии
load_dotenv()

DEFAULT_CLOUD_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"
DEFAULT_GENAI_BASE_URL = "https://ai.api.nvidia.com/v1/genai"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_HUGGINGFACE_BASE_URL = "https://api-inference.huggingface.co"
DEFAULT_MODEL = "qwen/qwen-image-edit"
DEFAULT_HUGGINGFACE_MODEL = "Qwen/Qwen-Image-Edit"

KNOWN_MODELS = [
    "qwen/qwen-image-edit",
    "Qwen/Qwen-Image-Edit",
    "qwen-image-edit",
    "qwen/qwen-image-edit-2511",
    "qwen/qwen-image-edit-2509",
    "qwen-image-edit-nvpcb-ovsl2sl",
]


class QwenError(Exception):
    """Базовый класс исключений для Qwen-Image-Edit."""
    pass


class QwenAuthError(QwenError):
    """Ошибка авторизации (401/403). Проверьте NVIDIA_API_KEY."""
    pass


class QwenRateLimitError(QwenError):
    """Превышен лимит запросов (429)."""
    pass


class QwenAPIError(QwenError):
    """Ошибка выполнения запроса к API NVIDIA."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


@dataclass
class ImageEditResult:
    """Результат редактирования изображения."""
    image_bytes: bytes
    b64_json: str
    prompt: str
    model: str
    latency_seconds: float
    url: Optional[str] = None
    is_demo: bool = False

    @property
    def data_uri(self) -> str:
        """Возвращает Data URI строку для использования в HTML/CSS."""
        return f"data:image/png;base64,{self.b64_json}"

    def to_pil(self) -> Image.Image:
        """Конвертирует результат в объект PIL.Image."""
        return Image.open(io.BytesIO(self.image_bytes))

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Сохраняет отредактированное изображение в файл.
        
        :param output_path: Путь для сохранения (например 'result.png' или 'output.jpg')
        :return: Path сохраненного файла
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.image_bytes)
        return path


class QwenImageEditClient:
    """
    Клиент для вызова модели Qwen-Image-Edit через NVIDIA Build API / NIM.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        timeout: int = 120,
        enable_demo_fallback: bool = False,
    ):
        """
        Инициализация клиента.

        :param api_key: NVIDIA API Key. Если не указан, берется из NVIDIA_API_KEY или NGC_API_KEY
        :param base_url: Базовый URL API (по умолчанию https://integrate.api.nvidia.com/v1)
        :param default_model: Имя модели (по умолчанию qwen/qwen-image-edit)
        :param timeout: Таймаут запроса в секундах (по умолчанию 120)
        :param enable_demo_fallback: Автоматически переключаться в режим симуляции при 404
        """
        raw_base_url = (
            base_url
            or os.getenv("QWEN_IMAGE_EDIT_BASE_URL")
            or os.getenv("NVIDIA_BASE_URL")
            or DEFAULT_CLOUD_BASE_URL
        )
        self.base_url = raw_base_url.rstrip("/")
        if api_key:
            self.api_key = api_key
        elif "dashscope" in self.base_url or "aliyuncs.com" in self.base_url:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        elif "huggingface" in self.base_url or "hf.co" in self.base_url:
            self.api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        else:
            self.api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")
        self.default_model = default_model or os.getenv("NVIDIA_MODEL") or os.getenv("QWEN_IMAGE_EDIT_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
        env_fallback = os.getenv("DEMO_FALLBACK")
        if env_fallback is not None:
            self.enable_demo_fallback = env_fallback.lower() in ("1", "true", "yes")
        else:
            self.enable_demo_fallback = bool(enable_demo_fallback) and self.demo_mode
        self.genai_base_url = (os.getenv("NVIDIA_GENAI_BASE_URL") or DEFAULT_GENAI_BASE_URL).rstrip("/")

        self.session = requests.Session()

    def _get_headers(self, is_json: bool = True) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "Qwen-Image-Edit-Client/1.0",
        }
        if is_json:
            headers["Content-Type"] = "application/json"
        # Локальный NVIDIA NIM обычно не требует Authorization на inference.
        # Не отправляем сохраненный в браузере cloud-key в localhost, чтобы не ловить ложный 401.
        if self.api_key and not self._is_local_nim():
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _normalize_image_input(
        image: Union[str, Path, bytes, Image.Image, io.BytesIO]
    ) -> Tuple[bytes, str]:
        """
        Преобразует входное изображение любого поддерживаемого типа в байты и MIME-тип.
        """
        if isinstance(image, (str, Path)):
            path_or_str = str(image).strip()
            # Проверка, является ли строка base64 Data URI
            if path_or_str.startswith("data:image/"):
                header, encoded = path_or_str.split(",", 1)
                mime = header.split(";")[0].replace("data:", "")
                return base64.b64decode(encoded), mime
            
            # Проверка, если это локальный файл
            img_path = Path(path_or_str)
            if img_path.exists() and img_path.is_file():
                mime, _ = mimetypes.guess_type(str(img_path))
                mime = mime or "image/png"
                with open(img_path, "rb") as f:
                    return f.read(), mime
            
            raise FileNotFoundError(f"Image file not found: {path_or_str}")

        if isinstance(image, bytes):
            return image, "image/png"

        if isinstance(image, io.BytesIO):
            return image.getvalue(), "image/png"

        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            mime = f"image/{fmt.lower()}"
            return buf.getvalue(), mime

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def prepare_image_bytes(
        img_bytes: bytes,
        mime_type: str = "image/png",
        max_side: int = 1280,
    ) -> Tuple[bytes, str]:
        """Уменьшает слишком большие фото, чтобы API принял запрос."""
        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception:
            return img_bytes, mime_type or "image/png"

        img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"

    @staticmethod
    def _to_data_uri(img_bytes: bytes, mime_type: str = "image/png") -> str:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _is_local_nim(self) -> bool:
        parsed = urlparse(self.base_url)
        return parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    def _is_dashscope(self) -> bool:
        return "dashscope" in self.base_url or "aliyuncs.com" in self.base_url

    def _is_huggingface(self) -> bool:
        return "huggingface" in self.base_url or "hf.co" in self.base_url

    @staticmethod
    def _to_huggingface_model_id(model: Optional[str]) -> str:
        if not model or model.lower() in {"qwen/qwen-image-edit", "qwen-image-edit"}:
            return DEFAULT_HUGGINGFACE_MODEL
        return model

    def _validate_provider_key(self) -> None:
        if not self.api_key:
            return
        key = self.api_key.strip()
        if self._is_dashscope() and key.startswith("nvapi-"):
            raise QwenAuthError(
                "A QwenCloud/DashScope Base URL is selected, but an NVIDIA nvapi-... key was provided. "
                "DashScope requires a DASHSCOPE_API_KEY / QwenCloud API key. "
                "Alternatively, switch the Base URL to local NVIDIA NIM at http://localhost:8000/v1."
            )
        if self._is_huggingface() and not key.startswith("hf_"):
            raise QwenAuthError(
                "Hugging Face is selected, but the key does not look like an hf_... token. "
                "Create a token with Inference Providers access at https://huggingface.co/settings/tokens."
            )
        if "nvidia.com" in self.base_url and not key.startswith("nvapi-"):
            raise QwenAuthError(
                "An NVIDIA Base URL is selected, but the key does not look like an nvapi-... NVIDIA API key. "
                "Check the provider settings or use local NIM / DashScope."
            )

    def _network_error(self, endpoint: str, exc: requests.exceptions.RequestException) -> QwenAPIError:
        if self._is_local_nim():
            return QwenAPIError(
                f"Qwen-Image-Edit NIM is not running or cannot be reached at {self.base_url}. "
                "Start the local qwen/qwen-image-edit NVIDIA NIM on port 8000, "
                "or select Hugging Face / DashScope in Settings and enter the matching API key. "
                f"Technical error: {exc}",
                status_code=503,
            )
        return QwenAPIError(f"Network error while requesting {endpoint}: {exc}", status_code=503)

    def edit_image(
        self,
        image: Union[str, Path, bytes, Image.Image, io.BytesIO],
        prompt: str,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        size: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> ImageEditResult:
        """
        Выполнить редактирование изображения с помощью Qwen-Image-Edit.

        :param image: Исходное изображение (путь к файлу, байты, PIL Image или Data URI)
        :param prompt: Текстовая инструкция для редактирования (на русском или английском)
        :param negative_prompt: Негативный промпт (то, чего не должно быть на изображении)
        :param model: Имя модели (по умолчанию self.default_model)
        :param seed: Сид генерации для воспроизводимости
        :param steps: Количество шагов диффузии
        :param guidance_scale: Шкала следования промпту
        :param size: Желаемый размер ('1024x1024', '768x768' и т.д.)
        :param extra_params: Дополнительные специфичные параметры
        :return: Объект ImageEditResult
        """
        if not prompt or not prompt.strip():
            raise ValueError("The 'prompt' parameter cannot be empty.")

        target_model = model or self.default_model
        img_bytes, mime_type = self._normalize_image_input(image)
        img_bytes, mime_type = self.prepare_image_bytes(img_bytes, mime_type)
        data_uri = self._to_data_uri(img_bytes, mime_type)

        if not negative_prompt:
            negative_prompt = (
                "unchanged original image, simple enhancement, restoration only, upscaling only, "
                "sharpening only, denoise only, color correction only, same unmodified object, "
                "generic stock object, different material, different fabric texture, changed weave, "
                "changed seams, changed pockets, changed buttons, changed zipper, changed hardware, "
                "lost wear marks, lost scratches, lost patina, invented pattern, plastic-looking replacement, "
                "before after collage, watermark, text overlay, logo, low quality, blurry"
            )

        # Проверка наличия ключа при обращении к облачным API.
        is_cloud = "nvidia.com" in self.base_url
        is_dashscope = self._is_dashscope()
        is_huggingface = self._is_huggingface()
        self._validate_provider_key()
        if (is_cloud or is_dashscope or is_huggingface) and not self.api_key:
            raise QwenAuthError(
                "No API key was provided. Pass one to QwenImageEditClient(api_key='...'), "
                "set NVIDIA_API_KEY, NGC_API_KEY, DASHSCOPE_API_KEY, QWEN_API_KEY, HF_TOKEN, "
                "or HUGGINGFACE_API_KEY, or add it to the .env file."
            )

        if self.demo_mode:
            time.sleep(1.0)
            return self._generate_demo_result(img_bytes=img_bytes, prompt=prompt, model=target_model)

        start_time = time.time()
        errors: List[str] = []
        result: Optional[ImageEditResult] = None

        if is_huggingface:
            result = self._call_huggingface_endpoint(
                img_bytes=img_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                size=size,
                extra_params=extra_params,
            )

        if is_dashscope and result is None:
            result = self._call_dashscope_endpoint(
                data_uri=data_uri,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                size=size,
                extra_params=extra_params,
            )

        if is_cloud and result is None:
            result = self._call_genai_endpoint(
                img_bytes=img_bytes,
                mime_type=mime_type,
                data_uri=data_uri,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                extra_params=extra_params,
            )
            if result is None:
                errors.append("NVIDIA GenAI endpoint unavailable (404)")

        if result is None and self._is_local_nim():
            result = self._call_infer_endpoint(
                data_uri=data_uri,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                extra_params=extra_params,
            )

        if result is None:
            result = self._call_images_edits(
                img_bytes=img_bytes,
                mime_type=mime_type,
                data_uri=data_uri,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                size=size,
                extra_params=extra_params,
            )

        if result is None and not self._is_local_nim():
            result = self._call_infer_endpoint(
                data_uri=data_uri,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=target_model,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                extra_params=extra_params,
            )

        if result is None:
            if self.enable_demo_fallback:
                result = self._generate_demo_result(
                    img_bytes=img_bytes, prompt=prompt, model=target_model
                )
            else:
                raise QwenAPIError(
                    "Qwen-Image-Edit endpoint not found: /images/edits, /infer, and GenAI all returned 404. "
                    "For NVIDIA, this model is currently available as a self-hosted NIM: start the "
                    "qwen/qwen-image-edit container on port 8000 and use Base URL http://localhost:8000/v1. "
                    "The hosted NVIDIA endpoint https://integrate.api.nvidia.com/v1 does not serve this image-edit route. "
                    "For cloud use without local NIM, use Hugging Face Base URL "
                    "https://api-inference.huggingface.co with HF_TOKEN, or QwenCloud/DashScope Base URL "
                    "https://dashscope-intl.aliyuncs.com/api/v1 with DASHSCOPE_API_KEY. "
                    + (" ".join(errors) if errors else ""),
                    status_code=404,
                )

        latency = time.time() - start_time
        result.latency_seconds = latency
        return result

    def _call_dashscope_endpoint(
        self,
        data_uri: str,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        seed: Optional[int],
        size: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> Optional[ImageEditResult]:
        endpoint = self.base_url
        if not endpoint.endswith("/generation"):
            endpoint = f"{endpoint}/services/aigc/multimodal-generation/generation"

        dashscope_model = model.split("/", 1)[-1] if model.startswith("qwen/") else model
        parameters: Dict[str, Any] = {"n": 1}
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt[:500]
        if seed is not None:
            parameters["seed"] = seed
        if size and dashscope_model != "qwen-image-edit":
            parameters["size"] = size.replace("x", "*")
        if extra_params:
            parameters.update(extra_params)

        payload: Dict[str, Any] = {
            "model": dashscope_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": data_uri},
                            {"text": prompt},
                        ],
                    }
                ]
            },
            "parameters": parameters,
        }

        try:
            resp = self.session.post(
                endpoint,
                headers=self._get_headers(is_json=True),
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise self._network_error(endpoint, e) from e

        if resp.status_code == 404:
            return None
        if resp.status_code in (401, 403):
            raise QwenAuthError(f"Authentication error ({resp.status_code}): {resp.text}")
        if resp.status_code == 429:
            raise QwenRateLimitError(f"Rate limit exceeded (429): {resp.text}")
        if resp.status_code != 200:
            raise QwenAPIError(
                f"DashScope API returned status {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_data=resp.text,
            )

        return self._parse_image_response(resp.json(), prompt=prompt, model=dashscope_model)

    def _call_huggingface_endpoint(
        self,
        img_bytes: bytes,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        seed: Optional[int],
        steps: Optional[int],
        guidance_scale: Optional[float],
        size: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> ImageEditResult:
        try:
            from huggingface_hub import InferenceClient
            from huggingface_hub.errors import HfHubHTTPError
        except ImportError as exc:
            raise QwenAPIError(
                "The huggingface_hub package is required for the Hugging Face provider. "
                "Install dependencies with: python3 -m pip install -r requirements.txt",
                status_code=500,
            ) from exc

        hf_model = self._to_huggingface_model_id(model)
        provider = os.getenv("HF_PROVIDER") or "auto"
        target_size = None
        if size and "x" in size:
            try:
                width, height = [int(part) for part in size.lower().split("x", 1)]
                target_size = {"width": width, "height": height}
            except ValueError:
                target_size = None

        kwargs: Dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if extra_params:
            kwargs.update(extra_params)

        try:
            client = InferenceClient(
                provider=provider,
                api_key=self.api_key,
                timeout=float(self.timeout),
            )
            image = client.image_to_image(
                img_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                model=hf_model,
                target_size=target_size,
                **kwargs,
            )
        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403):
                raise QwenAuthError(f"Hugging Face authentication error ({status_code}): {exc}") from exc
            if status_code == 429:
                raise QwenRateLimitError(f"Hugging Face returned a rate limit error (429): {exc}") from exc
            raise QwenAPIError(
                f"Hugging Face Inference Providers returned error {status_code or ''}: {exc}",
                status_code=status_code,
            ) from exc
        except Exception as exc:
            raise QwenAPIError(
                "Hugging Face Inference Providers error. Check that HF_TOKEN has Inference Providers access, "
                "Qwen/Qwen-Image-Edit is available, and the account has sufficient credits or quota. "
                f"Technical error: {exc}",
                status_code=503,
            ) from exc

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        out_bytes = buf.getvalue()
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")
        return ImageEditResult(
            image_bytes=out_bytes,
            b64_json=out_b64,
            prompt=prompt,
            model=f"{hf_model} [Hugging Face]",
            latency_seconds=0.0,
        )

    def _call_genai_endpoint(
        self,
        img_bytes: bytes,
        mime_type: str,
        data_uri: str,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        seed: Optional[int],
        steps: Optional[int],
        guidance_scale: Optional[float],
        extra_params: Optional[Dict[str, Any]],
    ) -> Optional[ImageEditResult]:
        # NVIDIA Build for this NIM exposes /v1/images/edits and /v1/infer.
        # Keep this stub so older configs do not fail before the supported endpoints are tried.
        return None

    def _call_images_edits(
        self,
        img_bytes: bytes,
        mime_type: str,
        data_uri: str,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        seed: Optional[int],
        steps: Optional[int],
        guidance_scale: Optional[float],
        size: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> Optional[ImageEditResult]:
        """
        Вызов /v1/images/edits (поддерживает как multipart/form-data, так и application/json).
        """
        endpoint = f"{self.base_url}/images/edits"

        # Попытка через JSON с data_uri (стандарт NVIDIA Build Cloud)
        json_payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "image": data_uri,
            "response_format": "b64_json",
        }
        if negative_prompt:
            json_payload["negative_prompt"] = negative_prompt
        if seed is not None:
            json_payload["seed"] = seed
        if steps is not None:
            json_payload["num_inference_steps"] = steps
        if guidance_scale is not None:
            json_payload["guidance_scale"] = guidance_scale
        if size:
            json_payload["size"] = size
        if extra_params:
            json_payload.update(extra_params)

        try:
            resp = self.session.post(
                endpoint,
                headers=self._get_headers(is_json=True),
                json=json_payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise self._network_error(endpoint, e) from e

        if resp.status_code == 404:
            # Эндпоинт отсутствует на этом сервере, пробуем другие варианты
            return None

        if resp.status_code in (401, 403):
            raise QwenAuthError(
                f"Authentication error ({resp.status_code}): {resp.text}. "
                "Check the NVIDIA API key at https://build.nvidia.com"
            )
        if resp.status_code == 429:
            raise QwenRateLimitError(f"Rate limit exceeded (429): {resp.text}")

        # Если JSON вернул 415/422 или требует multipart/form-data:
        if resp.status_code in (415, 422) and "multipart" in resp.text.lower():
            return self._call_images_edits_multipart(
                img_bytes=img_bytes,
                mime_type=mime_type,
                prompt=prompt,
                model=model,
                seed=seed,
                size=size,
                extra_params=extra_params,
            )

        if resp.status_code != 200:
            raise QwenAPIError(
                f"API returned status {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_data=resp.text,
            )

        return self._parse_image_response(resp.json(), prompt=prompt, model=model)

    def _call_images_edits_multipart(
        self,
        img_bytes: bytes,
        mime_type: str,
        prompt: str,
        model: str,
        seed: Optional[int],
        size: Optional[str],
        extra_params: Optional[Dict[str, Any]],
    ) -> ImageEditResult:
        endpoint = f"{self.base_url}/images/edits"
        files = {
            "image": ("input_image.png", img_bytes, mime_type),
        }
        data: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if seed is not None:
            data["seed"] = str(seed)
        if size:
            data["size"] = size
        if extra_params:
            for k, v in extra_params.items():
                data[k] = str(v)

        try:
            resp = self.session.post(
                endpoint,
                headers=self._get_headers(is_json=False),
                files=files,
                data=data,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise self._network_error(endpoint, e) from e

        if resp.status_code != 200:
            raise QwenAPIError(
                f"API returned status {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_data=resp.text,
            )

        return self._parse_image_response(resp.json(), prompt=prompt, model=model)

    def _call_infer_endpoint(
        self,
        data_uri: str,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        seed: Optional[int],
        steps: Optional[int],
        guidance_scale: Optional[float],
        extra_params: Optional[Dict[str, Any]],
    ) -> ImageEditResult:
        """
        Вызов эндпоинта /v1/infer (для локального NIM или кастомных эндпоинтов).
        """
        endpoint = f"{self.base_url}/infer"
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image": data_uri,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        if steps is not None:
            payload["num_inference_steps"] = steps
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if extra_params:
            payload.update(extra_params)

        try:
            resp = self.session.post(
                endpoint,
                headers=self._get_headers(is_json=True),
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise self._network_error(endpoint, e) from e

        if resp.status_code in (401, 403):
            raise QwenAuthError(f"Authentication error ({resp.status_code}): {resp.text}")
        if resp.status_code == 429:
            raise QwenRateLimitError(f"Rate limit exceeded (429): {resp.text}")
        if resp.status_code == 404 and self.enable_demo_fallback:
            return self._generate_demo_result(
                data_uri=data_uri,
                prompt=prompt,
                model=model,
            )
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise QwenAPIError(
                f"Endpoint {endpoint} returned error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                response_data=resp.text,
            )

        return self._parse_image_response(resp.json(), prompt=prompt, model=model)

    def _generate_demo_result(
        self,
        img_bytes: Optional[bytes] = None,
        data_uri: Optional[str] = None,
        prompt: str = "",
        model: str = DEFAULT_MODEL,
    ) -> ImageEditResult:
        """
        Генерирует демонстрационный результат симуляции (когда удаленный сервер возвращает 404).
        """
        if not img_bytes and data_uri:
            try:
                b64_part = data_uri.split(",", 1)[-1]
                img_bytes = base64.b64decode(b64_part)
            except Exception:
                img_bytes = b""

        edited_bytes, edited_b64 = self._apply_demo_transformation(img_bytes or b"", prompt)
        return ImageEditResult(
            image_bytes=edited_bytes,
            b64_json=edited_b64,
            prompt=prompt,
            model=f"{model} [Demo Preview]",
            latency_seconds=1.2,
            is_demo=True,
        )

    @staticmethod
    def _apply_demo_transformation(img_bytes: bytes, prompt: str) -> Tuple[bytes, str]:
        from PIL import ImageEnhance, ImageFilter, ImageDraw, ImageFont
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        except Exception:
            img = Image.new("RGBA", (512, 512), color=(30, 41, 59, 255))

        w, h = img.size
        p_lower = prompt.lower()

        if any(k in p_lower for k in ["лофт", "loft", "металл", "графит", "черн", "industrial"]):
            # Лофт: глубокий матовый графит, приглушенная сатурация, высокий контраст
            gray = img.convert("L")
            gray = ImageEnhance.Contrast(gray).enhance(1.4)
            r = ImageEnhance.Brightness(gray).enhance(0.9)
            g = ImageEnhance.Brightness(gray).enhance(0.92)
            b = ImageEnhance.Brightness(gray).enhance(0.98)
            img = Image.merge("RGBA", (r, g, b, img.split()[3]))
            img = img.filter(ImageFilter.SHARPEN)
        elif any(k in p_lower for k in ["сканди", "scandi", "эко", "дерево", "wood", "минимализм"]):
            # Сканди: светлые теплые тона, естественное дерево, мягкий свет
            r, g, b, a = img.split()
            r = ImageEnhance.Brightness(r).enhance(1.15)
            g = ImageEnhance.Brightness(g).enhance(1.10)
            b = ImageEnhance.Brightness(b).enhance(0.95)
            img = Image.merge("RGBA", (r, g, b, a))
            img = ImageEnhance.Color(img).enhance(1.25)
            img = ImageEnhance.Contrast(img).enhance(1.1)
        elif any(k in p_lower for k in ["золот", "kintsugi", "кинцуги", "поталь", "gold"]):
            # Кинцуги/золото: насыщенный теплый оттенок с акцентом золотистого свечения
            r, g, b, a = img.split()
            r = ImageEnhance.Brightness(r).enhance(1.3)
            g = ImageEnhance.Brightness(g).enhance(1.18)
            b = ImageEnhance.Brightness(b).enhance(0.7)
            img = Image.merge("RGBA", (r, g, b, a))
            img = ImageEnhance.Contrast(img).enhance(1.3)
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif any(k in p_lower for k in ["кастом", "арт", "роспись", "ярк", "pop-art", "граффити"]):
            # Арт-кастом: сочные поп-арт цвета
            img = ImageEnhance.Color(img).enhance(1.8)
            img = ImageEnhance.Contrast(img).enhance(1.35)
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        elif any(k in p_lower for k in ["реставрация", "восстанов", "винтаж", "лак", "глянец"]):
            # Реставрация: глубина дерева, насыщенность, четкость
            img = ImageEnhance.Color(img).enhance(1.4)
            img = ImageEnhance.Contrast(img).enhance(1.25)
            img = img.filter(ImageFilter.SHARPEN)
        elif any(k in p_lower for k in ["киберпанк", "cyberpunk", "неон", "neon"]):
            r, g, b, a = img.split()
            r = ImageEnhance.Brightness(r).enhance(1.2)
            b = ImageEnhance.Brightness(b).enhance(1.4)
            g = ImageEnhance.Brightness(g).enhance(0.8)
            img = Image.merge("RGBA", (r, g, b, a))
            img = ImageEnhance.Color(img).enhance(1.6)
            img = ImageEnhance.Contrast(img).enhance(1.2)
        elif any(k in p_lower for k in ["черно-бел", "ч/б", "bw", "monochrome", "чб"]):
            gray = img.convert("L")
            img = ImageEnhance.Contrast(gray).enhance(1.3).convert("RGBA")
        else:
            img = ImageEnhance.Color(img).enhance(1.35)
            img = ImageEnhance.Contrast(img).enhance(1.2)
            img = img.filter(ImageFilter.SHARPEN)

        # Оверлей бейджа
        banner_h = max(28, int(h * 0.075))
        banner = Image.new("RGBA", (w, banner_h), (15, 23, 42, 215))
        img.paste(banner, (0, h - banner_h), banner)

        draw = ImageDraw.Draw(img)
        badge_text = f"Qwen-Image-Edit (Demo): {prompt}"
        if len(badge_text) > 42:
            badge_text = badge_text[:39] + "..."
        try:
            font = ImageFont.load_default()
            draw.text((10, h - banner_h + int(banner_h * 0.25)), badge_text, fill=(56, 189, 248, 255), font=font)
        except Exception:
            pass

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        out_bytes = buf.getvalue()
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")
        return out_bytes, out_b64

    def _parse_image_response(
        self,
        data: Dict[str, Any],
        prompt: str,
        model: str,
    ) -> ImageEditResult:
        """
        Парсит различные форматы ответа NVIDIA NIM / OpenAI Image API.
        """
        b64_str: Optional[str] = None
        img_url: Optional[str] = None

        # Формат QwenCloud/DashScope: {"output": {"choices": [{"message": {"content": [{"image": "..."}]}}]}}
        output = data.get("output")
        if isinstance(output, dict):
            for choice in output.get("choices") or []:
                message = choice.get("message") if isinstance(choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict) and item.get("image"):
                        img_url = item["image"]
                        break
                if img_url:
                    break

        # Формат OpenAI/NVIDIA: {"data": [{"b64_json": "..."}]} или {"data": [{"url": "..."}]}
        if not img_url and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            item = data["data"][0]
            if isinstance(item, dict):
                b64_str = item.get("b64_json") or item.get("base64")
                img_url = item.get("url")
            elif isinstance(item, str):
                b64_str = item

        # Формат NIM: {"artifacts": [{"base64": "..."}]}
        elif "artifacts" in data and isinstance(data["artifacts"], list) and len(data["artifacts"]) > 0:
            item = data["artifacts"][0]
            if isinstance(item, dict):
                b64_str = item.get("base64") or item.get("b64_json")
                img_url = item.get("url")

        # Формат прямого объекта: {"image": "..."} или {"b64_json": "..."}
        elif "image" in data:
            val = data["image"]
            if isinstance(val, str):
                b64_str = val
        elif "b64_json" in data:
            b64_str = data["b64_json"]

        # Если есть URL, но нет base64 - загружаем байты по URL
        if not b64_str and img_url:
            try:
                img_resp = requests.get(img_url, timeout=30)
                img_bytes = img_resp.content
                b64_str = base64.b64encode(img_bytes).decode("utf-8")
                return ImageEditResult(
                    image_bytes=img_bytes,
                    b64_json=b64_str,
                    prompt=prompt,
                    model=model,
                    latency_seconds=0.0,
                    url=img_url,
                )
            except Exception as e:
                raise QwenAPIError(f"Could not download the generated image from {img_url}: {e}")

        if not b64_str:
            raise QwenAPIError(
                f"Could not extract an image from the API response. Response: {str(data)[:400]}",
                response_data=data,
            )

        # Очистка data:image/png;base64,... префикса при наличии
        if "," in b64_str and "base64" in b64_str[:50]:
            b64_str = b64_str.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(b64_str)
        except Exception as e:
            raise QwenAPIError(f"Could not decode the Base64 response: {e}")

        return ImageEditResult(
            image_bytes=raw_bytes,
            b64_json=b64_str,
            prompt=prompt,
            model=model,
            latency_seconds=0.0,
            url=img_url,
        )

    def check_health(self) -> Dict[str, Any]:
        """
        Проверка доступности API и корректности ключа.
        """
        status: Dict[str, Any] = {
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
            "masked_key": f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 10 else None,
            "default_model": self.default_model,
            "is_cloud": "nvidia.com" in self.base_url or self._is_dashscope() or self._is_huggingface(),
            "provider": "huggingface" if self._is_huggingface() else "dashscope" if self._is_dashscope() else "nvidia",
        }

        # Пытаемся сделать легкий запрос к /models
        try:
            resp = self.session.get(
                f"{self.base_url}/models",
                headers=self._get_headers(is_json=True),
                timeout=10,
            )
            status["connected"] = resp.status_code == 200
            status["status_code"] = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                model_ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
                status["available_models_count"] = len(model_ids)
                status["supports_qwen"] = any("qwen" in m.lower() for m in model_ids)
        except Exception as e:
            status["connected"] = False
            status["error"] = str(e)

        return status
