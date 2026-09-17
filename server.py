"""FastAPI server and web interface for Qwen-Image-Edit."""

from __future__ import annotations

import base64
import os
import sys
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests

from agent import UpcycleAgent
from qwen_client import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_HUGGINGFACE_BASE_URL,
    DEFAULT_MODEL,
    KNOWN_MODELS,
    QwenAPIError,
    QwenAuthError,
    QwenError,
    QwenImageEditClient,
    QwenRateLimitError,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

upcycle_agent = UpcycleAgent()

app = FastAPI(
    title="UpcycleAI Studio (NVIDIA Build API)",
    description="An intelligent agent for redesigning and upcycling old items with Qwen-Image-Edit",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JsonEditRequest(BaseModel):
    image: str = Field(..., description="Image as a Base64 Data URI or plain Base64 string")
    prompt: str = Field(..., description="Text editing instruction")
    style_hint: Optional[str] = Field(None, description="Preferred upcycling style")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    model: Optional[str] = Field(DEFAULT_MODEL, description="Model name")
    seed: Optional[int] = Field(None, description="Generation seed")
    steps: Optional[int] = Field(None, description="Number of diffusion steps")
    guidance_scale: Optional[float] = Field(None, description="Guidance scale")
    size: Optional[str] = Field(None, description="Image size")
    api_key: Optional[str] = Field(None, description="Custom API key (optional)")
    base_url: Optional[str] = Field(None, description="Custom Base URL (optional)")


@app.get("/api/health")
async def health_check():
    """Проверка конфигурации и доступности выбранного backend'а генерации."""
    base_url = (
        os.getenv("QWEN_IMAGE_EDIT_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or DEFAULT_CLOUD_BASE_URL
    ).rstrip("/")
    is_dashscope = "dashscope" in base_url or "aliyuncs.com" in base_url
    is_huggingface = "huggingface" in base_url or "hf.co" in base_url
    is_local = urlparse(base_url).hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if is_dashscope:
        env_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    elif is_huggingface:
        env_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    else:
        env_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY")

    backend_connected = False
    backend_status_code = None
    backend_message = ""
    if is_local:
        try:
            resp = requests.get(f"{base_url}/health/ready", timeout=2)
            backend_status_code = resp.status_code
            backend_connected = resp.status_code == 200
            backend_message = "Local Qwen NIM is ready" if backend_connected else "Local Qwen NIM is not ready or is running on a different port"
        except requests.RequestException as exc:
            backend_message = f"Local Qwen NIM is not running: {exc}"
    elif is_huggingface:
        backend_connected = bool(env_key)
        backend_message = "Hugging Face token found" if env_key else "Hugging Face requires an HF_TOKEN with Inference Providers access"
    elif is_dashscope:
        backend_connected = bool(env_key)
        backend_message = "DashScope key found" if env_key else "DashScope requires a DASHSCOPE_API_KEY / QwenCloud key"
    else:
        backend_connected = False
        backend_message = "The hosted NVIDIA Build endpoint does not serve qwen/qwen-image-edit; use a local NVIDIA NIM."

    return {
        "status": "ok",
        "has_env_key": bool(env_key),
        "masked_key": f"{env_key[:6]}...{env_key[-4:]}" if env_key and len(env_key) > 10 else None,
        "default_base_url": base_url,
        "default_model": os.getenv("NVIDIA_MODEL") or os.getenv("QWEN_IMAGE_EDIT_MODEL") or DEFAULT_MODEL,
        "supported_models": KNOWN_MODELS,
        "backend_connected": backend_connected,
        "backend_status_code": backend_status_code,
        "backend_message": backend_message,
    }


@app.get("/api/models")
async def get_models():
    """Список поддерживаемых моделей."""
    return {
        "models": [
            {
                "id": "qwen/qwen-image-edit",
                "name": "Qwen-Image-Edit (Cloud default)",
                "description": "20B multimodal image-editing model available through NVIDIA NIM, DashScope, or Hugging Face",
                "recommended": True,
            },
            {
                "id": "Qwen/Qwen-Image-Edit",
                "name": "Qwen-Image-Edit (Hugging Face)",
                "description": "Canonical model ID for Hugging Face Inference Providers",
                "recommended": False,
            },
            {
                "id": "qwen-image-edit",
                "name": "Qwen-Image-Edit (Standard NIM)",
                "description": "Standard model identifier for a local NVIDIA NIM container",
                "recommended": False,
            },
            {
                "id": "qwen/qwen-image-edit-2511",
                "name": "Qwen-Image-Edit v2511",
                "description": "Updated revision of the Qwen-Image-Edit model",
                "recommended": False,
            },
            {
                "id": "qwen-image-edit-nvpcb-ovsl2sl",
                "name": "Qwen-Image-Edit NVPCB (Physical AI)",
                "description": "Specialized NVIDIA model for PCB style transfer and industrial inspection",
                "recommended": False,
            },
        ]
    }


@app.get("/api/presets")
async def get_presets():
    """Универсальные пресеты для апсайклинга и редизайна старых вещей."""
    return {
        "presets": [
            {
                "id": "scandi_chair",
                "category": "Furniture",
                "title": "Scandi Minimalism",
                "icon": "🪑",
                "prompt": "Restore the old wooden chair in Scandinavian style with light natural wood, beige velour upholstery, and clean minimalist lines",
                "style": "scandi",
            },
            {
                "id": "loft_dresser",
                "category": "Furniture",
                "title": "Loft & Graphite",
                "icon": "🗄️",
                "prompt": "Transform the dresser in industrial loft style with a deep matte graphite finish, matte-black metal handles, and a natural wood top",
                "style": "loft",
            },
            {
                "id": "custom_jacket",
                "category": "Clothing",
                "title": "Custom Art Jacket",
                "icon": "🧥",
                "prompt": "Create a bold custom denim jacket with neat pop-art painting on the back, vintage patches, and light distressing",
                "style": "art_custom",
            },
            {
                "id": "kintsugi_vase",
                "category": "Decor",
                "title": "Japanese Kintsugi",
                "icon": "✨",
                "prompt": "Restore the old ceramic vase in traditional Japanese Kintsugi style with elegant gold seams and gold-leaf cracks",
                "style": "kintsugi",
            },
            {
                "id": "industrial_lamp",
                "category": "Decor",
                "title": "Loft Lamp",
                "icon": "💡",
                "prompt": "Turn the old table lamp into a stylish loft light made from dark metal pipes with a warm vintage Edison bulb",
                "style": "loft",
            },
            {
                "id": "custom_sneakers",
                "category": "Footwear",
                "title": "Custom Sneakers",
                "icon": "👟",
                "prompt": "Transform the old sneakers with a stylish monochrome repaint, contrasting laces, and clean designer graphics",
                "style": "art_custom",
            },
        ]
    }


@app.get("/api/samples")
async def get_samples():
    """Образцы старых вещей для моментального тестирования без загрузки своих фото."""
    return {
        "samples": [
            {
                "id": "chair",
                "title": "Vintage Chair",
                "subtitle": "Wood, 1970s",
                "category": "Furniture",
                "imageUrl": "/static/samples/old_chair.png",
                "prompt": "Restore the old wooden chair in Scandinavian style with light natural wood and beige velour fabric",
                "style": "scandi",
            },
            {
                "id": "dresser",
                "title": "Old Dresser",
                "subtitle": "Solid pine",
                "category": "Furniture",
                "imageUrl": "/static/samples/old_dresser.png",
                "prompt": "Repaint the dresser in industrial loft style with deep matte graphite and black handles",
                "style": "loft",
            },
            {
                "id": "jacket",
                "title": "Denim Jacket",
                "subtitle": "Distressed denim",
                "category": "Clothing",
                "imageUrl": "/static/samples/old_jacket.png",
                "prompt": "Create a bold custom denim jacket with pop-art graphics and patches",
                "style": "art_custom",
            },
            {
                "id": "lamp",
                "title": "Retro Lamp",
                "subtitle": "Metal and brass",
                "category": "Decor",
                "imageUrl": "/static/samples/old_lamp.png",
                "prompt": "Turn it into a loft lamp made from black metal pipes with a warm Edison bulb",
                "style": "loft",
            },
        ]
    }


@app.post("/api/edit")
async def edit_image_multipart(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    style_hint: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(DEFAULT_MODEL),
    seed: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    size: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    base_url: Optional[str] = Form(None),
):
    """
    Основной эндпоинт апсайклинг-редактирования через загрузку файла (multipart/form-data).
    """
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="The prompt cannot be empty.")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded file: {e}")

    # Агент 1: Генерируем детальный DIY-план и улучшаем визуальный промпт
    diy_plan = upcycle_agent.generate_diy_plan(user_prompt=prompt, style_hint=style_hint)
    enhanced_prompt = upcycle_agent.enhance_prompt(user_prompt=prompt, style_hint=style_hint)

    # Агент 2: Вызываем модель Qwen-Image-Edit
    client = QwenImageEditClient(
        api_key=api_key,
        base_url=base_url,
        default_model=model or DEFAULT_MODEL,
    )

    try:
        result = client.edit_image(
            image=content,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            model=model,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            size=size,
        )
        return {
            "success": True,
            "image": result.data_uri,
            "latency": round(result.latency_seconds, 2),
            "model": result.model,
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "diy_plan": diy_plan.to_dict(),
            "size_bytes": len(result.image_bytes),
            "is_demo": getattr(result, "is_demo", False),
        }
    except QwenAuthError as e:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "AUTH_ERROR",
                "message": str(e),
                "hint": "Check the key for the selected Base URL: Hugging Face uses HF_TOKEN, QwenCloud/DashScope uses DASHSCOPE_API_KEY, and local NVIDIA NIM uses an NGC/NVIDIA key when starting the container.",
            },
        )
    except QwenRateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"success": False, "error": "RATE_LIMIT", "message": str(e)},
        )
    except QwenAPIError as e:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "API_ERROR",
                "status_code": e.status_code,
                "message": str(e),
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "INTERNAL_ERROR", "message": str(e)},
        )


@app.post("/api/edit-json")
async def edit_image_json(req: JsonEditRequest):
    """
    Альтернативный эндпоинт апсайклинг-редактирования через JSON с Base64 Data URI.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="The prompt cannot be empty.")

    # Агент 1: Генерируем детальный DIY-план и улучшаем визуальный промпт
    diy_plan = upcycle_agent.generate_diy_plan(user_prompt=req.prompt, style_hint=req.style_hint)
    enhanced_prompt = upcycle_agent.enhance_prompt(user_prompt=req.prompt, style_hint=req.style_hint)

    # Агент 2: Вызываем модель Qwen-Image-Edit
    client = QwenImageEditClient(
        api_key=req.api_key,
        base_url=req.base_url,
        default_model=req.model or DEFAULT_MODEL,
    )

    try:
        result = client.edit_image(
            image=req.image,
            prompt=enhanced_prompt,
            negative_prompt=req.negative_prompt,
            model=req.model,
            seed=req.seed,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            size=req.size,
        )
        return {
            "success": True,
            "image": result.data_uri,
            "latency": round(result.latency_seconds, 2),
            "model": result.model,
            "prompt": req.prompt,
            "enhanced_prompt": enhanced_prompt,
            "diy_plan": diy_plan.to_dict(),
            "size_bytes": len(result.image_bytes),
            "is_demo": getattr(result, "is_demo", False),
        }
    except QwenAuthError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "AUTH_ERROR", "message": str(e)},
        )
    except QwenRateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"success": False, "error": "RATE_LIMIT", "message": str(e)},
        )
    except QwenAPIError as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": "API_ERROR", "message": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "INTERNAL_ERROR", "message": str(e)},
        )


# Раздача статики
if (STATIC_DIR / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    async def serve_fallback():
        return {
            "service": "Qwen-Image-Edit NVIDIA Build API Server",
            "status": "online",
            "docs": "/docs",
            "endpoints": ["/api/edit", "/api/edit-json", "/api/health", "/api/models", "/api/presets"],
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    print(f"🚀 Starting Qwen-Image-Edit server at http://{host}:{port}...")
    uvicorn.run("server:app", host=host, port=port, reload=True)
