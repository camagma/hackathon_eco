"""Unit tests for the Qwen-Image-Edit client and FastAPI server."""

import base64
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from fastapi.testclient import TestClient

from qwen_client import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_HUGGINGFACE_BASE_URL,
    DEFAULT_MODEL,
    ImageEditResult,
    QwenAPIError,
    QwenAuthError,
    QwenError,
    QwenImageEditClient,
    QwenRateLimitError,
)
from server import app


class TestQwenImageEditClient(unittest.TestCase):
    def setUp(self):
        self.dummy_key = "nvapi-test-key-1234567890"
        self.client = QwenImageEditClient(api_key=self.dummy_key, base_url=DEFAULT_CLOUD_BASE_URL)

        # Create a 64x64 test image.
        self.test_img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        self.test_img.save(buf, format="PNG")
        self.test_img_bytes = buf.getvalue()
        self.test_b64 = base64.b64encode(self.test_img_bytes).decode("utf-8")
        self.test_data_uri = f"data:image/png;base64,{self.test_b64}"

    def test_normalize_image_input(self):
        # 1. Bytes
        b, mime = self.client._normalize_image_input(self.test_img_bytes)
        self.assertEqual(b, self.test_img_bytes)
        self.assertEqual(mime, "image/png")

        # 2. BytesIO
        bio = io.BytesIO(self.test_img_bytes)
        b, mime = self.client._normalize_image_input(bio)
        self.assertEqual(b, self.test_img_bytes)

        # 3. PIL Image
        b, mime = self.client._normalize_image_input(self.test_img)
        self.assertTrue(len(b) > 0)

        # 4. Data URI
        b, mime = self.client._normalize_image_input(self.test_data_uri)
        self.assertEqual(b, self.test_img_bytes)
        self.assertEqual(mime, "image/png")

        # 5. Non-existent file
        with self.assertRaises(FileNotFoundError):
            self.client._normalize_image_input("non_existent_file_12345.png")

    def test_empty_prompt_validation(self):
        with self.assertRaises(ValueError):
            self.client.edit_image(image=self.test_img_bytes, prompt="")

    @patch.dict("os.environ", {"NVIDIA_API_KEY": "", "NGC_API_KEY": ""})
    def test_missing_api_key_on_cloud(self):
        cloud_client = QwenImageEditClient(api_key="", base_url=DEFAULT_CLOUD_BASE_URL)
        cloud_client.api_key = None
        with self.assertRaises(QwenAuthError):
            cloud_client.edit_image(image=self.test_img_bytes, prompt="Add sunglasses")

    @patch("requests.Session.post")
    def test_successful_edit_openai_format(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"b64_json": self.test_b64}]
        }
        mock_post.return_value = mock_resp

        res = self.client.edit_image(
            image=self.test_img,
            prompt="Turn into cyberpunk style",
            seed=42,
        )

        self.assertIsInstance(res, ImageEditResult)
        self.assertEqual(res.b64_json, self.test_b64)
        self.assertEqual(res.prompt, "Turn into cyberpunk style")
        self.assertEqual(res.model, DEFAULT_MODEL)
        self.assertTrue(res.data_uri.startswith("data:image/png;base64,"))

        # Test PIL conversion.
        pil_img = res.to_pil()
        self.assertEqual(pil_img.size, (64, 64))

    @patch("requests.Session.post")
    def test_successful_edit_nim_artifacts_format(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "artifacts": [{"base64": self.test_b64}]
        }
        mock_post.return_value = mock_resp

        res = self.client.edit_image(
            image=self.test_img_bytes,
            prompt="Add cat ears",
        )
        self.assertEqual(res.b64_json, self.test_b64)


    @patch("huggingface_hub.InferenceClient")
    def test_successful_edit_huggingface_provider(self, mock_hf_client):
        out_img = Image.new("RGB", (32, 32), color="green")
        mock_hf_client.return_value.image_to_image.return_value = out_img

        hf_client = QwenImageEditClient(
            api_key="hf_test_token",
            base_url=DEFAULT_HUGGINGFACE_BASE_URL,
        )
        res = hf_client.edit_image(
            image=self.test_img_bytes,
            prompt="Turn this old chair into a modern stool",
            model="qwen/qwen-image-edit",
            steps=25,
            guidance_scale=4.0,
        )

        self.assertIsInstance(res, ImageEditResult)
        self.assertIn("Qwen/Qwen-Image-Edit", res.model)
        self.assertTrue(res.data_uri.startswith("data:image/png;base64,"))
        generated = Image.open(io.BytesIO(res.image_bytes))
        self.assertEqual(generated.size, (32, 32))
        mock_hf_client.return_value.image_to_image.assert_called_once()

    @patch("requests.Session.post")
    def test_auth_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized: Invalid API key"
        mock_post.return_value = mock_resp

        with self.assertRaises(QwenAuthError):
            self.client.edit_image(image=self.test_img_bytes, prompt="Make it neon")

    @patch("requests.Session.post")
    def test_rate_limit_error_handling(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        mock_post.return_value = mock_resp

        with self.assertRaises(QwenRateLimitError):
            self.client.edit_image(image=self.test_img_bytes, prompt="Make it neon")

    @patch("requests.Session.post")
    def test_fallback_to_infer_on_404(self, mock_post):
        # First /images/edits call -> 404; second /infer call -> 200.
        resp_404 = MagicMock()
        resp_404.status_code = 404

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"image": self.test_b64}

        mock_post.side_effect = [resp_404, resp_200]

        res = self.client.edit_image(image=self.test_img_bytes, prompt="Test fallback")
        self.assertEqual(res.b64_json, self.test_b64)
        self.assertEqual(mock_post.call_count, 2)


class TestFastAPIServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("supported_models", data)

    def test_models_endpoint(self):
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)
        self.assertTrue(any(m["id"] == "qwen/qwen-image-edit" for m in data["models"]))

    def test_presets_endpoint(self):
        resp = self.client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)
        self.assertTrue(len(data["presets"]) > 0)


if __name__ == "__main__":
    unittest.main()
