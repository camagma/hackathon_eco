"""Tests for UpcycleAgent and the UpcycleAI server endpoints."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent import UpcycleAgent, DIYPlan
from server import app


class TestUpcycleAgent(unittest.TestCase):
    def setUp(self):
        self.agent = UpcycleAgent()

    def test_detect_style_and_category_furniture(self):
        style, cat, _ = self.agent.detect_style_and_category("an old wooden chair in Scandinavian style")
        self.assertIn("scandinavian", style["name"].lower())
        self.assertEqual(cat["category_title"], "Furniture & Interiors")

    def test_detect_style_and_category_clothing(self):
        style, cat, _ = self.agent.detect_style_and_category("a custom denim jacket with pop-art painting")
        self.assertIn("custom", style["name"].lower())
        self.assertEqual(cat["category_title"], "Clothing, Footwear & Textiles")

    def test_detect_style_and_category_decor_kintsugi(self):
        style, cat, _ = self.agent.detect_style_and_category("a ceramic Kintsugi vase with gold seams")
        self.assertIn("kintsugi", style["name"].lower())
        self.assertEqual(cat["category_title"], "Lighting & Home Decor")

    def test_enhance_prompt(self):
        enhanced = self.agent.enhance_prompt("repaint the dresser in loft style")
        self.assertIn("Transform this old item", enhanced)
        self.assertIn("Keep the original structure", enhanced)
        self.assertIn("dresser", enhanced)
        self.assertIn("loft", enhanced.lower())

    def test_generate_diy_plan(self):
        plan = self.agent.generate_diy_plan("an old wooden chair in Scandinavian style")
        self.assertIsInstance(plan, DIYPlan)
        self.assertTrue(len(plan.materials) > 0)
        self.assertTrue(len(plan.tools) > 0)
        self.assertTrue(len(plan.steps) >= 4)
        self.assertIn(plan.difficulty, ["Easy", "Intermediate", "Advanced"])

        plan_dict = plan.to_dict()
        self.assertIn("materials", plan_dict)
        self.assertIn("steps", plan_dict)
        self.assertIsInstance(plan_dict["steps"], list)
        self.assertIn("title", plan_dict["steps"][0])

    def test_generate_diy_plan_for_pants_to_shorts(self):
        plan = self.agent.generate_diy_plan("turn these pants into shorts")
        self.assertEqual(plan.category, "Clothing, Footwear & Textiles")
        self.assertEqual(plan.item_name, "Shorts Made from Old Pants")
        self.assertEqual(plan.difficulty, "Easy")
        self.assertEqual(plan.estimated_time, "20–45 minutes")
        self.assertEqual(plan.estimated_cost, "0–150 ₴")
        self.assertIn("scissors", " ".join(plan.tools).lower())

        steps_text = " ".join(f"{s.title} {s.desc} {s.markup} {s.checkpoint}" for s in plan.steps).lower()
        self.assertIn("allowance", steps_text)
        self.assertIn("cut", steps_text)
        self.assertIn("hem", steps_text)
        self.assertNotIn("furniture item", plan.item_name.lower())
        self.assertNotIn("primer", steps_text)
        self.assertNotIn("hardware", steps_text)
        self.assertTrue(all(s.visual_hint for s in plan.steps))
        self.assertTrue(all(s.markup for s in plan.steps))
        self.assertTrue(all(s.checkpoint for s in plan.steps))
        self.assertEqual(plan.steps[0].visual_type, "measure_line")
        self.assertEqual(plan.steps[2].visual_type, "cut_line")


class TestUpcycleServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_presets_endpoint(self):
        resp = self.client.get("/api/presets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("presets", data)
        self.assertTrue(len(data["presets"]) >= 4)
        first = data["presets"][0]
        self.assertIn("title", first)
        self.assertIn("prompt", first)
        self.assertIn("category", first)

    def test_samples_endpoint(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("samples", data)
        self.assertTrue(len(data["samples"]) >= 3)
        sample = data["samples"][0]
        self.assertIn("imageUrl", sample)
        self.assertIn("prompt", sample)

    @patch.dict("os.environ", {"DEMO_MODE": "true"})
    def test_edit_json_returns_diy_plan(self):
        payload = {
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "prompt": "Restore the old chair in loft style with black metal",
            "base_url": "http://localhost:8000/v1",
        }
        resp = self.client.post("/api/edit-json", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertIn("enhanced_prompt", data)
        self.assertIn("diy_plan", data)
        plan = data["diy_plan"]
        self.assertIn("materials", plan)
        self.assertIn("steps", plan)
        self.assertTrue(len(plan["steps"]) >= 4)


if __name__ == "__main__":
    unittest.main()
