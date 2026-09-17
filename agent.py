"""
UpcycleAI Agent: an intelligent agent for redesigning and upcycling old items.

The agent performs two key tasks:
1. Enhanced Prompt: turns a short user request into a detailed prompt for
   Qwen-Image-Edit while preserving geometry and photorealistic materials.
2. DIY Plan & Materials: generates a practical project plan with materials,
   tools, step-by-step instructions, difficulty, time, and environmental impact.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DIYStep:
    number: int
    title: str
    desc: str
    visual_hint: str = ""
    markup: str = ""
    checkpoint: str = ""
    visual_type: str = "progress"


@dataclass
class ItemAnalysis:
    item_name: str
    category_key: str
    category_title: str
    materials: str
    colors: str
    condition: str
    description: str
    source: str = "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DIYPlan:
    item_name: str
    category: str
    style: str
    difficulty: str  # "Easy", "Intermediate", "Advanced"
    estimated_time: str
    estimated_cost: str
    eco_impact: str
    materials: List[str]
    tools: List[str]
    steps: List[DIYStep]
    pro_tip: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["steps"] = [asdict(s) if isinstance(s, DIYStep) else s for s in self.steps]
        return data


# Upcycling style knowledge base. Russian keywords are retained for compatibility.
STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "scandi": {
        "name": "Scandinavian Eco-Minimalism",
        "keywords": ["scandi", "scandinavian", "eco", "wood", "minimalism", "light", "natural", "сканди", "эко", "дерево", "минимализм", "светлый", "натуральн"],
        "prompt_suffix": "Scandi minimalist upcycle, natural light wood finish, muted warm beige tones, clean modern lines, eco-friendly aesthetic, high-end interior photography, sharp focus",
        "materials": ["Light-oak hardwax oil", "White matte acrylic paint", "Natural linen or cotton fabric", "Sanding pads"],
        "tools": ["Orbital sander or P180/P320 sandpaper", "Paint roller", "Synthetic-bristle brush"],
        "pro_tip": "Do not hide attractive wood grain under opaque paint—use a lightening oil or matte wax instead.",
    },
    "loft": {
        "name": "Loft / Industrial",
        "keywords": ["loft", "metal", "industrial", "graphite", "charcoal", "black", "steel", "concrete", "лофт", "металл", "индустриальн", "графит", "черный", "сталь", "бетон"],
        "prompt_suffix": "Industrial loft style upcycle, dark charcoal graphite matte finish, raw textured metal accents, matte black hardware, exposed craftsmanship, atmospheric studio lighting",
        "materials": ["Matte charcoal or graphite paint", "3-in-1 metal primer enamel", "Matte black metal handles or legs", "Copper or rust-effect patina"],
        "tools": ["Wire brush", "P120 sandpaper", "Painter's tape", "Screwdriver or drill driver"],
        "pro_tip": "The contrast between dark unfinished wood and rugged black metal is the signature of loft style.",
    },
    "kintsugi": {
        "name": "Kintsugi / Golden Restoration",
        "keywords": ["gold", "golden", "kintsugi", "gold leaf", "crack", "japanese", "wabi-sabi", "золот", "кинцуги", "поталь", "трещин", "японск", "ваби-саби"],
        "prompt_suffix": "Japanese Kintsugi inspired artful upcycling, refined gold leaf seam lines, delicate cracked texture filled with golden lacquer, wabi-sabi philosophy, luxurious craftsmanship, museum lighting",
        "materials": ["Gold leaf or antique-gold metallic acrylic enamel", "Gold-leaf size adhesive", "Two-part epoxy", "Protective shellac finish"],
        "tools": ["Soft fan brushes", "Cotton swabs", "Gold-leaf tweezers", "Protective gloves"],
        "pro_tip": "Kintsugi does not hide cracks and chips; it highlights them in gold as part of the object's history.",
    },
    "art_custom": {
        "name": "Art Custom & Pop Art",
        "keywords": ["custom", "art", "painted", "bright", "pop art", "patch", "graffiti", "streetwear", "кастом", "арт", "роспись", "ярк", "поп-арт", "нашивк", "граффити"],
        "prompt_suffix": "Vibrant custom streetwear redesign, hand-painted pop-art graphics, bold contrasting color blocks, artistic bespoke details, creative designer aesthetic, trendy lookbook shot",
        "materials": ["Special acrylic paint for fabric or wood", "Fabric medium for clothing", "Iron-on patches and metal studs", "Protective clear coat or spray"],
        "tools": ["Fine art brushes (sizes 1 and 3)", "Stencils", "Iron for heat-setting fabric", "Tailor's scissors"],
        "pro_tip": "When painting clothing, always place cardboard inside so the paint cannot bleed through to the back.",
    },
    "matte_minimal": {
        "name": "Matte Monochrome",
        "keywords": ["matte", "monochrome", "minimalism", "black", "white", "minimal", "modern", "матовый", "монохром", "минимализм", "черный", "белый", "лаконичн", "современ"],
        "prompt_suffix": "Ultra-matte monochrome modern redesign, silky smooth chalk paint finish, sophisticated designer hardware, contemporary sleek aesthetics, architectural catalog shot",
        "materials": ["Premium chalk paint", "Protective chalk-paint wax", "Designer hardware in brass or black aluminum"],
        "tools": ["Round chalk-paint brush", "Lint-free cloth for buffing wax", "Fine P400 sanding pad"],
        "pro_tip": "Chalk paint adheres beautifully to old varnished surfaces, even without aggressive sanding.",
    },
    "vintage_restore": {
        "name": "Classic Restoration",
        "keywords": ["restoration", "restore", "vintage", "retro", "classic", "varnish", "refinish", "renew", "реставрация", "винтаж", "ретро", "восстанов", "классик", "лак", "обновить"],
        "prompt_suffix": "Masterful vintage restoration, deep rich wood grain, polished brass fittings, restored original beauty, premium conservation quality, warm soft ambient lighting",
        "materials": ["Old-varnish remover", "Alcohol-based walnut or rosewood stain", "Hard oil with carnauba wax", "Metal polishing compound for brass"],
        "tools": ["Putty knife", "Steel wool or abrasive pad", "Cotton cloths", "Respirator"],
        "pro_tip": "Use grade 0000 steel wool for the final buff—it creates a refined, silky sheen.",
    },
}

# Item categories. Russian keywords are retained for compatibility.
ITEM_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "furniture": {
        "keywords": ["chair", "table", "dresser", "cabinet", "wardrobe", "armchair", "sofa", "shelf", "stool", "furniture", "стул", "стол", "комод", "тумб", "шкаф", "кресло", "диван", "полка", "этажерка", "буфет", "табурет", "мебель"],
        "default_name": "Furniture item",
        "category_title": "Furniture & Interiors",
        "time": "4–8 hours",
        "cost": "700 – 1 800 ₴",
        "eco": "Diverted from landfill, preventing about 22 kg of CO2 emissions—the approximate impact of manufacturing a replacement",
    },
    "clothing": {
        "keywords": [
            "jacket", "jeans", "blazer", "shirt", "hoodie", "sneaker", "shoe", "bag",
            "backpack", "clothing", "coat", "pants", "trousers", "shorts", "skirt", "dress",
            "t-shirt", "sweater", "fabric", "denim",
            "куртк", "джинс", "пиджак", "рубашк", "худи", "свитшот", "кед", "кроссовк",
            "обувь", "сумк", "рюкзак", "одежд", "пальто", "штаны", "штан", "брюки",
            "брюк", "шорты", "шорт", "юбк", "плать", "футболк", "майк", "свитер",
            "кофт", "ткан", "деним"
        ],
        "default_name": "Wardrobe item",
        "category_title": "Clothing, Footwear & Textiles",
        "time": "2–4 hours",
        "cost": "300 – 900 ₴",
        "eco": "Saves up to 7,000 liters of clean water that could be used to produce one new garment",
    },
    "decor": {
        "keywords": ["lamp", "light", "vase", "mirror", "frame", "candleholder", "clock", "pot", "planter", "decor", "dish", "ламп", "светильник", "торшер", "ваз", "зеркал", "рамк", "подсвечник", "часы", "горшок", "кашпо", "декор", "посуд"],
        "default_name": "Decor item",
        "category_title": "Lighting & Home Decor",
        "time": "1.5–3 hours",
        "cost": "250 – 700 ₴",
        "eco": "Extends the item's life and reduces non-recyclable glass or ceramic waste",
    },
    "gadget": {
        "keywords": ["radio", "record player", "turntable", "suitcase", "bicycle", "clock", "box", "crate", "guitar", "instrument", "device", "радио", "проигрыватель", "чемодан", "велосипед", "часы", "коробк", "ящик", "гитар", "инструмент"],
        "default_name": "Vintage Item / Device",
        "category_title": "Vintage Items & Devices",
        "time": "3–6 hours",
        "cost": "500 – 1 500 ₴",
        "eco": "Preserves the item's authentic history and reduces electronic waste",
    },
}


VISION_MODELS = [
    "google/gemma-3-4b-it",
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/vila",
    "microsoft/phi-3-vision-128k-instruct",
]


class UpcycleAgent:
    """Intelligent coordinator for redesigning and upcycling old items."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = (base_url or os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1").rstrip("/")

    def detect_style_and_category(
        self,
        prompt: str,
        style_hint: Optional[str] = None,
        analysis: Optional[ItemAnalysis] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Detect the requested style, when present, and the item category."""
        p_lower = prompt.lower()
        style_key = None

        if style_hint and style_hint in STYLE_PRESETS:
            style_key = style_hint
        else:
            for key, style_data in STYLE_PRESETS.items():
                if any(k in p_lower for k in style_data["keywords"]):
                    style_key = key
                    break

        matched_style = STYLE_PRESETS[style_key] if style_key else None

        matched_cat = None
        if analysis and analysis.category_key in ITEM_CATEGORIES:
            matched_cat = ITEM_CATEGORIES[analysis.category_key]
        else:
            search_text = p_lower
            if analysis:
                search_text = f"{p_lower} {analysis.item_name.lower()} {analysis.description.lower()}"
            for key, cat_data in ITEM_CATEGORIES.items():
                if any(k in search_text for k in cat_data["keywords"]):
                    matched_cat = cat_data
                    break
        if not matched_cat:
            matched_cat = ITEM_CATEGORIES["furniture"]

        return matched_style, matched_cat, style_key

    def analyze_image(self, image_bytes: bytes, mime_type: str = "image/png") -> ItemAnalysis:
        """Inspect the uploaded image and describe the item to be edited."""
        if self.api_key:
            try:
                vision = self._call_vlm_for_item(image_bytes, mime_type)
                if vision:
                    return vision
            except Exception:
                pass
        return ItemAnalysis(
            item_name="Item in the photo",
            category_key="furniture",
            category_title=ITEM_CATEGORIES["furniture"]["category_title"],
            materials="unknown",
            colors="unknown",
            condition="as shown",
            description="The object in the uploaded photo. Preserve its shape and camera angle, and change only what the user requests.",
            source="fallback",
        )

    def _call_vlm_for_item(self, image_bytes: bytes, mime_type: str) -> Optional[ItemAnalysis]:
        from qwen_client import QwenImageEditClient

        prepared, mime_type = QwenImageEditClient.prepare_image_bytes(
            image_bytes, mime_type, max_side=768
        )
        data_uri = QwenImageEditClient._to_data_uri(prepared, mime_type)
        system = (
            "You identify items in photos for upcycling projects. "
            "Return ONLY valid JSON without Markdown: "
            '{"item_name":"specific item name",'
            '"category_key":"furniture|clothing|decor|gadget",'
            '"materials":"materials",'
            '"colors":"colors",'
            '"condition":"condition",'
            '"description":"1-2 sentences describing the object, its shape, and details"}. '
            "Write every value in English."
        )
        user_text = "What item is shown in this photo? Describe only the object to be upcycled."
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        endpoint = f"{self.base_url}/chat/completions"

        for model in VISION_MODELS:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            }
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=25)
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            try:
                content = resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                continue
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            cat_key = str(data.get("category_key", "furniture")).strip().lower()
            if cat_key not in ITEM_CATEGORIES:
                cat_key = "furniture"
            item_name = str(data.get("item_name") or "Item in the photo").strip()
            return ItemAnalysis(
                item_name=item_name,
                category_key=cat_key,
                category_title=ITEM_CATEGORIES[cat_key]["category_title"],
                materials=str(data.get("materials") or "not specified"),
                colors=str(data.get("colors") or "not specified"),
                condition=str(data.get("condition") or "as shown"),
                description=str(data.get("description") or item_name),
                source=model,
            )
        return None

    def enhance_prompt(
        self,
        user_prompt: str,
        style_hint: Optional[str] = None,
        analysis: Optional[ItemAnalysis] = None,
    ) -> str:
        """Build an image-edit instruction from the photo and user request."""
        style_data, cat_data, _ = self.detect_style_and_category(
            user_prompt, style_hint=style_hint, analysis=analysis
        )
        category_key = self._category_key_for_data(cat_data)
        clean_prompt = user_prompt.strip()
        item_bits = []
        if analysis:
            item_bits.append(f"The photo shows: {analysis.item_name}.")
            if analysis.description:
                item_bits.append(analysis.description)
            if analysis.materials and analysis.materials != "unknown":
                item_bits.append(f"Materials: {analysis.materials}.")
            if analysis.colors and analysis.colors != "unknown":
                item_bits.append(f"Current colors: {analysis.colors}.")
        item_context = " ".join(item_bits) if item_bits else "Edit the object shown in this photo."
        preservation = self._preservation_instruction(category_key, clean_prompt)

        enhanced = (
            "Image 1 is the source photo of an old or unwanted physical item. "
            f"{item_context} "
            f"Transform this old item, THAT SAME source item, into the user's requested final product: {clean_prompt}. "
            "Generate a photorealistic image of the finished upcycled result, not a cleaned-up version of the input photo. "
            "Before editing, carefully inspect the source image and carry over the object's identity. "
            f"{preservation} "
            "Preserve the original camera angle, scale, lighting direction, and visible material character unless the user explicitly asks to change them. "
            "Do not replace the object with a generic stock-looking item; do not invent a different fabric, wood grain, hardware, pattern, or construction. "
            "Reuse recognizable source-item details: material texture, color accents, proportions, wear marks, scratches, seams, stitching, handles, legs, buttons, labels, pockets, and distinctive shapes. "
            "Keep the original structure where it helps the transformation, and adapt those details into the new object so it is clear what the original item became. "
            "Do not merely enhance, restore, sharpen, upscale, denoise, relight, recolor, or beautify the original photo. "
            "The object must visibly become the requested new item while still looking made from the exact source object. "
            "Show one clear final result on a simple neutral background with realistic lighting. "
            "No before/after collage, no text labels, no watermark."
        )
        if style_data:
            enhanced += f" Style cues: {style_data['prompt_suffix']}."
        return enhanced

    def generate_diy_plan(
        self,
        user_prompt: str,
        style_hint: Optional[str] = None,
        image_summary: Optional[str] = None,
        analysis: Optional[ItemAnalysis] = None,
    ) -> DIYPlan:
        """Create a detailed upcycling plan and materials list."""
        use_text_agent = os.getenv("ENABLE_TEXT_PLAN_AGENT", "false").lower() in ("1", "true", "yes")
        if use_text_agent and self.api_key and "nvapi-" in self.api_key:
            try:
                plan = self._call_llm_for_plan(user_prompt, style_hint, analysis=analysis)
                if plan:
                    return plan
            except Exception:
                pass

        return self._generate_template_plan(user_prompt, style_hint, analysis=analysis)

    def _generate_template_plan(
        self,
        user_prompt: str,
        style_hint: Optional[str] = None,
        analysis: Optional[ItemAnalysis] = None,
    ) -> DIYPlan:
        """Generate a local DIY plan from the built-in expert system."""
        style_data, cat_data, _ = self.detect_style_and_category(
            user_prompt, style_hint=style_hint, analysis=analysis
        )
        category_key = self._category_key_for_data(cat_data)
        intent = self._detect_intent(user_prompt, category_key)
        if not style_data:
            style_data = self._fallback_style_data(category_key, intent)

        item_title = None
        if analysis and analysis.item_name:
            item_title = analysis.item_name
        else:
            p_words = user_prompt.split()
            for word in p_words:
                w_clean = re.sub(r"[^\w\s]", "", word.lower())
                for c_info in ITEM_CATEGORIES.values():
                    for kw in c_info["keywords"]:
                        if kw in w_clean:
                            item_title = word.capitalize()
                            break
                    if item_title:
                        break
                if item_title:
                    break

        item_title = self._normalize_item_title(item_title or cat_data["default_name"], user_prompt, intent)
        clean_goal = self._clean_user_goal(user_prompt)

        # Генерируем предметные шаги в зависимости от категории и намерения, а не общий ремонтный шаблон.
        steps = self._build_template_steps(
            category_key=category_key,
            intent=intent,
            item_title=item_title,
            style_name=style_data["name"],
            clean_goal=clean_goal,
        )
        resources = self._plan_resources(category_key, intent, style_data, cat_data)

        return DIYPlan(
            item_name=self._plan_title(item_title, style_data["name"], intent),
            category=cat_data["category_title"],
            style=style_data["name"],
            difficulty=resources["difficulty"],
            estimated_time=resources["time"],
            estimated_cost=resources["cost"],
            eco_impact=resources["eco"],
            materials=resources["materials"],
            tools=resources["tools"],
            steps=steps,
            pro_tip=resources["pro_tip"],
        )

    @staticmethod
    def _clean_user_goal(user_prompt: str) -> str:
        goal = re.sub(r"\s+", " ", user_prompt.strip())
        goal = goal.strip(" .,!?:;«»\"'")
        return goal[:180] or "update the item as envisioned"

    @staticmethod
    def _category_key_for_data(cat_data: Dict[str, Any]) -> str:
        for key, value in ITEM_CATEGORIES.items():
            if value is cat_data or value.get("category_title") == cat_data.get("category_title"):
                return key
        return "furniture"

    @staticmethod
    def _preservation_instruction(category_key: str, user_prompt: str) -> str:
        text = user_prompt.lower()
        allow_color_change = any(word in text for word in ["paint", "repaint", "colour", "color", "recolor", "перекрас", "покрас", "цвет"])
        color_rule = (
            "Only change colors requested by the user; preserve all other original colors and accents."
            if allow_color_change
            else "Preserve the original colors, fading, stains, patina, and color accents."
        )
        if category_key == "clothing":
            return (
                "For clothing, preserve the exact fabric type, weave, denim grain, seams, pockets, rivets, buttons, zipper, belt loops, stitching color, wrinkles, fading, distress marks, and labels. "
                f"{color_rule} If turning pants into shorts, keep the same waistband, pockets, fly, belt loops, side seams, and fabric texture; only shorten the legs and finish the hem."
            )
        if category_key == "furniture":
            return (
                "For furniture, preserve the exact silhouette, wood grain, joints, legs, handles, screws, scratches, worn edges, upholstery structure, and existing hardware where possible. "
                f"{color_rule}"
            )
        if category_key == "decor":
            return (
                "For decor items, preserve the object's silhouette, material surface, rim shape, cracks, patina, decorative relief, glass/ceramic/metal texture, and distinctive marks. "
                f"{color_rule}"
            )
        return (
            "Preserve the source object's construction, surface texture, hardware, wear, labels, distinctive proportions, and recognizable details. "
            f"{color_rule}"
        )

    @staticmethod
    def _detect_intent(user_prompt: str, category_key: str) -> str:
        text = user_prompt.lower()
        if category_key == "clothing":
            has_pants = any(word in text for word in ["pants", "trousers", "jeans", "joggers", "leggings", "штан", "брюк", "джинс", "джоггер", "леггинс"])
            wants_shorts = any(word in text for word in ["shorts", "bermuda", "шорт", "бермуд"])
            if has_pants and wants_shorts:
                return "pants_to_shorts"
            if any(word in text for word in ["cut", "shorten", "trim", "make shorter", "отреж", "укорот", "обреж", "подреж", "сделай короче"]):
                return "simple_cut"
            if any(word in text for word in ["patch", "paint", "print", "custom", "tie-dye", "нашив", "роспис", "принт", "кастом", "покрас", "тай-дай"]):
                return "clothing_custom"
            return "clothing_basic"
        if category_key == "furniture":
            if any(word in text for word in ["paint", "repaint", "color", "colour", "graphite", "white", "black", "перекрас", "цвет", "покрас", "графит", "белый", "черный"]):
                return "furniture_paint"
            return "furniture_restore"
        return f"{category_key}_basic"

    @staticmethod
    def _fallback_style_data(category_key: str, intent: str) -> Dict[str, Any]:
        if intent == "pants_to_shorts":
            return {
                "name": "Simple Clothing Upcycle",
                "prompt_suffix": "practical clothing upcycle, clean cut shorts from old pants, realistic fabric edges, wearable result",
                "materials": ["Tailor's chalk or dry soap", "Thread matching the fabric, if hemming", "Pins or sewing clips"],
                "tools": ["Sharp tailor's scissors", "Measuring tape or ruler", "Iron", "Needle or sewing machine (optional)"],
                "pro_tip": "Always add a seam allowance. It is safer to cut slightly long and shorten after fitting than to remove too much at once.",
            }
        if category_key == "clothing":
            return {
                "name": "Practical Clothing Customization",
                "prompt_suffix": "practical wearable clothing upcycle, clean tailoring, realistic fabric details",
                "materials": ["Tailor's chalk or dry soap", "Thread matching the fabric", "Pins or sewing clips"],
                "tools": ["Tailor's scissors", "Measuring tape", "Needle or sewing machine", "Iron"],
                "pro_tip": "Baste and test the fit before the final seam so the length and fit remain easy to adjust.",
            }
        if category_key == "decor":
            return STYLE_PRESETS["matte_minimal"]
        if category_key == "gadget":
            return STYLE_PRESETS["vintage_restore"]
        return STYLE_PRESETS["vintage_restore"]

    @staticmethod
    def _normalize_item_title(item_title: str, user_prompt: str, intent: str) -> str:
        text = user_prompt.lower()
        if intent == "pants_to_shorts":
            if "jean" in text or "джинс" in text:
                return "jeans"
            if "trouser" in text or "брюк" in text:
                return "trousers"
            return "pants"
        return item_title

    @staticmethod
    def _plan_title(item_title: str, style_name: str, intent: str) -> str:
        if intent == "pants_to_shorts":
            return f"Shorts Made from Old {item_title.title()}"
        if style_name.startswith("Practical") or style_name.startswith("Simple"):
            return f"Upcycle: {item_title}"
        return f"{item_title} in {style_name} Style"

    @staticmethod
    def _plan_resources(
        category_key: str,
        intent: str,
        style_data: Dict[str, Any],
        cat_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if intent == "pants_to_shorts":
            return {
                "difficulty": "Easy",
                "time": "20–45 minutes",
                "cost": "0–150 ₴",
                "eco": "Keeps the garment in use instead of buying new shorts; waste is limited to the removed fabric.",
                "materials": style_data["materials"],
                "tools": style_data["tools"],
                "pro_tip": style_data["pro_tip"],
            }
        if category_key == "clothing":
            return {
                "difficulty": "Easy",
                "time": "40–90 minutes",
                "cost": "0–250 ₴",
                "eco": cat_data["eco"],
                "materials": style_data["materials"],
                "tools": style_data["tools"],
                "pro_tip": style_data["pro_tip"],
            }
        difficulty = "Easy" if category_key == "decor" else "Intermediate"
        if "kintsugi" in style_data["name"].lower() or category_key == "gadget":
            difficulty = "Advanced"
        return {
            "difficulty": difficulty,
            "time": cat_data["time"],
            "cost": cat_data["cost"],
            "eco": cat_data["eco"],
            "materials": style_data["materials"],
            "tools": style_data["tools"],
            "pro_tip": style_data["pro_tip"],
        }

    @staticmethod
    def _build_template_steps(
        category_key: str,
        intent: str,
        item_title: str,
        style_name: str,
        clean_goal: str,
    ) -> List[DIYStep]:
        item = item_title.lower()
        quoted_goal = f'"{clean_goal}"'

        if intent == "pants_to_shorts":
            return [
                DIYStep(
                    1,
                    "Choose the Length During a Fitting",
                    f"Put on the {item} and mark the desired shorts length on one leg with chalk. Take them off and add a seam allowance below the mark: 2–3 cm for a folded hem or 0.5–1 cm for a raw edge.",
                    "Length mark",
                    "Draw a chalk line on one leg for the desired length, plus the allowance below it.",
                    "The cutting line must sit below the final length so the shorts do not end up too short.",
                    "measure_line",
                ),
                DIYStep(
                    2,
                    "Transfer the Line to Both Legs",
                    "Lay the pants flat and align the waistband, inseams, and side seams. Draw the cutting line across the first leg, then transfer it to the second with a ruler or the removed fabric as a guide.",
                    "Symmetrical marks",
                    "Make two matching horizontal cutting lines and compare their distance from the crotch seam on both legs.",
                    "Both lines must be at the same height or the shorts will have uneven legs.",
                    "mark_line",
                ),
                DIYStep(
                    3,
                    "Cut Off the Excess Length",
                    "Cut slowly along the marked line with sharp scissors. Do not stretch the fabric. If it slips, pin the layers and cut each leg separately.",
                    "Excess fabric removed",
                    "Cut exactly along the lower allowance line, keeping pockets and side seams away from the scissors.",
                    "Fold the shorts in half after cutting and compare both leg lengths.",
                    "cut_line",
                ),
                DIYStep(
                    4,
                    "Finish the Lower Edge",
                    "For a clean hem, fold the edge up 1 cm and press, then fold another 1 cm and stitch. For a raw denim edge, pull out a few threads and secure the side seams with short stitches to stop fraying above the intended level.",
                    "Edge finished",
                    "Keep the hem line parallel to the edge and the fold width consistent around each leg.",
                    "The edge should not twist, and the side seams should lie flat.",
                    "hem_line",
                ),
                DIYStep(
                    5,
                    "Final Fitting and Adjustment",
                    "Try on the shorts, sit down, and check comfort. If one side is longer, trim it a few millimeters at a time, press the hem again, and remove remaining chalk with a damp cloth.",
                    "Finished shorts",
                    "The final hemline is even at the front and back, and both legs look identical.",
                    "The shorts feel comfortable in motion and do not pull at the crotch seam.",
                    "finish_line",
                ),
            ]

        if category_key == "clothing":
            return [
                DIYStep(
                    1,
                    "Fit and Mark the Changes",
                    f"Put on or lay out the {item}. Mark the areas to change for {quoted_goal}: where to shorten, where to add decoration, and where to retain the original shape.",
                    "Plan on the fabric",
                    "Mark only with chalk or dry soap, not a permanent marker, so every line can be removed easily.",
                    "All lines are visible, and no important garment detail falls inside an accidental cutting area.",
                    "measure_line",
                ),
                DIYStep(
                    2,
                    "Make the Initial Alteration",
                    "Perform only irreversible actions already confirmed by the markings: shorten, unpick excess material, or prepare an area for decoration. Work in small increments.",
                    "Initial shape",
                    "Keep a 1–3 cm allowance wherever an edge will later be folded or stitched.",
                    "The shape now resembles the goal but still has room for adjustment.",
                    "cut_line",
                ),
                DIYStep(
                    3,
                    "Secure the Edge or Decoration",
                    "Fold, stitch, or attach the decorative elements. If using fabric paint, place cardboard between the layers and let the paint dry completely.",
                    "Edge secured",
                    "The seam or decoration follows the marked line without twisting or pulling the fabric.",
                    "Lift and gently shake the garment: nothing should shift or come loose.",
                    "hem_line",
                ),
                DIYStep(
                    4,
                    "Final Fitting",
                    "Try on the item and move around to check fit and symmetry. Remove chalk lines, press the garment, and reinforce weak areas with extra stitches.",
                    "Finished garment",
                    "Final edges and decoration are symmetrical relative to seams, pockets, or the garment center.",
                    "The item is comfortable and looks intentionally finished rather than temporarily cut.",
                    "finish_line",
                ),
            ]

        if category_key == "decor":
            return [
                DIYStep(1, "Clean and Inspect the Base", f"Clean the {item}; remove dust, grease, and loose old finishes. Inspect cracks, chips, fasteners, and electrical parts, if present.", "Original decor item"),
                DIYStep(2, "Prepare the Surface", "Scuff the surface with fine sandpaper or degrease it with a suitable cleaner. Mask any areas that should remain uncoated.", "Prepared surface"),
                DIYStep(3, "Repair Defects", "Fill chips, cracks, and uneven areas with a compatible compound. Once dry, level the surface and remove all dust.", "Shape restored"),
                DIYStep(4, f"Decorative Finish: {style_name}", f"Apply the main colors and accents for {quoted_goal}. Use thin coats to preserve the item's shape and details.", "New style visible"),
                DIYStep(5, "Protect and Reassemble", "Apply a varnish, wax, or sealant suitable for the material. Reinstall hardware, the bulb, or decorative elements only after everything is fully dry.", "Finished item"),
            ]

        if category_key == "gadget":
            return [
                DIYStep(1, "Disassemble Safely and Take Reference Photos", f"Photograph the {item} from every side, remove detachable parts, and label the fasteners. Do not power electrical components until they have been inspected.", "Before disassembly"),
                DIYStep(2, "Clean the Case and Parts", "Clean the case with a soft brush and degrease the surfaces. Handle fragile decorative components separately by hand.", "Clean case"),
                DIYStep(3, "Repair and Prepare for Finishing", "Reinforce weak fasteners, sand chipped areas, and restore the case geometry. Mask labels, mechanisms, and other protected areas.", "Case ready"),
                DIYStep(4, f"New Look: {style_name}", f"Apply the finish and decorative details for {quoted_goal} without covering controls or ventilation openings.", "New design"),
                DIYStep(5, "Reassemble and Test", "Reassemble the item in reverse order and check stability and mechanical parts. Test electrical components only after verifying the insulation.", "Finished object"),
            ]

        return [
            DIYStep(1, "Inspect and Disassemble", f"Inspect the {item}, photograph all fasteners, and remove handles, legs, the seat, or other detachable parts. Mark cracks, chips, and weak areas.", "Original item"),
            DIYStep(2, "Clean and Remove the Old Finish", "Wash and degrease the surface. Remove peeling varnish or paint, then lightly sand areas that will receive the new finish.", "Clean base"),
            DIYStep(3, "Repair and Prime", "Fill deep defects, tighten fasteners, and apply a primer compatible with the material. Once dry, smooth the surface with fine sanding.", "Base restored"),
            DIYStep(4, f"Finish and Details: {style_name}", f"Apply two thin base coats and add accents for {quoted_goal}, such as hardware, contrasting elements, fabric, or patina.", "New design emerging"),
            DIYStep(5, "Protect and Reassemble", "Seal the surface with a durable varnish, oil, or wax. Install the hardware, reassemble the item, and allow the finish to cure.", "Finished result"),
        ]

    def _call_llm_for_plan(
        self,
        user_prompt: str,
        style_hint: Optional[str] = None,
        analysis: Optional[ItemAnalysis] = None,
    ) -> Optional[DIYPlan]:
        """Call an NVIDIA text model to generate a custom upcycling plan."""
        system_prompt = (
            "You are a professional expert in upcycling, restoration, and DIY redesign. "
            "The user uploads a photo of an old item and describes the desired change. "
            "Create a practical, realistic, and inspiring do-it-yourself plan. "
            "Write every value in English. Return only valid JSON without Markdown, using these fields:\n"
            "{\n"
            '  "item_name": "Item and style name",\n'
            '  "category": "Category",\n'
            '  "style": "Style name",\n'
            '  "difficulty": "Easy | Intermediate | Advanced",\n'
            '  "estimated_time": "Working time",\n'
            '  "estimated_cost": "Estimated budget",\n'
            '  "eco_impact": "Environmental benefit of keeping the item",\n'
            '  "materials": ["materials", "list"],\n'
            '  "tools": ["tools", "list"],\n'
            '  "steps": [{"number": 1, "title": "...", "desc": "...", "markup": "what and where to mark", "checkpoint": "how to verify the step", "visual_hint": "short caption", "visual_type": "measure_line|mark_line|cut_line|hem_line|finish_line|progress"}],\n'
            '  "pro_tip": "Most important professional tip"\n'
            "}"
        )

        user_content = f"The user wants to change the item as follows: '{user_prompt}'."
        if analysis:
            user_content += (
                f" Detected in the photo: {analysis.item_name}. "
                f"Description: {analysis.description}. "
                f"Materials: {analysis.materials}. Condition: {analysis.condition}."
            )
        if style_hint:
            user_content += f" Desired style or direction: {style_hint}."

        payload = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.5,
            "max_tokens": 1000,
        }

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=12)
        if resp.status_code != 200:
            return None

        content = resp.json()["choices"][0]["message"]["content"]
        # Извлекаем JSON из ответа
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None

        data = json.loads(match.group(0))
        steps = [
            DIYStep(
                number=s.get("number", i + 1),
                title=s.get("title", ""),
                desc=s.get("desc", ""),
                visual_hint=s.get("visual_hint", ""),
                markup=s.get("markup", ""),
                checkpoint=s.get("checkpoint", ""),
                visual_type=s.get("visual_type", "progress"),
            )
            for i, s in enumerate(data.get("steps", []))
        ]

        return DIYPlan(
            item_name=data.get("item_name", "Upcycled Item"),
            category=data.get("category", "Upcycling"),
            style=data.get("style", "Modern"),
            difficulty=data.get("difficulty", "Intermediate"),
            estimated_time=data.get("estimated_time", "3–5 hours"),
            estimated_cost=data.get("estimated_cost", "~800 ₴"),
            eco_impact=data.get("eco_impact", "The item was diverted from disposal"),
            materials=data.get("materials", []),
            tools=data.get("tools", []),
            steps=steps,
            pro_tip=data.get("pro_tip", "Work in a well-ventilated area."),
        )
