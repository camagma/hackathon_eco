"""
Command-line image editing with Qwen-Image-Edit.

Examples:
1. Edit an image:
   python cli.py --image photo.jpg --prompt "Give the background a cyberpunk look" --output result.png

2. Specify an API key and seed:
   python cli.py --image avatar.png --prompt "Add stylish sunglasses" --api-key nvapi-... --seed 42

3. Check the API connection:
   python cli.py --health
"""

import argparse
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from qwen_client import (
    DEFAULT_CLOUD_BASE_URL,
    DEFAULT_MODEL,
    KNOWN_MODELS,
    QwenAPIError,
    QwenAuthError,
    QwenError,
    QwenImageEditClient,
    QwenRateLimitError,
)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit CLI (NVIDIA Build API & NIM)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to the source image (PNG, JPG, WebP)",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Text description of the edit, for example: 'Change the background to Paris at night'",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_edited.png",
        help="Output path (default: output_edited.png)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}).\nAvailable options: {', '.join(KNOWN_MODELS)}",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt describing what to avoid",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Prompt guidance scale",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Image size, for example '1024x1024'",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="NVIDIA API key (nvapi-...). Defaults to the NVIDIA_API_KEY environment variable",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=f"Base URL (default: {DEFAULT_CLOUD_BASE_URL}; local NIM: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check the API connection and configuration",
    )

    args = parser.parse_args()

    client = QwenImageEditClient(
        api_key=args.api_key,
        base_url=args.base_url,
        default_model=args.model,
    )

    # Режим проверки здоровья API
    if args.health:
        print("\n🔍 Checking the API connection...")
        print(f"   Base URL:      {client.base_url}")
        print(f"   Model:         {client.default_model}")
        print(f"   API Key:       {'[Set]' if client.api_key else '[NOT SET! Get one at https://build.nvidia.com]'}")
        
        status = client.check_health()
        print(f"   Connected:     {'✅ Yes' if status.get('connected') else '❌ No'}")
        if status.get("error"):
            print(f"   Details:       {status['error']}")
        if status.get("available_models_count"):
            print(f"   API models:    {status['available_models_count']}")
        print()
        sys.exit(0 if status.get("connected") else 1)

    # Режим редактирования
    if not args.image or not args.prompt:
        print("❌ Error: both --image (-i) and --prompt (-p) are required.")
        print("   Example: python cli.py -i input.jpg -p 'Change the background to outer space' -o result.png")
        print("   Run python cli.py --help for usage information.")
        sys.exit(1)

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ Error: input file not found: {img_path.resolve()}")
        sys.exit(1)

    print("\n🚀 Starting Qwen-Image-Edit...")
    print(f"   📁 Source file: {img_path.name}")
    print(f"   💬 Prompt:      \"{args.prompt}\"")
    print(f"   🤖 Model:       {args.model}")
    print(f"   🌐 Endpoint:    {client.base_url}")
    if args.seed is not None:
        print(f"   🎲 Seed:          {args.seed}")

    try:
        start = time.time()
        result = client.edit_image(
            image=img_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            model=args.model,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance,
            size=args.size,
        )
        saved_path = result.save(args.output)
        elapsed = time.time() - start

        print("\n✨ Image edited successfully!")
        print(f"   💾 Saved to:        {saved_path.resolve()}")
        print(f"   ⏱️ Generation time: {result.latency_seconds:.2f} sec (total: {elapsed:.2f} sec)")
        print(f"   📦 File size:       {len(result.image_bytes) / 1024:.1f} KB\n")

    except QwenAuthError as e:
        print(f"\n🔑 Authentication error: {e}")
        print("   💡 Pass a key with --api-key nvapi-... or set NVIDIA_API_KEY in .env.")
        sys.exit(1)
    except QwenRateLimitError as e:
        print(f"\n⏳ Rate limit exceeded: {e}")
        sys.exit(1)
    except QwenAPIError as e:
        print(f"\n⚠️ API error ({e.status_code}): {e}")
        sys.exit(1)
    except QwenError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by the user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
