"""Pre-download Media-Service ML models for Docker image build.

Usage:
    python scripts/download_models.py [--output /path/to/models]
"""

import argparse
import os


def download_whisper(output_dir: str, model_size: str = "base") -> None:
    from faster_whisper import WhisperModel

    print(f"Downloading Whisper model: {model_size}")
    WhisperModel(model_size, device="cpu", compute_type="int8", download_root=output_dir)
    print(f"Whisper {model_size} downloaded to {output_dir}")


def download_clip(output_dir: str, model_name: str = "clip-ViT-B-32") -> None:
    """Pre-cache CLIP image embedder for the /v1/embed/image endpoint."""
    from sentence_transformers import SentenceTransformer

    print(f"Downloading CLIP model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder=output_dir)
    get_dim = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(get_dim):
        print(f"CLIP model downloaded. Dimensions: {get_dim()}")
    else:
        print("CLIP model downloaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Media-Service ML models")
    parser.add_argument("--output", default="./models", help="Output directory for models")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--clip-model", default="clip-ViT-B-32", help="CLIP model")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    download_whisper(args.output, args.whisper_model)
    download_clip(args.output, args.clip_model)

    print(f"\nAll Media-Service models downloaded to {args.output}")


if __name__ == "__main__":
    main()
