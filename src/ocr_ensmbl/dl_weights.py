import os
from pathlib import Path

import easyocr
from huggingface_hub import hf_hub_download
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

CACHE_DIR = Path(os.environ.get("UFOIL_CACHE", "~/.cache/ufoil")).expanduser()

# original clovaai github release link is dead, using this HF mirror
CRAFT_REPO_ID = "Manbehindthemadness/craft_mlt_25k"
CRAFT_FILENAME = "craft_mlt_25k.pth"

TROCR_CHECKPOINT = "microsoft/trocr-large-handwritten"


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_craft_weights(force: bool = False) -> Path:
    ensure_cache_dir()
    path = hf_hub_download(
        repo_id=CRAFT_REPO_ID,
        filename=CRAFT_FILENAME,
        cache_dir=str(CACHE_DIR),
        force_download=force,
    )
    return Path(path)


def download_easyocr_weights(langs: list[str] | None = None) -> None:
    # easyocr pulls its own weights into ~/.EasyOCR on first use, this
    # just forces that up front instead of doing it lazily mid-pipeline
    langs = langs or ["en"]
    easyocr.Reader(langs, download_enabled=True, verbose=False)


def download_trocr_weights(checkpoint: str = TROCR_CHECKPOINT) -> None:
    # use_fast=False sidesteps a transformers bug where the fast
    # tokenizer path fails even with sentencepiece installed
    TrOCRProcessor.from_pretrained(checkpoint, use_fast=False)
    VisionEncoderDecoderModel.from_pretrained(checkpoint)


def download_all(force: bool = False) -> None:
    print("tesseract binary needs to be installed separately (apt/brew)")
    download_craft_weights(force=force)
    download_easyocr_weights()
    download_trocr_weights()
    print("all weights cached")


if __name__ == "__main__":
    download_all()
