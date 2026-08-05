from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from PIL import Image
from transformers import pipeline


CATEGORY_LABELS = {
    "t-shirt": "t-shirt",
    "polo shirt": "polo shirt",
    "button-up shirt": "button-up shirt",
    "hoodie": "hoodie",
    "sweatshirt": "sweatshirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "windbreaker": "windbreaker",
    "coat": "coat",
    "jeans": "jeans",
    "pants": "pants",
    "sweatpants": "sweatpants",
    "shorts": "shorts",
    "cargo shorts": "cargo shorts",
    "skirt": "skirt",
    "dress": "dress",
    "baseball cap": "baseball cap",
    "shoes": "shoes",
}

COLOUR_LABELS = {
    "black": "black clothing item",
    "white": "white clothing item",
    "grey": "grey or gray clothing item",
    "blue": "blue clothing item",
    "navy": "navy blue clothing item",
    "red": "red clothing item",
    "green": "green clothing item",
    "olive": "olive green clothing item",
    "brown": "brown clothing item",
    "beige": "beige clothing item",
    "cream": "cream coloured clothing item",
    "yellow": "yellow clothing item",
    "orange": "orange clothing item",
    "pink": "pink clothing item",
    "purple": "purple clothing item",
    "multicolour": "multicoloured clothing item",
}

LAYER_TYPE_LABELS = {
    "base layer": "clothing worn as a base layer",
    "mid layer": "clothing worn as a middle layer",
    "outerwear": "outerwear worn over other clothing",
    "single layer": "clothing normally worn by itself",
}

STYLE_LABELS = {
    "casual": "casual fashion clothing",
    "streetwear": "streetwear fashion clothing",
    "smart casual": "smart casual fashion clothing",
    "formal": "formal fashion clothing",
    "athletic": "athletic or sportswear clothing",
    "workwear": "workwear fashion clothing",
    "minimalist": "minimalist fashion clothing",
    "vintage": "vintage fashion clothing",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class Prediction(TypedDict):
    label: str
    confidence: float
    confidence_percent: float


class ClothingAttributes(TypedDict):
    category: Prediction
    colour: Prediction
    layer_type: Prediction
    style: Prediction


@lru_cache(maxsize=1)
def get_classifier():
    """Load the CLIP model once and reuse it."""
    print("Loading clothing classifier...")

    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
    )


def validate_image_path(image_path: str | Path) -> Path:
    """Validate an image path before classification."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image was not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    return path


def load_image(image_path: str | Path) -> Image.Image:
    """Open an image and convert it to RGB."""

    path = validate_image_path(image_path)

    with Image.open(path) as opened_image:
        return opened_image.convert("RGB")


def classify_candidates(
    image: Image.Image,
    candidates: dict[str, str],
    top_k: int = 3,
) -> list[Prediction]:
    """Classify an image using a set of canonical labels and CLIP prompts."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    classifier = get_classifier()

    prompt_to_label = {
        prompt: label
        for label, prompt in candidates.items()
    }

    raw_predictions = classifier(
        image,
        candidate_labels=list(candidates.values()),
    )

    predictions: list[Prediction] = []

    for result in raw_predictions[:top_k]:
        prompt = str(result["label"])
        label = prompt_to_label.get(prompt, prompt)
        score = float(result["score"])

        predictions.append(
            {
                "label": label,
                "confidence": round(score, 4),
                "confidence_percent": round(score * 100, 1),
            }
        )

    return predictions


def classify_image(
    image_path: str | Path,
    top_k: int = 3,
) -> list[Prediction]:
    """Return the top clothing-category predictions for one image."""

    image = load_image(image_path)

    return classify_candidates(
        image=image,
        candidates=CATEGORY_LABELS,
        top_k=top_k,
    )


def classify_colour(
    image_path: str | Path,
    top_k: int = 3,
) -> list[Prediction]:
    """Return the top colour predictions for one clothing image."""

    image = load_image(image_path)

    return classify_candidates(
        image=image,
        candidates=COLOUR_LABELS,
        top_k=top_k,
    )


def classify_layer_type(
    image_path: str | Path,
    top_k: int = 3,
) -> list[Prediction]:
    """Return the top layering predictions for one clothing image."""

    image = load_image(image_path)

    return classify_candidates(
        image=image,
        candidates=LAYER_TYPE_LABELS,
        top_k=top_k,
    )


def classify_style(
    image_path: str | Path,
    top_k: int = 3,
) -> list[Prediction]:
    """Return the top style predictions for one clothing image."""

    image = load_image(image_path)

    return classify_candidates(
        image=image,
        candidates=STYLE_LABELS,
        top_k=top_k,
    )


def classify_clothing_attributes(
    image_path: str | Path,
) -> ClothingAttributes:
    """Predict category, colour, layer type, and style for one image."""

    image = load_image(image_path)

    category = classify_candidates(
        image=image,
        candidates=CATEGORY_LABELS,
        top_k=1,
    )[0]

    colour = classify_candidates(
        image=image,
        candidates=COLOUR_LABELS,
        top_k=1,
    )[0]

    layer_type = classify_candidates(
        image=image,
        candidates=LAYER_TYPE_LABELS,
        top_k=1,
    )[0]

    style = classify_candidates(
        image=image,
        candidates=STYLE_LABELS,
        top_k=1,
    )[0]

    return {
        "category": category,
        "colour": colour,
        "layer_type": layer_type,
        "style": style,
    }