import random
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

try:
    from backend.ml_classifier import (
        classify_colour,
        classify_image,
        classify_style,
    )
except ModuleNotFoundError:
    from ml_classifier import (
        classify_colour,
        classify_image,
        classify_style,
    )


APP_FOLDER = Path(__file__).resolve().parent
BACKEND_FOLDER = APP_FOLDER.parent
DATA_FOLDER = BACKEND_FOLDER / "data"
UPLOAD_FOLDER = BACKEND_FOLDER / "uploads"
WARDROBE_FILE = DATA_FOLDER / "wardrobe.csv"
FEEDBACK_FILE = DATA_FOLDER / "classification_feedback.csv"

DATA_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id",
    "name",
    "category",
    "colour",
    "season",
    "occasion",
    "warmth_level",
    "waterproof",
    "layer_type",
    "style",
    "image_path",
    "date_added",
]

FEEDBACK_COLUMNS = [
    "feedback_id",
    "item_id",
    "image_filename",
    "predicted_category",
    "predicted_category_detail",
    "final_category",
    "category_confidence",
    "category_corrected",
    "predicted_colour",
    "final_colour",
    "colour_confidence",
    "colour_corrected",
    "predicted_layer_type",
    "final_layer_type",
    "layer_type_confidence",
    "layer_type_corrected",
    "predicted_style",
    "final_style",
    "style_confidence",
    "style_corrected",
    "date_recorded",
]

TOP_CATEGORIES = {
    "t-shirt",
    "shirt",
    "button-up shirt",
    "polo shirt",
    "sweatshirt",
    "hoodie",
    "sweater",
    "tank top",
    "blouse",
}

BOTTOM_CATEGORIES = {
    "pants",
    "sweatpants",
    "jeans",
    "shorts",
    "cargo shorts",
    "skirt",
}

OUTERWEAR_CATEGORIES = {
    "jacket",
    "windbreaker",
    "coat",
    "blazer",
}

app = FastAPI(
    title="Smart Closet API",
    version="1.0.0",
)


class WardrobeItemUpdate(BaseModel):
    """Editable wardrobe fields sent by the React frontend."""

    name: str
    category: str
    colour: str
    season: str
    occasion: str
    warmth_level: int
    waterproof: str
    layer_type: str
    style: str = "Casual"


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads",
)


def load_wardrobe() -> pd.DataFrame:
    """Load and normalize the wardrobe CSV."""

    if not WARDROBE_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        wardrobe = pd.read_csv(WARDROBE_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read wardrobe.csv: {error}",
        ) from error

    if "colour" not in wardrobe.columns:
        wardrobe["colour"] = ""

    if "color" in wardrobe.columns:
        current_colour = (
            wardrobe["colour"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        old_color = (
            wardrobe["color"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        wardrobe["colour"] = current_colour.where(
            current_colour != "",
            old_color,
        )

    for column in COLUMNS:
        if column not in wardrobe.columns:
            wardrobe[column] = ""

    wardrobe["warmth_level"] = pd.to_numeric(
        wardrobe["warmth_level"],
        errors="coerce",
    ).fillna(3)

    return wardrobe[COLUMNS]


def save_wardrobe(wardrobe: pd.DataFrame) -> None:
    """Save the normalized wardrobe CSV."""

    wardrobe[COLUMNS].to_csv(WARDROBE_FILE, index=False)


def load_classification_feedback() -> pd.DataFrame:
    """Load and normalize the AI classification feedback CSV."""

    if not FEEDBACK_FILE.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    try:
        feedback = pd.read_csv(FEEDBACK_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read classification_feedback.csv: {error}",
        ) from error

    for column in FEEDBACK_COLUMNS:
        if column not in feedback.columns:
            feedback[column] = ""

    return feedback[FEEDBACK_COLUMNS]


def save_classification_feedback(feedback: pd.DataFrame) -> None:
    """Save the normalized AI classification feedback CSV."""

    feedback[FEEDBACK_COLUMNS].to_csv(FEEDBACK_FILE, index=False)


def parse_confidence(value: str) -> float:
    """Convert a submitted confidence value into a safe float."""

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def normalized_label(value: str) -> str:
    """Normalize labels before comparing predictions and final values."""

    return str(value).strip().lower()


def label_was_corrected(predicted: str, final: str) -> bool:
    """Return whether the user changed an AI-generated label."""

    predicted_label = normalized_label(predicted)

    if not predicted_label:
        return False

    return predicted_label != normalized_label(final)


def clean_value(value):
    """Convert pandas values into JSON-safe Python values."""

    if pd.isna(value):
        return ""

    if hasattr(value, "item"):
        return value.item()

    return value


def wardrobe_row_to_dict(row: pd.Series) -> dict:
    """Convert one CSV row into an API response object."""

    item = {
        column: clean_value(row.get(column, ""))
        for column in COLUMNS
    }

    try:
        item["warmth_level"] = int(float(item["warmth_level"]))
    except (TypeError, ValueError):
        item["warmth_level"] = 3

    saved_path = str(item.get("image_path", "")).strip()
    image_filename = Path(saved_path).name if saved_path else ""

    if image_filename and (UPLOAD_FOLDER / image_filename).is_file():
        item["image_url"] = f"/uploads/{quote(image_filename)}"
    else:
        item["image_url"] = None

    return item


def identify_outfit_slot(row: pd.Series) -> str | None:
    """Determine whether an item is a top, bottom, or outerwear."""

    category = str(row.get("category", "")).strip().lower()
    layer_type = str(row.get("layer_type", "")).strip().lower()

    if category in {"top", "tops", "upper body", "hoodie", "sweater"}:
        return "top"

    if category in {"bottom", "bottoms", "lower body", "skirt"}:
        return "bottom"

    if category in {
        "jacket",
        "outerwear",
        "outer wear",
        "outer layer",
    }:
        return "outerwear"

    if category in TOP_CATEGORIES:
        return "top"

    if category in BOTTOM_CATEGORIES:
        return "bottom"

    if category in OUTERWEAR_CATEGORIES:
        return "outerwear"

    if "top" in layer_type or "upper" in layer_type:
        return "top"

    if "bottom" in layer_type or "lower" in layer_type:
        return "bottom"

    if "outer" in layer_type:
        return "outerwear"

    return None


def infer_layer_type(category: str) -> str:
    """Infer a reliable layer type from the predicted category."""

    normalized_category = category.strip().lower()

    if normalized_category in OUTERWEAR_CATEGORIES:
        return "outerwear"

    if normalized_category in {
        "hoodie",
        "sweatshirt",
        "sweater",
        "cardigan",
    }:
        return "mid layer"

    if normalized_category in {
        "t-shirt",
        "shirt",
        "button-up shirt",
        "polo shirt",
        "tank top",
        "blouse",
    }:
        return "base layer"

    return "single layer"


def suggest_full_item_details(
    category_label: str,
    colour_label: str,
    style_label: str,
) -> dict:
    """
    Generate editable wardrobe suggestions from CLIP predictions.

    Category, colour, and style come from CLIP. Properties that
    cannot be reliably identified from a single image are inferred
    conservatively from the predicted garment type.
    """

    category = str(category_label).strip().lower()
    colour = str(colour_label).strip()
    style = str(style_label).strip().lower()

    display_names = {
        "t-shirt": "T-shirt",
        "polo shirt": "Polo Shirt",
        "button-up shirt": "Button-up Shirt",
        "tank top": "Tank Top",
        "baseball cap": "Baseball Cap",
        "cargo shorts": "Cargo Shorts",
    }

    garment_name = display_names.get(
        category,
        category.title() or "Clothing Item",
    )

    suggested_name = " ".join(
        part
        for part in (
            colour.title(),
            garment_name,
        )
        if part
    )

    season = "All seasons"
    occasion = "Casual"
    warmth_level = 3
    waterproof = "No"

    summer_light_items = {
        "t-shirt",
        "tank top",
        "shorts",
        "cargo shorts",
    }

    light_items = {
        "shirt",
        "button-up shirt",
        "polo shirt",
        "blouse",
        "skirt",
        "dress",
    }

    medium_items = {
        "jeans",
        "pants",
        "sweatpants",
        "shoes",
        "baseball cap",
        "blazer",
        "jacket",
        "windbreaker",
    }

    warm_items = {
        "hoodie",
        "sweatshirt",
        "sweater",
        "cardigan",
    }

    if category in summer_light_items:
        season = "Summer"
        warmth_level = 1

    elif category in light_items:
        season = "All seasons"
        warmth_level = 2

    elif category in medium_items:
        season = (
            "Fall"
            if category in {
                "jacket",
                "windbreaker",
            }
            else "All seasons"
        )
        warmth_level = 3

    elif category in warm_items:
        season = "Fall"
        warmth_level = 4

    elif category == "coat":
        season = "Winter"
        warmth_level = 5

    if category == "windbreaker":
        waterproof = "Light"
    elif category in {
        "jacket",
        "coat",
    }:
        waterproof = "Unknown"

    athletic_styles = {
        "athletic",
        "sporty",
        "sport",
    }

    work_styles = {
        "workwear",
        "smart casual",
    }

    if style in athletic_styles:
        occasion = "Exercise"
    elif style == "formal":
        occasion = "Formal"
    elif style in work_styles:
        occasion = "Work"

    return {
        "name": suggested_name,
        "season": season,
        "occasion": occasion,
        "warmth_level": warmth_level,
        "waterproof": waterproof,
    }


@app.get("/api/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "wardrobe_file": str(WARDROBE_FILE),
        "wardrobe_file_exists": WARDROBE_FILE.exists(),
        "feedback_file": str(FEEDBACK_FILE),
        "feedback_file_exists": FEEDBACK_FILE.exists(),
    }


@app.post("/api/classify-image")
async def classify_uploaded_image(
    image: UploadFile = File(...),
) -> dict:
    original_filename = image.filename or ""
    extension = Path(original_filename).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, PNG, and WEBP images are supported.",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    temporary_path = (
        UPLOAD_FOLDER
        / f".classification-{uuid4()}{extension}"
    )

    try:
        temporary_path.write_bytes(image_bytes)

        category_predictions = await run_in_threadpool(
            classify_image,
            temporary_path,
            3,
        )

        colour_predictions = await run_in_threadpool(
            classify_colour,
            temporary_path,
            1,
        )

        style_predictions = await run_in_threadpool(
            classify_style,
            temporary_path,
            1,
        )

        category_prediction = category_predictions[0]
        colour_prediction = colour_predictions[0]
        style_prediction = style_predictions[0]

        layer_type = infer_layer_type(
            category_prediction["label"]
        )

        full_details = suggest_full_item_details(
            category_prediction["label"],
            colour_prediction["label"],
            style_prediction["label"],
        )

        return {
            "filename": original_filename,
            "predictions": category_predictions,
            "attributes": {
                "category": category_prediction,
                "colour": colour_prediction,
                "layer_type": {
                    "label": layer_type,
                    "confidence": 1.0,
                    "confidence_percent": 100.0,
                },
                "style": style_prediction,
                "name": full_details["name"],
                "season": full_details["season"],
                "occasion": full_details["occasion"],
                "warmth_level": full_details["warmth_level"],
                "waterproof": full_details["waterproof"],
            },
        }

    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not process the uploaded image: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {error}",
        ) from error

    finally:
        temporary_path.unlink(missing_ok=True)


@app.get("/api/wardrobe")
def get_wardrobe() -> list[dict]:
    wardrobe = load_wardrobe()

    return [
        wardrobe_row_to_dict(row)
        for _, row in wardrobe.iloc[::-1].iterrows()
    ]


@app.delete("/api/wardrobe/{item_id}")
def delete_wardrobe_item(item_id: str) -> dict:
    """Delete one wardrobe item and its saved image."""

    wardrobe = load_wardrobe()

    matching_rows = (
        wardrobe["id"]
        .fillna("")
        .astype(str)
        .str.strip()
        == item_id.strip()
    )

    if not matching_rows.any():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clothing item not found.",
        )

    item_row = wardrobe.loc[matching_rows].iloc[0]
    deleted_item = wardrobe_row_to_dict(item_row)

    updated_wardrobe = wardrobe.loc[~matching_rows].copy()

    try:
        save_wardrobe(updated_wardrobe)
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not update wardrobe.csv: {error}",
        ) from error

    saved_path = str(item_row.get("image_path", "")).strip()
    image_filename = Path(saved_path).name if saved_path else ""

    if image_filename:
        try:
            (UPLOAD_FOLDER / image_filename).unlink(missing_ok=True)
        except OSError:
            pass

    return {
        "message": (
            f"{deleted_item['name']} was deleted "
            "from your wardrobe."
        ),
        "item": deleted_item,
    }




@app.put("/api/wardrobe/{item_id}")
def update_wardrobe_item(
    item_id: str,
    update: WardrobeItemUpdate,
) -> dict:
    """Update the editable details of one saved wardrobe item."""

    wardrobe = load_wardrobe()

    matching_rows = (
        wardrobe["id"]
        .fillna("")
        .astype(str)
        .str.strip()
        == item_id.strip()
    )

    if not matching_rows.any():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clothing item not found.",
        )

    name = update.name.strip()
    colour = update.colour.strip()

    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item name is required.",
        )

    if not colour:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Colour is required.",
        )

    if not 1 <= update.warmth_level <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Warmth level must be between 1 and 5.",
        )

    row_index = wardrobe.index[matching_rows][0]

    editable_values = {
        "name": name,
        "category": update.category.strip(),
        "colour": colour,
        "season": update.season.strip(),
        "occasion": update.occasion.strip(),
        "warmth_level": update.warmth_level,
        "waterproof": update.waterproof.strip(),
        "layer_type": update.layer_type.strip(),
        "style": update.style.strip() or "Casual",
    }

    for column, value in editable_values.items():
        wardrobe.at[row_index, column] = value

    try:
        save_wardrobe(wardrobe)
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not update wardrobe.csv: {error}",
        ) from error

    updated_item = wardrobe_row_to_dict(
        wardrobe.loc[row_index],
    )

    return {
        **updated_item,
        "message": f"{updated_item['name']} was updated.",
    }


@app.post(
    "/api/wardrobe",
    status_code=status.HTTP_201_CREATED,
)
async def create_wardrobe_item(
    image: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form(...),
    colour: str = Form(...),
    season: str = Form(...),
    occasion: str = Form(...),
    warmth_level: int = Form(...),
    waterproof: str = Form(...),
    layer_type: str = Form(...),
    style: str = Form(...),
    ai_category: str = Form(""),
    ai_category_detail: str = Form(""),
    ai_category_confidence: str = Form(""),
    ai_colour: str = Form(""),
    ai_colour_confidence: str = Form(""),
    ai_layer_type: str = Form(""),
    ai_layer_type_confidence: str = Form(""),
    ai_style: str = Form(""),
    ai_style_confidence: str = Form(""),
) -> dict:
    name = name.strip()
    colour = colour.strip()

    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item name is required.",
        )

    if not colour:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Colour is required.",
        )

    original_filename = image.filename or ""
    extension = Path(original_filename).suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPG, JPEG, and PNG images are supported.",
        )

    if not 1 <= warmth_level <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Warmth level must be between 1 and 5.",
        )

    item_id = str(uuid4())
    image_filename = f"{item_id}{extension}"
    image_destination = UPLOAD_FOLDER / image_filename

    try:
        image_destination.write_bytes(await image.read())
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save image: {error}",
        ) from error

    wardrobe = load_wardrobe()

    new_item = pd.DataFrame(
        [
            {
                "id": item_id,
                "name": name,
                "category": category,
                "colour": colour,
                "season": season,
                "occasion": occasion,
                "warmth_level": warmth_level,
                "waterproof": waterproof,
                "layer_type": layer_type,
                "style": style,
                "image_path": str(
                    Path("backend") / "uploads" / image_filename
                ),
                "date_added": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
        ],
        columns=COLUMNS,
    )

    updated_wardrobe = pd.concat(
        [wardrobe, new_item],
        ignore_index=True,
    )

    try:
        save_wardrobe(updated_wardrobe)
    except OSError as error:
        image_destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save wardrobe.csv: {error}",
        ) from error

    prediction_was_used = any(
        normalized_label(value)
        for value in (
            ai_category,
            ai_category_detail,
            ai_colour,
            ai_layer_type,
            ai_style,
        )
    )

    if prediction_was_used:
        feedback = load_classification_feedback()

        feedback_row = pd.DataFrame(
            [
                {
                    "feedback_id": str(uuid4()),
                    "item_id": item_id,
                    "image_filename": image_filename,
                    "predicted_category": ai_category,
                    "predicted_category_detail": ai_category_detail,
                    "final_category": category,
                    "category_confidence": parse_confidence(
                        ai_category_confidence
                    ),
                    "category_corrected": label_was_corrected(
                        ai_category,
                        category,
                    ),
                    "predicted_colour": ai_colour,
                    "final_colour": colour,
                    "colour_confidence": parse_confidence(
                        ai_colour_confidence
                    ),
                    "colour_corrected": label_was_corrected(
                        ai_colour,
                        colour,
                    ),
                    "predicted_layer_type": ai_layer_type,
                    "final_layer_type": layer_type,
                    "layer_type_confidence": parse_confidence(
                        ai_layer_type_confidence
                    ),
                    "layer_type_corrected": label_was_corrected(
                        ai_layer_type,
                        layer_type,
                    ),
                    "predicted_style": ai_style,
                    "final_style": style,
                    "style_confidence": parse_confidence(
                        ai_style_confidence
                    ),
                    "style_corrected": label_was_corrected(
                        ai_style,
                        style,
                    ),
                    "date_recorded": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            ],
            columns=FEEDBACK_COLUMNS,
        )

        updated_feedback = pd.concat(
            [feedback, feedback_row],
            ignore_index=True,
        )

        try:
            save_classification_feedback(updated_feedback)
        except OSError as error:
            try:
                save_wardrobe(wardrobe)
            finally:
                image_destination.unlink(missing_ok=True)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Could not save classification_feedback.csv: "
                    f"{error}"
                ),
            ) from error

    return wardrobe_row_to_dict(new_item.iloc[0])


@app.get("/api/classification-feedback")
def get_classification_feedback() -> list[dict]:
    """Return the saved AI predictions and user corrections."""

    feedback = load_classification_feedback()

    return [
        {
            column: clean_value(row.get(column, ""))
            for column in FEEDBACK_COLUMNS
        }
        for _, row in feedback.iloc[::-1].iterrows()
    ]


@app.get("/api/outfit-recommendation")
def get_outfit_recommendation(
    occasion: str = "",
    weather: str = "",
) -> dict:
    """Create and rank complete outfits for the selected situation."""

    wardrobe = load_wardrobe()

    if wardrobe.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Your wardrobe is empty.",
        )

    def normalize_text(value) -> str:
        if pd.isna(value):
            return ""

        return str(value or "").strip().lower()

    def normalized_value(
        row: pd.Series,
        field: str,
    ) -> str:
        return normalize_text(row.get(field, ""))

    def warmth(row: pd.Series | None) -> float:
        if row is None:
            return 0.0

        try:
            return float(row.get("warmth_level", 0))
        except (TypeError, ValueError):
            return 0.0

    def item_id(row: pd.Series | None) -> str:
        if row is None:
            return ""

        return normalize_text(row.get("id", ""))

    requested_occasion = normalize_text(occasion)
    requested_weather = normalize_text(weather)

    occasion_aliases = {
        "exercise": "gym",
        "sport": "gym",
        "athletic": "gym",
        "business": "work",
        "smart casual": "date",
    }

    requested_occasion = occasion_aliases.get(
        requested_occasion,
        requested_occasion,
    )

    if requested_weather not in {
        "",
        "hot",
        "mild",
        "cold",
        "rainy",
    }:
        requested_weather = ""

    base_tops: list[pd.Series] = []
    flexible_tops: list[pd.Series] = []
    mid_layers: list[pd.Series] = []
    bottoms: list[pd.Series] = []
    outerwear_items: list[pd.Series] = []

    base_top_categories = {
        "top",
        "tops",
        "upper body",
        "t-shirt",
        "t shirt",
        "shirt",
        "button-up",
        "button-up shirt",
        "polo",
        "polo shirt",
        "tank top",
        "blouse",
    }

    mid_layer_categories = {
        "hoodie",
        "sweater",
        "sweatshirt",
        "cardigan",
    }

    bottom_categories = {
        "bottom",
        "bottoms",
        "lower body",
        "pants",
        "jeans",
        "sweatpants",
        "shorts",
        "cargo shorts",
        "skirt",
    }

    outerwear_categories = {
        "jacket",
        "outerwear",
        "outer wear",
        "coat",
        "blazer",
        "windbreaker",
    }

    outerwear_name_words = (
        "jacket",
        "windbreaker",
        "coat",
        "blazer",
        "parka",
        "raincoat",
        "rain coat",
    )

    mid_layer_name_words = (
        "hoodie",
        "zip-up",
        "zip up",
        "sweater",
        "sweatshirt",
        "cardigan",
    )

    # Bottoms and unmistakable outerwear categories are resolved
    # first. This prevents a jacket or windbreaker with an incorrect
    # saved layer type from being placed underneath another jacket.
    for _, row in wardrobe.iterrows():
        category = normalized_value(row, "category")
        name = normalized_value(row, "name")
        layer_type = normalized_value(row, "layer_type")

        if category in bottom_categories:
            bottoms.append(row)
            continue

        if (
            category in outerwear_categories
            or any(
                word in name
                for word in outerwear_name_words
            )
        ):
            outerwear_items.append(row)
            continue

        # Flexible tops may be worn alone or over a thin base shirt.
        if "flexible" in layer_type:
            flexible_tops.append(row)
            continue

        if "base" in layer_type:
            base_tops.append(row)
            continue

        if "mid" in layer_type:
            mid_layers.append(row)
            continue

        if "outer" in layer_type:
            outerwear_items.append(row)
            continue

        if (
            category in mid_layer_categories
            or any(
                word in name
                for word in mid_layer_name_words
            )
        ):
            mid_layers.append(row)
            continue

        if category in base_top_categories:
            base_tops.append(row)
            continue

        slot = identify_outfit_slot(row)

        if slot == "bottom":
            bottoms.append(row)
        elif slot == "outerwear":
            outerwear_items.append(row)
        elif slot == "top":
            base_tops.append(row)

    if not base_tops and not flexible_tops:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No usable tops were found. Add a base layer "
                "or flexible top."
            ),
        )

    if not bottoms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No bottoms were found in your wardrobe.",
        )

    def is_thin_base(row: pd.Series) -> bool:
        category = normalized_value(row, "category")
        name = normalized_value(row, "name")

        thick_words = (
            "hoodie",
            "sweater",
            "sweatshirt",
            "cardigan",
            "jacket",
            "coat",
        )

        return (
            category in base_top_categories
            and warmth(row) <= 2.5
            and not any(word in name for word in thick_words)
        )

    def can_accept_mid_layer(row: pd.Series) -> bool:
        category = normalized_value(row, "category")
        name = normalized_value(row, "name")

        substantial_words = (
            "hoodie",
            "sweater",
            "sweatshirt",
            "cardigan",
            "jacket",
            "coat",
        )

        return (
            category not in mid_layer_categories
            and warmth(row) <= 3.0
            and not any(
                word in name
                for word in substantial_words
            )
        )

    def is_standalone_base_top(
        row: pd.Series,
    ) -> bool:
        """
        A substantial garment explicitly saved as a base layer
        should be worn as the visible main top, without an
        undershirt or another sweater/hoodie above it.
        """

        category = normalized_value(row, "category")
        name = normalized_value(row, "name")
        layer_type = normalized_value(row, "layer_type")

        standalone_categories = {
            "sweater",
            "hoodie",
            "sweatshirt",
            "cardigan",
        }

        standalone_name_words = (
            "sweater",
            "hoodie",
            "sweatshirt",
            "cardigan",
            "crewneck",
            "crew neck",
        )

        return (
            "base" in layer_type
            and (
                category in standalone_categories
                or any(
                    word in name
                    for word in standalone_name_words
                )
            )
        )


    thin_base_tops = [
        row
        for row in base_tops
        if is_thin_base(row)
    ]

    # Each configuration is:
    # (base_top, mid_layer, top_mode)
    #
    # flexible_only:
    #   flexible top is the visible main top
    # base_flexible:
    #   thin shirt underneath, flexible top above it
    # base_mid:
    #   ordinary base top + ordinary hoodie/sweater
    top_configurations: list[
        tuple[pd.Series, pd.Series | None, str]
    ] = []

    for base_top in base_tops:
        top_mode = (
            "standalone_top"
            if is_standalone_base_top(base_top)
            else "base_only"
        )

        top_configurations.append(
            (base_top, None, top_mode)
        )

    for flexible_top in flexible_tops:
        top_configurations.append(
            (flexible_top, None, "flexible_only")
        )

    if requested_weather != "hot":
        for base_top in base_tops:
            if not can_accept_mid_layer(base_top):
                continue

            for mid_layer in mid_layers:
                if item_id(base_top) != item_id(mid_layer):
                    top_configurations.append(
                        (base_top, mid_layer, "base_mid")
                    )

        for base_top in thin_base_tops:
            for flexible_top in flexible_tops:
                if item_id(base_top) != item_id(flexible_top):
                    top_configurations.append(
                        (
                            base_top,
                            flexible_top,
                            "base_flexible",
                        )
                    )

    def is_water_resistant(row: pd.Series) -> bool:
        value = normalized_value(row, "waterproof")

        return value in {
            "yes",
            "true",
            "waterproof",
            "light",
            "water resistant",
            "water-resistant",
        }

    resistant_outerwear = [
        row
        for row in outerwear_items
        if is_water_resistant(row)
    ]

    raw_candidates: list[
        tuple[
            pd.Series,
            pd.Series | None,
            pd.Series,
            pd.Series | None,
            str,
        ]
    ] = []

    for base_top, mid_layer, top_mode in top_configurations:
        for bottom in bottoms:
            if requested_weather == "hot":
                if mid_layer is None:
                    raw_candidates.append(
                        (
                            base_top,
                            None,
                            bottom,
                            None,
                            top_mode,
                        )
                    )
                continue

            if requested_weather == "mild":
                # Mild weather gets at most one extra layer:
                # either a mid/flexible layer OR outerwear.
                if mid_layer is not None:
                    raw_candidates.append(
                        (
                            base_top,
                            mid_layer,
                            bottom,
                            None,
                            top_mode,
                        )
                    )
                else:
                    raw_candidates.append(
                        (
                            base_top,
                            None,
                            bottom,
                            None,
                            top_mode,
                        )
                    )

                    for outerwear in outerwear_items:
                        raw_candidates.append(
                            (
                                base_top,
                                None,
                                bottom,
                                outerwear,
                                top_mode,
                            )
                        )
                continue

            if requested_weather == "rainy":
                # Rainy is not automatically cold. Use a main top,
                # bottom, and rain-ready outerwear without forcing a
                # fourth clothing piece.
                if mid_layer is not None:
                    continue

                rainy_outerwear = (
                    resistant_outerwear
                    if resistant_outerwear
                    else outerwear_items
                )

                if rainy_outerwear:
                    for outerwear in rainy_outerwear:
                        raw_candidates.append(
                            (
                                base_top,
                                None,
                                bottom,
                                outerwear,
                                top_mode,
                            )
                        )
                else:
                    raw_candidates.append(
                        (
                            base_top,
                            None,
                            bottom,
                            None,
                            top_mode,
                        )
                    )
                continue

            if requested_weather == "cold":
                # Cold weather requires either a warm visible top,
                # a mid/flexible layer, or outerwear.
                outerwear_choices = [None, *outerwear_items]

                for outerwear in outerwear_choices:
                    has_warm_top = (
                        top_mode == "flexible_only"
                        and warmth(base_top) >= 3.5
                    )

                    if (
                        mid_layer is None
                        and outerwear is None
                        and not has_warm_top
                    ):
                        continue

                    raw_candidates.append(
                        (
                            base_top,
                            mid_layer,
                            bottom,
                            outerwear,
                            top_mode,
                        )
                    )
                continue

            # "Any" weather keeps the combinations broad but still
            # obeys the flexible-top layering rules.
            raw_candidates.append(
                (
                    base_top,
                    mid_layer,
                    bottom,
                    None,
                    top_mode,
                )
            )

            for outerwear in outerwear_items:
                raw_candidates.append(
                    (
                        base_top,
                        mid_layer,
                        bottom,
                        outerwear,
                        top_mode,
                    )
                )

    # Gym outfits should not stack a hoodie/sweater and jacket
    # unless the user explicitly selected cold weather.
    if (
        requested_occasion == "gym"
        and requested_weather != "cold"
    ):
        single_extra_layer_candidates = [
            candidate
            for candidate in raw_candidates
            if not (
                candidate[1] is not None
                and candidate[3] is not None
            )
        ]

        if single_extra_layer_candidates:
            raw_candidates = single_extra_layer_candidates

    # Party outfits should use either a mid-layer or outerwear,
    # not both, unless cold weather explicitly requires layering.
    if (
        requested_occasion == "party"
        and requested_weather != "cold"
    ):
        party_single_layer_candidates = [
            candidate
            for candidate in raw_candidates
            if not (
                candidate[1] is not None
                and candidate[3] is not None
            )
        ]

        if party_single_layer_candidates:
            raw_candidates = party_single_layer_candidates

    # Only explicitly cold weather may stack both a mid-layer
    # and outerwear. "Any" weather should not assume that four
    # clothing pieces are necessary.
    if requested_weather != "cold":
        non_cold_single_extra_candidates = [
            candidate
            for candidate in raw_candidates
            if not (
                candidate[1] is not None
                and candidate[3] is not None
            )
        ]

        if non_cold_single_extra_candidates:
            raw_candidates = non_cold_single_extra_candidates

    def is_soft_extra_layer(
        row: pd.Series | None,
    ) -> bool:
        """
        Hoodies, sweaters, and light windbreakers occupy the same
        practical layering position and should not stack together.
        """

        if row is None:
            return False

        category = normalized_value(row, "category")
        name = normalized_value(row, "name")

        searchable_text = f"{category} {name}"

        extra_layer_words = (
            "hoodie",
            "zip-up",
            "zip up",
            "sweater",
            "sweatshirt",
            "cardigan",
            "crewneck",
            "crew neck",
            "windbreaker",
            "track jacket",
        )

        return any(
            word in searchable_text
            for word in extra_layer_words
        )

    # A windbreaker and hoodie/sweater belong to the same practical
    # layer group. Never recommend them together, including in cold
    # weather. Heavy coats and substantial jackets may still be worn
    # over a hoodie when cold weather requires it.
    compatible_layer_candidates = [
        candidate
        for candidate in raw_candidates
        if not (
            candidate[1] is not None
            and candidate[3] is not None
            and is_soft_extra_layer(candidate[1])
            and is_soft_extra_layer(candidate[3])
        )
    ]

    if compatible_layer_candidates:
        raw_candidates = compatible_layer_candidates

    if not raw_candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No valid outfit matched these settings. "
                "Try another weather option or update an item's "
                "layer type."
            ),
        )

    def canonical_colour(row: pd.Series) -> str:
        value = normalized_value(row, "colour")

        colour_words = (
            "black",
            "white",
            "grey",
            "gray",
            "beige",
            "cream",
            "brown",
            "navy",
            "blue",
            "green",
            "red",
            "burgundy",
            "pink",
            "purple",
            "yellow",
            "orange",
        )

        for colour_word in colour_words:
            if colour_word in value:
                return (
                    "grey"
                    if colour_word == "gray"
                    else colour_word
                )

        return value

    neutral_colours = {
        "black",
        "white",
        "grey",
        "beige",
        "cream",
        "brown",
        "navy",
    }

    compatible_colour_pairs = {
        frozenset({"black", "white"}),
        frozenset({"black", "grey"}),
        frozenset({"black", "beige"}),
        frozenset({"black", "brown"}),
        frozenset({"black", "red"}),
        frozenset({"black", "burgundy"}),
        frozenset({"white", "blue"}),
        frozenset({"white", "green"}),
        frozenset({"white", "brown"}),
        frozenset({"white", "navy"}),
        frozenset({"grey", "blue"}),
        frozenset({"grey", "burgundy"}),
        frozenset({"beige", "brown"}),
        frozenset({"beige", "navy"}),
        frozenset({"brown", "blue"}),
        frozenset({"brown", "navy"}),
        frozenset({"red", "blue"}),
    }

    def colour_pair_score(
        first: pd.Series,
        second: pd.Series,
    ) -> float:
        first_colour = canonical_colour(first)
        second_colour = canonical_colour(second)

        if not first_colour or not second_colour:
            return 0.0

        if first_colour == second_colour:
            return 1.5

        if (
            first_colour in neutral_colours
            or second_colour in neutral_colours
        ):
            return 3.0

        if (
            frozenset({first_colour, second_colour})
            in compatible_colour_pairs
        ):
            return 2.5

        return -2.5

    occasion_preferences = {
        "casual": {
            "occasions": {"casual", "school", ""},
            "styles": {
                "casual",
                "streetwear",
                "minimal",
                "minimalist",
                "vintage",
                "smart casual",
            },
        },
        "party": {
            "occasions": {
                "party",
                "casual",
                "formal",
                "date",
                "",
            },
            "styles": {
                "smart casual",
                "formal",
                "vintage",
                "minimal",
                "minimalist",
                "streetwear",
            },
        },
        "date": {
            "occasions": {
                "date",
                "casual",
                "formal",
                "school",
                "",
            },
            "styles": {
                "smart casual",
                "formal",
                "vintage",
                "minimal",
                "minimalist",
            },
        },
        "work": {
            "occasions": {
                "work",
                "business",
                "formal",
                "school",
                "",
            },
            "styles": {
                "formal",
                "smart casual",
                "minimal",
                "minimalist",
            },
        },
        "formal": {
            "occasions": {
                "formal",
                "business",
                "work",
                "",
            },
            "styles": {
                "formal",
                "smart casual",
                "minimal",
                "minimalist",
            },
        },
        "gym": {
            "occasions": {
                "gym",
                "exercise",
                "sport",
                "athletic",
                "",
            },
            "styles": {
                "athletic",
                "sporty",
                "sport",
            },
        },
    }

    dressy_occasion_words = {
        "party",
        "date",
        "work",
        "formal",
    }

    athletic_name_words = (
        "sweatpants",
        "track pants",
        "track pant",
        "gym",
        "athletic",
        "sport shorts",
    )

    dressy_name_words = (
        "button-up",
        "button up",
        "fitted",
        "polo",
        "ruffle",
        "pleated",
        "leather",
        "high collar",
    )

    def occasion_item_score(row: pd.Series) -> float:
        if not requested_occasion:
            return 0.0

        item_occasion = normalized_value(
            row,
            "occasion",
        )
        item_style = normalized_value(row, "style")
        item_name = normalized_value(row, "name")
        item_category = normalized_value(
            row,
            "category",
        )
        item_layer = normalized_value(
            row,
            "layer_type",
        )

        searchable_text = " ".join(
            [
                item_name,
                item_category,
                item_style,
                item_occasion,
                item_layer,
            ]
        )

        preferences = occasion_preferences.get(
            requested_occasion,
            {
                "occasions": {
                    requested_occasion,
                    "",
                },
                "styles": set(),
            },
        )

        def contains_any(words: tuple[str, ...]) -> bool:
            return any(
                word in searchable_text
                for word in words
            )

        score = 0.0

        # First respect the occasion saved with the item.
        if item_occasion == requested_occasion:
            score += 9.0
        elif item_occasion in preferences["occasions"]:
            score += 3.0
        elif not item_occasion:
            score += 1.0
        else:
            score -= 3.0

        if item_style in preferences["styles"]:
            score += 4.0

        athletic_words = (
            "sweatpants",
            "track pants",
            "track pant",
            "jogger",
            "gym",
            "athletic",
            "exercise",
            "sport shorts",
            "basketball shorts",
        )

        polished_words = (
            "button-up",
            "button up",
            "dress shirt",
            "polo",
            "blouse",
            "cardigan",
            "sweater",
            "trouser",
            "slacks",
            "pleated",
            "blazer",
        )

        # Work: clean and presentable, with strong penalties
        # for obvious gym clothing.
        if requested_occasion == "work":
            if contains_any(polished_words):
                score += 6.0

            if contains_any(
                (
                    "pants",
                    "trouser",
                    "slacks",
                )
            ):
                score += 4.0

            if "jeans" in searchable_text:
                score += 1.0

            if contains_any(athletic_words):
                score -= 12.0

            if "hoodie" in searchable_text:
                score -= 7.0

            if "shorts" in searchable_text:
                score -= 7.0

            if contains_any(
                (
                    "t-shirt",
                    "t shirt",
                    "graphic tee",
                )
            ):
                score -= 2.0

        # Date: styled but not overly formal.
        elif requested_occasion == "date":
            if contains_any(
                (
                    "sweater",
                    "cardigan",
                    "polo",
                    "button-up",
                    "button up",
                    "blouse",
                )
            ):
                score += 6.0

            if contains_any(
                (
                    "jacket",
                    "blazer",
                    "leather",
                )
            ):
                score += 4.0

            if contains_any(
                (
                    "jeans",
                    "pants",
                    "skirt",
                    "pleated",
                )
            ):
                score += 3.0

            if contains_any(athletic_words):
                score -= 10.0

            if "hoodie" in searchable_text:
                score -= 3.0

        # Party: favor expressive or structured pieces.
        elif requested_occasion == "party":
            # Strong bonuses are reserved for genuinely styled
            # details. A generic winter jacket should not dominate.
            if contains_any(
                (
                    "leather",
                    "blazer",
                    "fitted",
                    "high collar",
                    "ruffle",
                    "pleated",
                )
            ):
                score += 6.0

            if contains_any(
                (
                    "v-neck",
                    "v neck",
                )
            ):
                score += 3.0

            if contains_any(
                (
                    "polo",
                    "button-up",
                    "button up",
                    "sweater",
                    "cardigan",
                )
            ):
                score += 4.0

            if contains_any(
                (
                    "jacket",
                    "coat",
                    "windbreaker",
                )
            ):
                score += 2.0

                # With no weather selected, avoid constantly choosing
                # heavy winter outerwear for an indoor party outfit.
                if not requested_weather:
                    if warmth(row) >= 4.5:
                        score -= 6.0

                    if (
                        normalized_value(row, "season")
                        == "winter"
                    ):
                        score -= 4.0

            if contains_any(
                (
                    "jeans",
                    "pants",
                    "skirt",
                )
            ):
                score += 2.0

            if contains_any(athletic_words):
                score -= 10.0

            if "hoodie" in searchable_text:
                score -= 4.0

        # Formal: much stricter than Work.
        elif requested_occasion == "formal":
            if contains_any(
                (
                    "button-up",
                    "button up",
                    "dress shirt",
                    "blouse",
                    "blazer",
                    "trouser",
                    "slacks",
                    "pleated",
                )
            ):
                score += 9.0

            if "polo" in searchable_text:
                score += 3.0

            if "jeans" in searchable_text:
                score -= 6.0

            if contains_any(athletic_words):
                score -= 15.0

            if "hoodie" in searchable_text:
                score -= 12.0

            if contains_any(
                (
                    "t-shirt",
                    "t shirt",
                    "shorts",
                )
            ):
                score -= 8.0

        # Gym: prioritize athletic basics and outerwear.
        elif requested_occasion == "gym":
            if contains_any(athletic_words):
                score += 10.0

            if contains_any(
                (
                    "t-shirt",
                    "t shirt",
                    "hoodie",
                    "sweatshirt",
                    "shorts",
                    "windbreaker",
                )
            ):
                score += 6.0

            if contains_any(
                (
                    "jeans",
                    "skirt",
                    "leather",
                    "button-up",
                    "button up",
                    "dress shirt",
                    "blazer",
                )
            ):
                score -= 12.0

        # Casual stays broad and forgiving.
        elif requested_occasion == "casual":
            if contains_any(
                (
                    "t-shirt",
                    "t shirt",
                    "jeans",
                    "hoodie",
                    "sweater",
                    "sweatshirt",
                    "cargo",
                )
            ):
                score += 3.0

        return score


    def weather_item_score(
        row: pd.Series,
        slot: str,
    ) -> float:
        if not requested_weather:
            return 0.0

        item_season = normalized_value(row, "season")
        item_warmth = warmth(row)
        score = 0.0

        season_scores = {
            "hot": {
                "summer": 5.0,
                "all season": 2.5,
                "all seasons": 2.5,
                "spring": 1.0,
                "fall": -6.0,
                "winter": -10.0,
            },
            "mild": {
                "spring": 4.0,
                "fall": 4.0,
                "all season": 3.0,
                "all seasons": 3.0,
                "summer": 1.0,
                "winter": -3.0,
            },
            "cold": {
                "winter": 6.0,
                "fall": 4.0,
                "all season": 2.0,
                "all seasons": 2.0,
                "spring": 0.0,
                "summer": -7.0,
            },
            "rainy": {
                "spring": 4.0,
                "fall": 4.0,
                "all season": 2.0,
                "all seasons": 2.0,
                "summer": 0.0,
                "winter": 0.0,
            },
        }

        score += season_scores.get(
            requested_weather,
            {},
        ).get(item_season, 0.0)

        slot_targets = {
            "hot": {
                "base_top": 1.5,
                "bottom": 1.5,
            },
            "mild": {
                "base_top": 2.0,
                "bottom": 2.5,
                "mid_layer": 2.5,
                "outerwear": 3.0,
            },
            "cold": {
                "base_top": 2.5,
                "bottom": 3.5,
                "mid_layer": 4.0,
                "outerwear": 4.5,
            },
            "rainy": {
                "base_top": 2.0,
                "bottom": 2.5,
                "outerwear": 3.5,
            },
        }

        target = slot_targets.get(
            requested_weather,
            {},
        ).get(slot)

        if target is not None:
            score += max(
                -5.0,
                3.0 - abs(item_warmth - target),
            )

        if (
            requested_weather == "rainy"
            and slot == "outerwear"
            and is_water_resistant(row)
        ):
            score += 8.0

        return score

    def style_compatibility_score(
        rows: list[pd.Series],
    ) -> float:
        styles = [
            normalized_value(row, "style")
            for row in rows
            if normalized_value(row, "style")
        ]

        if len(styles) < 2:
            return 0.0

        score = 0.0

        for index, first_style in enumerate(styles):
            for second_style in styles[index + 1:]:
                if first_style == second_style:
                    score += 1.5
                elif {
                    first_style,
                    second_style,
                } <= {
                    "casual",
                    "streetwear",
                    "minimal",
                    "minimalist",
                    "smart casual",
                    "vintage",
                }:
                    score += 0.75
                elif (
                    first_style
                    in {"athletic", "sporty", "sport"}
                    and second_style
                    not in {
                        "athletic",
                        "sporty",
                        "sport",
                        "casual",
                    }
                ):
                    score -= 3.0
                elif (
                    second_style
                    in {"athletic", "sporty", "sport"}
                    and first_style
                    not in {
                        "athletic",
                        "sporty",
                        "sport",
                        "casual",
                    }
                ):
                    score -= 3.0

        return score

    feedback_file = DATA_FOLDER / "outfit_feedback.csv"
    exact_feedback_scores: dict[
        tuple[str, str, str, str],
        float,
    ] = {}
    item_feedback_scores: dict[str, float] = {}

    if feedback_file.exists():
        try:
            feedback = pd.read_csv(feedback_file)
        except (
            pd.errors.EmptyDataError,
            OSError,
        ):
            feedback = pd.DataFrame()

        for _, feedback_row in feedback.iterrows():
            rating = normalize_text(
                feedback_row.get("rating", "")
            )

            rating_value = (
                1.0
                if rating == "like"
                else -1.0
                if rating == "dislike"
                else 0.0
            )

            if rating_value == 0:
                continue

            feedback_key = (
                normalize_text(
                    feedback_row.get("base_top_id", "")
                ),
                normalize_text(
                    feedback_row.get("mid_layer_id", "")
                ),
                normalize_text(
                    feedback_row.get("bottom_id", "")
                ),
                normalize_text(
                    feedback_row.get("outerwear_id", "")
                ),
            )

            exact_feedback_scores[feedback_key] = (
                exact_feedback_scores.get(
                    feedback_key,
                    0.0,
                )
                + rating_value
            )

            for column in (
                "base_top_id",
                "mid_layer_id",
                "bottom_id",
                "outerwear_id",
            ):
                feedback_item_id = normalize_text(
                    feedback_row.get(column, "")
                )

                if feedback_item_id:
                    item_feedback_scores[feedback_item_id] = (
                        item_feedback_scores.get(
                            feedback_item_id,
                            0.0,
                        )
                        + rating_value
                    )

    total_warmth_targets = {
        "hot": 3.0,
        "mild": 6.0,
        "cold": 12.0,
        "rainy": 7.5,
    }

    candidate_outfits: list[
        tuple[
            float,
            pd.Series,
            pd.Series | None,
            pd.Series,
            pd.Series | None,
            str,
        ]
    ] = []

    for (
        base_top,
        mid_layer,
        bottom,
        outerwear,
        top_mode,
    ) in raw_candidates:
        rows = [
            row
            for row in (
                base_top,
                mid_layer,
                bottom,
                outerwear,
            )
            if row is not None
        ]

        ids = [item_id(row) for row in rows]

        if len(ids) != len(set(ids)):
            continue

        score = 0.0

        score += occasion_item_score(base_top)
        score += occasion_item_score(bottom)
        score += weather_item_score(
            base_top,
            "base_top",
        )
        score += weather_item_score(
            bottom,
            "bottom",
        )

        if mid_layer is not None:
            score += occasion_item_score(mid_layer)
            score += weather_item_score(
                mid_layer,
                "mid_layer",
            )

        if outerwear is not None:
            score += occasion_item_score(outerwear)
            score += weather_item_score(
                outerwear,
                "outerwear",
            )

        for index, first_row in enumerate(rows):
            for second_row in rows[index + 1:]:
                score += colour_pair_score(
                    first_row,
                    second_row,
                )

        non_neutral_colours = {
            canonical_colour(row)
            for row in rows
            if canonical_colour(row)
            and canonical_colour(row)
            not in neutral_colours
        }

        if len(non_neutral_colours) > 2:
            score -= 5.0 * (
                len(non_neutral_colours) - 2
            )

        score += style_compatibility_score(rows)

        total_warmth = sum(
            warmth(row)
            for row in rows
        )

        target_total_warmth = total_warmth_targets.get(
            requested_weather
        )

        if target_total_warmth is not None:
            score += max(
                -10.0,
                7.0
                - 1.8
                * abs(
                    total_warmth
                    - target_total_warmth
                ),
            )

        # Prefer the special flexible modes when they fit the weather,
        # but do not force an undershirt in every cold outfit.
        if top_mode in {
            "flexible_only",
            "standalone_top",
        }:
            score += (
                4.0
                if top_mode == "standalone_top"
                else 2.5
            )

            if requested_weather == "cold":
                score += max(
                    0.0,
                    warmth(base_top) - 2.5,
                )

        if top_mode == "base_flexible":
            if requested_weather == "cold":
                score += 4.0
            elif requested_weather == "mild":
                score += 1.0
            else:
                score -= 1.0

        if requested_weather == "cold":
            if outerwear is not None:
                score += 4.0

            if (
                mid_layer is not None
                and outerwear is not None
            ):
                score += 2.0

        if requested_weather == "rainy":
            if outerwear is None:
                score -= 25.0
            elif is_water_resistant(outerwear):
                score += 8.0

        if requested_weather == "mild":
            if (
                mid_layer is None
                and outerwear is None
            ):
                score += 1.0

        feedback_key = (
            item_id(base_top),
            item_id(mid_layer),
            item_id(bottom),
            item_id(outerwear),
        )

        exact_feedback = exact_feedback_scores.get(
            feedback_key,
            0.0,
        )

        if exact_feedback < 0:
            score -= 30.0
        elif exact_feedback > 0:
            score += 15.0

        for row in rows:
            score += (
                1.25
                * item_feedback_scores.get(
                    item_id(row),
                    0.0,
                )
            )

        score += random.uniform(0.0, 0.75)

        candidate_outfits.append(
            (
                score,
                base_top,
                mid_layer,
                bottom,
                outerwear,
                top_mode,
            )
        )

    if not candidate_outfits:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No valid outfit matched these settings. "
                "Try a different occasion or weather."
            ),
        )

    candidate_outfits.sort(
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    def full_outfit_signature(
        candidate,
    ) -> tuple[str, str, str, str]:
        return (
            item_id(candidate[1]),
            item_id(candidate[2]),
            item_id(candidate[3]),
            item_id(candidate[4]),
        )

    def top_bottom_signature(
        candidate,
    ) -> tuple[str, str, str]:
        visible_top = (
            candidate[2]
            if candidate[2] is not None
            else candidate[1]
        )

        return (
            item_id(visible_top),
            item_id(candidate[3]),
            item_id(candidate[4]),
        )

    if requested_occasion == "party":
        top_score = candidate_outfits[0][0]
        party_pool = []
        seen_top_bottom_pairs = set()
        party_outerwear_counts = {}

        # Build a shortlist with genuinely different main tops and
        # bottoms instead of six layer variants of the same outfit.
        for candidate in candidate_outfits[:40]:
            pair = top_bottom_signature(candidate)
            outerwear_key = item_id(candidate[4])

            if pair in seen_top_bottom_pairs:
                continue

            # A specific jacket may appear in no more than two
            # Party shortlist candidates.
            if (
                outerwear_key
                and party_outerwear_counts.get(
                    outerwear_key,
                    0,
                ) >= 2
            ):
                continue

            if (
                candidate[0] < top_score - 14.0
                and len(party_pool) >= 6
            ):
                break

            seen_top_bottom_pairs.add(pair)

            if outerwear_key:
                party_outerwear_counts[outerwear_key] = (
                    party_outerwear_counts.get(
                        outerwear_key,
                        0,
                    )
                    + 1
                )

            party_pool.append(candidate)

            if len(party_pool) >= 15:
                break

        shortlist = (
            party_pool
            if party_pool
            else candidate_outfits[
                : min(10, len(candidate_outfits))
            ]
        )
    else:
        shortlist = candidate_outfits[
            : min(6, len(candidate_outfits))
        ]

    # Keep a small in-memory history so Generate Another Outfit does
    # not immediately return the same result again.
    history_store = getattr(
        get_outfit_recommendation,
        "_recent_outfit_history",
        {},
    )

    history_key = (
        requested_occasion,
        requested_weather,
    )

    recent_signatures = history_store.get(
        history_key,
        [],
    )

    if (
        requested_occasion == "party"
        and recent_signatures
    ):
        recent_visible_top_ids = {
            signature[0]
            for signature in recent_signatures
            if len(signature) >= 1
        }

        recent_outerwear_ids = {
            signature[2]
            for signature in recent_signatures
            if (
                len(signature) >= 3
                and signature[2]
            )
        }

        fully_fresh_shortlist = [
            candidate
            for candidate in shortlist
            if (
                top_bottom_signature(candidate)[0]
                not in recent_visible_top_ids
                and (
                    not item_id(candidate[4])
                    or item_id(candidate[4])
                    not in recent_outerwear_ids
                )
            )
        ]

        if fully_fresh_shortlist:
            shortlist = fully_fresh_shortlist
        else:
            fresh_top_shortlist = [
                candidate
                for candidate in shortlist
                if (
                    top_bottom_signature(candidate)[0]
                    not in recent_visible_top_ids
                )
            ]

            if fresh_top_shortlist:
                shortlist = fresh_top_shortlist

    signature_function = (
        top_bottom_signature
        if requested_occasion == "party"
        else full_outfit_signature
    )

    non_recent_shortlist = [
        candidate
        for candidate in shortlist
        if signature_function(candidate)
        not in recent_signatures
    ]

    if non_recent_shortlist:
        shortlist = non_recent_shortlist

    if requested_occasion == "party":
        # Party results remain high-scoring, but the flatter weights
        # provide more visible variety.
        shortlist_weights = [
            max(1.0, 2.0 - index * 0.08)
            for index in range(len(shortlist))
        ]
    else:
        shortlist_weights = list(
            range(len(shortlist), 0, -1)
        )

    (
        selected_score,
        selected_base_top,
        selected_mid_layer,
        selected_bottom,
        selected_outerwear,
        selected_top_mode,
    ) = random.choices(
        shortlist,
        weights=shortlist_weights,
        k=1,
    )[0]

    selected_candidate = (
        selected_score,
        selected_base_top,
        selected_mid_layer,
        selected_bottom,
        selected_outerwear,
        selected_top_mode,
    )

    selected_signature = signature_function(
        selected_candidate
    )

    updated_history = [
        signature
        for signature in recent_signatures
        if signature != selected_signature
    ]

    updated_history.append(selected_signature)

    history_store[history_key] = updated_history[-4:]

    setattr(
        get_outfit_recommendation,
        "_recent_outfit_history",
        history_store,
    )

    base_top_result = wardrobe_row_to_dict(
        selected_base_top
    )

    role_labels = {
        "base_top": (
            "Main top"
            if selected_top_mode in {
                "flexible_only",
                "standalone_top",
            }
            else "Base top"
        ),
        "mid_layer": (
            "Flexible top"
            if selected_top_mode == "base_flexible"
            else "Mid layer"
        ),
        "bottom": "Bottom",
        "outerwear": "Outerwear",
    }

    return {
        "top": base_top_result,
        "base_top": base_top_result,
        "mid_layer": (
            wardrobe_row_to_dict(selected_mid_layer)
            if selected_mid_layer is not None
            else None
        ),
        "bottom": wardrobe_row_to_dict(
            selected_bottom
        ),
        "outerwear": (
            wardrobe_row_to_dict(selected_outerwear)
            if selected_outerwear is not None
            else None
        ),
        "roles": role_labels,
        "top_mode": selected_top_mode,
        "filters": {
            "occasion": requested_occasion or "any",
            "weather": requested_weather or "any",
        },
        "recommendation_score": round(
            selected_score,
            2,
        ),
    }


@app.post("/api/outfit-feedback")
def save_outfit_feedback(
    rating: str = Form(...),
    base_top_id: str = Form(""),
    mid_layer_id: str = Form(""),
    bottom_id: str = Form(""),
    outerwear_id: str = Form(""),
    occasion: str = Form(""),
) -> dict:
    """Save whether the user liked or disliked an outfit."""

    import csv
    from datetime import datetime
    from uuid import uuid4

    feedback_file = DATA_FOLDER / "outfit_feedback.csv"

    feedback_columns = [
        "feedback_id",
        "rating",
        "base_top_id",
        "base_top_name",
        "mid_layer_id",
        "mid_layer_name",
        "bottom_id",
        "bottom_name",
        "outerwear_id",
        "outerwear_name",
        "occasion",
        "date_recorded",
    ]

    normalized_rating = str(rating).strip().lower()

    if normalized_rating not in {"like", "dislike"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be either like or dislike.",
        )

    wardrobe = load_wardrobe()

    def item_details(item_id: str) -> tuple[str, str]:
        normalized_id = str(item_id or "").strip()

        if not normalized_id:
            return "", ""

        if "id" not in wardrobe.columns:
            return normalized_id, ""

        matches = wardrobe[
            wardrobe["id"].astype(str) == normalized_id
        ]

        if matches.empty:
            return normalized_id, ""

        item = matches.iloc[0]
        name_value = item.get("name", "")

        if pd.isna(name_value):
            item_name = ""
        else:
            item_name = str(name_value).strip()

        return normalized_id, item_name

    base_top_id, base_top_name = item_details(
        base_top_id
    )

    mid_layer_id, mid_layer_name = item_details(
        mid_layer_id
    )

    bottom_id, bottom_name = item_details(
        bottom_id
    )

    outerwear_id, outerwear_name = item_details(
        outerwear_id
    )

    if not base_top_id or not bottom_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The outfit must contain a base top "
                "and bottom."
            ),
        )

    feedback_id = str(uuid4())

    feedback_row = {
        "feedback_id": feedback_id,
        "rating": normalized_rating,
        "base_top_id": base_top_id,
        "base_top_name": base_top_name,
        "mid_layer_id": mid_layer_id,
        "mid_layer_name": mid_layer_name,
        "bottom_id": bottom_id,
        "bottom_name": bottom_name,
        "outerwear_id": outerwear_id,
        "outerwear_name": outerwear_name,
        "occasion": str(occasion or "").strip(),
        "date_recorded": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    feedback_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_has_content = (
        feedback_file.exists()
        and feedback_file.stat().st_size > 0
    )

    try:
        with feedback_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as opened_file:
            writer = csv.DictWriter(
                opened_file,
                fieldnames=feedback_columns,
            )

            if not file_has_content:
                writer.writeheader()

            writer.writerow(feedback_row)

    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save outfit feedback: {error}",
        ) from error

    return {
        "message": "Outfit feedback saved.",
        "feedback_id": feedback_id,
        "rating": normalized_rating,
    }


