from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
DATA_FOLDER = PROJECT_ROOT / "data"
WARDROBE_FILE = DATA_FOLDER / "wardrobe.csv"

UPLOAD_FOLDER.mkdir(exist_ok=True)
DATA_FOLDER.mkdir(exist_ok=True)


def load_wardrobe() -> pd.DataFrame:
    """Load saved clothing information."""

    if not WARDROBE_FILE.exists():
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "category",
                "color",
                "season",
                "occasion",
                "image_path",
                "date_added",
            ]
        )

    return pd.read_csv(WARDROBE_FILE)


def save_clothing_item(
    uploaded_file,
    name: str,
    category: str,
    color: str,
    season: str,
    occasion: str,
) -> None:
    """Save an uploaded clothing image and its information."""

    item_id = str(uuid4())
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")

    image_filename = f"{item_id}{file_extension}"
    image_path = UPLOAD_FOLDER / image_filename

    image_path.write_bytes(uploaded_file.getbuffer())

    wardrobe = load_wardrobe()

    new_item = pd.DataFrame(
        [
            {
                "id": item_id,
                "name": name.strip(),
                "category": category,
                "color": color.strip(),
                "season": season,
                "occasion": occasion,
                "image_path": str(image_path),
                "date_added": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )

    updated_wardrobe = pd.concat(
        [wardrobe, new_item],
        ignore_index=True,
    )

    updated_wardrobe.to_csv(WARDROBE_FILE, index=False)


st.set_page_config(
    page_title="Smart Closet",
    page_icon="👕",
    layout="wide",
)

st.title("👕 Smart Closet")
st.write("Upload and organize the clothes in your wardrobe.")

st.header("Add a clothing item")

with st.form("add_clothing_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Upload a clothing photo",
        type=["jpg", "jpeg", "png"],
    )

    name = st.text_input(
        "Item name",
        placeholder="Example: Black oversized hoodie",
    )

    category = st.selectbox(
        "Category",
        [
            "Top",
            "Bottom",
            "Dress",
            "Jacket",
            "Shoes",
            "Accessory",
            "Other",
        ],
    )

    color = st.text_input(
        "Color",
        placeholder="Example: Black",
    )

    season = st.selectbox(
        "Season",
        [
            "All seasons",
            "Spring",
            "Summer",
            "Fall",
            "Winter",
        ],
    )

    occasion = st.selectbox(
        "Occasion",
        [
            "Casual",
            "School",
            "Work",
            "Formal",
            "Exercise",
            "Other",
        ],
    )

    submitted = st.form_submit_button("Add to wardrobe")

if submitted:
    if uploaded_file is None:
        st.error("Please upload a clothing photo.")
    elif not name.strip():
        st.error("Please enter an item name.")
    elif not color.strip():
        st.error("Please enter a color.")
    else:
        try:
            save_clothing_item(
                uploaded_file=uploaded_file,
                name=name,
                category=category,
                color=color,
                season=season,
                occasion=occasion,
            )
            st.success("The clothing item was added successfully.")
        except ValueError as error:
            st.error(str(error))
        except OSError as error:
            st.error(f"The item could not be saved: {error}")

st.divider()
st.header("My Wardrobe")

wardrobe = load_wardrobe()

if wardrobe.empty:
    st.info("Your wardrobe is currently empty.")
else:
    for _, item in wardrobe.iloc[::-1].iterrows():
        image_column, information_column = st.columns([1, 2])

        with image_column:
            image_path = Path(item["image_path"])

            if image_path.exists():
                st.image(str(image_path), width=220)
            else:
                st.warning("Image unavailable.")

        with information_column:
            st.subheader(item["name"])
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Color:** {item['color']}")
            st.write(f"**Season:** {item['season']}")
            st.write(f"**Occasion:** {item['occasion']}")

        st.divider()