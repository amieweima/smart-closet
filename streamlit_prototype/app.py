from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_FOLDER = PROJECT_ROOT / "backend"
UPLOAD_FOLDER = BACKEND_FOLDER / "uploads"
DATA_FOLDER = BACKEND_FOLDER / "data"
WARDROBE_FILE = DATA_FOLDER / "wardrobe.csv"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

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


def load_wardrobe() -> pd.DataFrame:
    """Load saved clothing information."""

    if not WARDROBE_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        wardrobe = pd.read_csv(WARDROBE_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except Exception as error:
        st.error(f"Could not read wardrobe.csv: {error}")
        return pd.DataFrame(columns=COLUMNS)

    # Support clothing saved using the older American spelling.
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

    # Add any other missing columns.
    for column in COLUMNS:
        if column not in wardrobe.columns:
            wardrobe[column] = ""

    wardrobe["warmth_level"] = pd.to_numeric(
        wardrobe["warmth_level"],
        errors="coerce",
    ).fillna(3)

    return wardrobe[COLUMNS]

def resolve_image_path(saved_path) -> Path | None:
    """Find an image after the project folders have been reorganized."""

    if saved_path is None or pd.isna(saved_path):
        return None

    saved_path_text = str(saved_path).strip()

    if not saved_path_text:
        return None

    original_path = Path(saved_path_text)
    image_filename = original_path.name

    possible_paths = [
        original_path,
        PROJECT_ROOT / original_path,
        UPLOAD_FOLDER / image_filename,
        PROJECT_ROOT / "uploads" / image_filename,
        PROJECT_ROOT / "streamlit_prototype" / "uploads" / image_filename,
        PROJECT_ROOT / "_migration_backup" / "root_uploads" / image_filename,
        PROJECT_ROOT / "frontend" / "public" / "images" / image_filename,
    ]

    for possible_path in possible_paths:
        try:
            if possible_path.exists() and possible_path.is_file():
                return possible_path.resolve()
        except OSError:
            continue

    # Final fallback: search the project for the matching filename.
    try:
        for matching_path in PROJECT_ROOT.rglob(image_filename):
            if matching_path.is_file():
                return matching_path.resolve()
    except OSError:
        pass

    return None


def save_clothing_item(
    uploaded_file,
    name: str,
    category: str,
    colour: str,
    season: str,
    occasion: str,
    warmth_level: int,
    waterproof: str,
    layer_type: str,
    style: str,
) -> None:
    #Saves one Clothing Item
    """Save an uploaded clothing image and its information."""

    item_id = str(uuid4()) #Creates a random unique ID for the clothing item
    file_extension = Path(uploaded_file.name).suffix.lower() #Gets the file type of the uploaded file and converts it to lowercase

    if file_extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")

    image_filename = f"{item_id}{file_extension}" #Creates a file name using the unique ID
    image_path = UPLOAD_FOLDER / image_filename #Creates the full image path deciding where the image will be saved
    ''' 
    Example:
    UPLOAD_FOLDER = Path("uploads")
    Then the image path becomes
    uploads/8f34a639-2bd1-4c39-8d93-91abc.png
    '''
    image_path.write_bytes(uploaded_file.getbuffer()) #Saves the uploaded image into project folder

    wardrobe = load_wardrobe() #Loads the existing wardrob.csv 

    new_item = pd.DataFrame(
        [
            {
                "id": item_id,
                "name": name.strip(),
                "category": category,
                "colour": colour.strip(),
                "season": season,
                "occasion": occasion,
                "warmth_level": warmth_level,
                "waterproof": waterproof,
                "layer_type": layer_type,
                "style": style,
                "image_path": str(image_path), #Saves the image path as a string into the wardrobe.csv file
                "date_added": datetime.now().isoformat(timespec="seconds"),
            }
        ]
    )

    updated_wardrobe = pd.concat(
        [wardrobe, new_item],
        ignore_index=True,
    )
    #Adds the new item to the old wardrobe

    updated_wardrobe.to_csv(WARDROBE_FILE, index=False) #Saves everything back to CSV file
    

st.set_page_config(
    page_title="Smart Closet",
    page_icon="👕",
    layout="wide",
)

st.title("👕 Smart Closet")
st.write("Upload clothes, label them, and build your own outfit recommendation dataset.")

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

    colour = st.text_input(
        "Colour",
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

    warmth_level = st.slider(
        "Warmth level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = very light, 5 = very warm",
    )

    waterproof = st.selectbox(
        "Waterproof",
        [
            "No",
            "Light",
            "Yes",
            "Unknown",
        ],
    )

    layer_type = st.selectbox(
        "Layer type",
        [
            "Base layer",
            "Mid layer",
            "Outer layer",
            "Not a layer",
        ],
    )

    style = st.selectbox(
        "Style",
        [
            "Casual",
            "Streetwear",
            "Formal",
            "Sporty",
            "Minimal",
            "Other",
        ],
    )

    submitted = st.form_submit_button("Add to wardrobe")

if submitted:
    if uploaded_file is None:
        st.error("Please upload a clothing photo.")
    elif not name.strip():
        st.error("Please enter an item name.")
    elif not colour.strip():
        st.error("Please enter a colour.")
    else:
        try:
            save_clothing_item(
                uploaded_file=uploaded_file,
                name=name,
                category=category,
                colour=colour,
                season=season,
                occasion=occasion,
                warmth_level=warmth_level,
                waterproof=waterproof,
                layer_type=layer_type,
                style=style,
            )
            st.success("The clothing item was added successfully.")
        except ValueError as error:
            st.error(str(error))
        except OSError as error:
            st.error(f"The item could not be saved: {error}")

st.divider()
st.header("Outfit Recommender")

st.write("Choose the situation, and the app will recommend clothes from your wardrobe.")

recommender_wardrobe = load_wardrobe()

if recommender_wardrobe.empty:
    st.info("Upload some clothes before using the outfit recommender.")
else:
    weather = st.selectbox(
        "Weather",
        [
            "Cold",
            "Mild",
            "Hot",
        ],
    )

    is_rainy = st.checkbox("Is it rainy?")

    target_occasion = st.selectbox(
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

    def pick_best_item(
        wardrobe: pd.DataFrame,
        categories: list[str],
        weather: str,
        is_rainy: bool,
        target_occasion: str,
    ):
        """Pick the best clothing item using simple rule-based scoring."""

        candidates = wardrobe[wardrobe["category"].isin(categories)].copy()

        if candidates.empty:
            return None

        candidates["warmth_level"] = pd.to_numeric(
            candidates["warmth_level"],
            errors="coerce",
        ).fillna(3)

        candidates["score"] = 0

        # Occasion match
        candidates.loc[
            candidates["occasion"] == target_occasion,
            "score",
        ] += 4

        candidates.loc[
            candidates["occasion"] == "Casual",
            "score",
        ] += 1

        # Weather warmth rules
        if weather == "Cold":
            candidates["score"] += candidates["warmth_level"] * 2
            good_seasons = ["Winter", "Fall", "All seasons"]
        elif weather == "Hot":
            candidates["score"] += (6 - candidates["warmth_level"]) * 2
            good_seasons = ["Summer", "Spring", "All seasons"]
        else:
            candidates["score"] += (5 - (candidates["warmth_level"] - 3).abs()) * 2
            good_seasons = ["Spring", "Fall", "All seasons"]

        # Season match
        candidates.loc[
            candidates["season"].isin(good_seasons),
            "score",
        ] += 2

        # Rain rules
        if is_rainy:
            candidates.loc[
                candidates["waterproof"] == "Yes",
                "score",
            ] += 4

            candidates.loc[
                candidates["waterproof"] == "Light",
                "score",
            ] += 2

        best_item_index = candidates["score"].idxmax()

        return candidates.loc[best_item_index]

    def show_recommended_item(label: str, item):
        """Display one recommended clothing item."""

        st.subheader(label)

        image_column, information_column = st.columns([1, 2])

        with image_column:
            image_path = resolve_image_path(item["image_path"])

            if image_path is not None and image_path.exists():
                st.image(str(image_path), width=220)
            else:
                st.warning("Image unavailable.")

        with information_column:
            st.write(f"**Name:** {item['name']}")
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Colour:** {item['colour']}")
            st.write(f"**Season:** {item['season']}")
            st.write(f"**Occasion:** {item['occasion']}")
            st.write(f"**Warmth level:** {item['warmth_level']}/5")
            st.write(f"**Waterproof:** {item['waterproof']}")
            st.write(f"**Style:** {item['style']}")

    if st.button("Recommend outfit"):
        recommended_top = pick_best_item(
            recommender_wardrobe,
            ["Top"],
            weather,
            is_rainy,
            target_occasion,
        )

        recommended_bottom = pick_best_item(
            recommender_wardrobe,
            ["Bottom", "Dress"],
            weather,
            is_rainy,
            target_occasion,
        )

        recommended_shoes = pick_best_item(
            recommender_wardrobe,
            ["Shoes"],
            weather,
            is_rainy,
            target_occasion,
        )

        recommended_jacket = None

        if weather == "Cold" or is_rainy:
            recommended_jacket = pick_best_item(
                recommender_wardrobe,
                ["Jacket"],
                weather,
                is_rainy,
                target_occasion,
            )

        st.subheader("Recommended Outfit")

        if recommended_top is not None:
            show_recommended_item("Top", recommended_top)
            st.divider()
        else:
            st.warning("No top found in your wardrobe.")

        if recommended_bottom is not None:
            show_recommended_item("Bottom / Dress", recommended_bottom)
            st.divider()
        else:
            st.warning("No bottom or dress found in your wardrobe.")

        if recommended_shoes is not None:
            show_recommended_item("Shoes", recommended_shoes)
            st.divider()
        else:
            st.warning("No shoes found in your wardrobe.")

        if weather == "Cold" or is_rainy:
            if recommended_jacket is not None:
                show_recommended_item("Jacket", recommended_jacket)
                st.divider()
            else:
                st.warning("No jacket found, but the weather suggests you may need one.")
                
st.divider()
st.header("My Wardrobe")

wardrobe = load_wardrobe()

if wardrobe.empty:
    st.info("Your wardrobe is currently empty.")
else:
    st.write(f"Total items saved: **{len(wardrobe)}**")

    st.subheader("Filter wardrobe")

    def filter_options(column_name: str) -> list[str]:
        return ["All"] + sorted(
            wardrobe[column_name].dropna().astype(str).unique().tolist()
        )

    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

    with filter_col1:
        selected_category = st.selectbox(
            "Category",
            filter_options("category"),
        )

    with filter_col2:
        selected_season = st.selectbox(
            "Season",
            filter_options("season"),
        )

    with filter_col3:
        selected_occasion = st.selectbox(
            "Occasion",
            filter_options("occasion"),
        )

    with filter_col4:
        selected_waterproof = st.selectbox(
            "Waterproof",
            filter_options("waterproof"),
        )

    with filter_col5:
        selected_style = st.selectbox(
            "Style",
            filter_options("style"),
        )

    minimum_warmth = st.slider(
        "Minimum warmth level",
        min_value=1,
        max_value=5,
        value=1,
    )

    filtered_wardrobe = wardrobe.copy()

    if selected_category != "All":
        filtered_wardrobe = filtered_wardrobe[
            filtered_wardrobe["category"] == selected_category
        ]

    if selected_season != "All":
        filtered_wardrobe = filtered_wardrobe[
            filtered_wardrobe["season"] == selected_season
        ]

    if selected_occasion != "All":
        filtered_wardrobe = filtered_wardrobe[
            filtered_wardrobe["occasion"] == selected_occasion
        ]

    if selected_waterproof != "All":
        filtered_wardrobe = filtered_wardrobe[
            filtered_wardrobe["waterproof"] == selected_waterproof
        ]

    if selected_style != "All":
        filtered_wardrobe = filtered_wardrobe[
            filtered_wardrobe["style"] == selected_style
        ]

    filtered_wardrobe["warmth_level"] = pd.to_numeric(
        filtered_wardrobe["warmth_level"],
        errors="coerce",
    )

    filtered_wardrobe = filtered_wardrobe[
        filtered_wardrobe["warmth_level"] >= minimum_warmth
    ]

    st.write(f"Showing **{len(filtered_wardrobe)}** item(s).")

    if filtered_wardrobe.empty:
        st.info("No clothing items match these filters.")
    else:
        for _, item in filtered_wardrobe.iloc[::-1].iterrows():
            image_column, information_column = st.columns([1, 2])

            with image_column:
                image_path = resolve_image_path(item["image_path"])

                if image_path is not None and image_path.exists():
                    st.image(str(image_path), width=220)
                else:
                    st.warning("Image unavailable.")

            with information_column:
                st.subheader(item["name"])
                st.write(f"**Category:** {item['category']}")
                st.write(f"**Colour:** {item['colour']}")
                st.write(f"**Season:** {item['season']}")
                st.write(f"**Occasion:** {item['occasion']}")
                st.write(f"**Warmth level:** {item['warmth_level']}/5")
                st.write(f"**Waterproof:** {item['waterproof']}")
                st.write(f"**Layer type:** {item['layer_type']}")
                st.write(f"**Style:** {item['style']}")

            st.divider()