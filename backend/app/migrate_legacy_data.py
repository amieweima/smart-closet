from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd


APP_FOLDER = Path(__file__).resolve().parent
BACKEND_FOLDER = APP_FOLDER.parent
PROJECT_ROOT = BACKEND_FOLDER.parent

DATA_FOLDER = BACKEND_FOLDER / "data"
UPLOAD_FOLDER = BACKEND_FOLDER / "uploads"
WARDROBE_FILE = DATA_FOLDER / "wardrobe.csv"

SEARCH_FOLDERS = [
    UPLOAD_FOLDER,
    PROJECT_ROOT / "uploads",
    PROJECT_ROOT / "streamlit_prototype" / "uploads",
    PROJECT_ROOT / "_migration_backup" / "root_uploads",
    PROJECT_ROOT / "frontend" / "public" / "images",
]


def find_image(image_filename: str) -> Path | None:
    for folder in SEARCH_FOLDERS:
        if not folder.exists():
            continue

        direct_match = folder / image_filename

        if direct_match.is_file():
            return direct_match

        for match in folder.rglob(image_filename):
            if match.is_file():
                return match

    return None


def main() -> None:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    if not WARDROBE_FILE.exists():
        raise FileNotFoundError(
            f"wardrobe.csv was not found at {WARDROBE_FILE}"
        )

    wardrobe = pd.read_csv(WARDROBE_FILE)

    backup_name = (
        f"wardrobe-before-react-migration-"
        f"{datetime.now():%Y%m%d-%H%M%S}.csv"
    )
    backup_path = DATA_FOLDER / backup_name
    shutil.copy2(WARDROBE_FILE, backup_path)

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

    copied_images = 0
    missing_images = []

    for index, row in wardrobe.iterrows():
        saved_path = str(row.get("image_path", "")).strip()

        if not saved_path or saved_path.lower() == "nan":
            missing_images.append(
                str(row.get("name", f"Row {index + 1}"))
            )
            continue

        image_filename = Path(saved_path).name
        destination = UPLOAD_FOLDER / image_filename

        if not destination.is_file():
            source = find_image(image_filename)

            if source is None:
                missing_images.append(
                    str(row.get("name", f"Row {index + 1}"))
                )
                continue

            shutil.copy2(source, destination)
            copied_images += 1

        wardrobe.at[index, "image_path"] = str(
            Path("backend") / "uploads" / image_filename
        )

    wardrobe.to_csv(WARDROBE_FILE, index=False)

    print(f"CSV backup: {backup_path}")
    print(f"Clothing records: {len(wardrobe)}")
    print(f"Images copied: {copied_images}")

    if missing_images:
        print("Images still missing for:")
        for item_name in missing_images:
            print(f"  - {item_name}")
    else:
        print("All clothing images were found.")


if __name__ == "__main__":
    main()
