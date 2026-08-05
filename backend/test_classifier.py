from pathlib import Path

from ml_classifier import classify_clothing_attributes


BACKEND_FOLDER = Path(__file__).resolve().parent
UPLOADS_FOLDER = BACKEND_FOLDER / "uploads"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    image_files = sorted(
        path
        for path in UPLOADS_FOLDER.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(
            f"No images were found in: {UPLOADS_FOLDER}"
        )

    for image_path in image_files:
        print("\n" + "=" * 60)
        print(f"Testing image: {image_path.name}")

        try:
            attributes = classify_clothing_attributes(image_path)

            print(
                "Category: "
                f"{attributes['category']['label']} "
                f"({attributes['category']['confidence_percent']}%)"
            )

            print(
                "Colour: "
                f"{attributes['colour']['label']} "
                f"({attributes['colour']['confidence_percent']}%)"
            )

            print(
                "Layer type: "
                f"{attributes['layer_type']['label']} "
                f"({attributes['layer_type']['confidence_percent']}%)"
            )

            print(
                "Style: "
                f"{attributes['style']['label']} "
                f"({attributes['style']['confidence_percent']}%)"
            )

        except Exception as error:
            print(f"Could not classify image: {error}")


if __name__ == "__main__":
    main()