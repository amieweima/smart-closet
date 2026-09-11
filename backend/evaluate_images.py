from pathlib import Path

from ml_classifier import classify_image


EVAL_DIR = Path(__file__).parent / "eval_images"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def main():
    total = 0
    correct = 0

    print("=" * 45)
    print("SMART CLOSET IMAGE EVALUATION")
    print("=" * 45)

    for category_folder in sorted(EVAL_DIR.iterdir()):

        if not category_folder.is_dir():
            continue

        actual_label = category_folder.name.strip().lower()

        for image_path in sorted(category_folder.iterdir()):

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            predictions = classify_image(
                image_path,
                top_k=3,
            )

            predicted_label = predictions[0]["label"]
            confidence = predictions[0]["confidence_percent"]

            is_correct = predicted_label == actual_label

            total += 1

            if is_correct:
                correct += 1

            print()
            print(f"Image: {image_path.name}")
            print(f"Actual: {actual_label}")

            print("Top 3 predictions:")

            for i, prediction in enumerate(predictions, start=1):
                print(
                    f"  {i}. {prediction['label']} "
                    f"({prediction['confidence_percent']}%)"
                )

            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence}%")

            if is_correct:
                print("Result: CORRECT")
            else:
                print("Result: INCORRECT")

    print()
    print("=" * 45)

    if total > 0:
        accuracy = correct / total

        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1%}")
    else:
        print("No images found.")

    print("=" * 45)


if __name__ == "__main__":
    main()