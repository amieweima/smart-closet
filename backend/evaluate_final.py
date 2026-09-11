from pathlib import Path
from collections import defaultdict

from ml_classifier import classify_image


EVAL_DIR = Path(__file__).parent / "eval_final"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def main():
    total = 0
    correct = 0

    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    mistakes = []

    print("=" * 50)
    print("SMART CLOSET FINAL HELD-OUT EVALUATION")
    print("=" * 50)

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
            class_total[actual_label] += 1

            if is_correct:
                correct += 1
                class_correct[actual_label] += 1
            else:
                mistakes.append(
                    {
                        "image": image_path.name,
                        "actual": actual_label,
                        "predicted": predicted_label,
                        "confidence": confidence,
                    }
                )

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

            if is_correct:
                print("Result: CORRECT")
            else:
                print("Result: INCORRECT")

    print()
    print("=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)

    if total == 0:
        print("No images found.")
        return

    accuracy = correct / total

    print(f"Images evaluated: {total}")
    print(f"Correct: {correct}/{total}")
    print(f"Top-1 accuracy: {accuracy:.1%}")

    print()
    print("=" * 50)
    print("PER-CLASS ACCURACY")
    print("=" * 50)

    for label in sorted(class_total):
        class_accuracy = (
            class_correct[label] / class_total[label]
        )

        print(
            f"{label:20} "
            f"{class_correct[label]}/{class_total[label]} "
            f"({class_accuracy:.1%})"
        )

    print()
    print("=" * 50)
    print("MISCLASSIFICATIONS")
    print("=" * 50)

    if not mistakes:
        print("None")
    else:
        for mistake in mistakes:
            print(
                f"{mistake['image']}: "
                f"{mistake['actual']} -> "
                f"{mistake['predicted']} "
                f"({mistake['confidence']}%)"
            )

    print()
    print("=" * 50)

    if total != 48:
        print(
            f"WARNING: Expected 48 images, "
            f"but found {total}."
        )

    print("FINAL EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()