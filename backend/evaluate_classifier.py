from pathlib import Path

import pandas as pd


FEEDBACK_FILE = Path(__file__).parent / "data" / "classification_feedback.csv"

ATTRIBUTES = {
    "Category": {
        "predicted": "predicted_category",
        "final": "final_category",
        "confidence": "category_confidence",
    },
    "Colour": {
        "predicted": "predicted_colour",
        "final": "final_colour",
        "confidence": "colour_confidence",
    },
    "Layer type": {
        "predicted": "predicted_layer_type",
        "final": "final_layer_type",
        "confidence": "layer_type_confidence",
    },
    "Style": {
        "predicted": "predicted_style",
        "final": "final_style",
        "confidence": "style_confidence",
    },
}


def normalize_label(value: object) -> str:
    """Normalize labels so capitalization and spacing do not affect accuracy."""
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def evaluate_attribute(
    feedback: pd.DataFrame,
    name: str,
    predicted_column: str,
    final_column: str,
    confidence_column: str,
) -> None:
    """Print evaluation results for one predicted clothing attribute."""

    valid_rows = feedback[
        feedback[predicted_column].notna()
        & feedback[final_column].notna()
    ].copy()

    if valid_rows.empty:
        print(f"\n{name}")
        print("-" * len(name))
        print("No feedback available.")
        return

    valid_rows["predicted_normalized"] = valid_rows[predicted_column].apply(
        normalize_label
    )
    valid_rows["final_normalized"] = valid_rows[final_column].apply(
        normalize_label
    )

    valid_rows["correct"] = (
        valid_rows["predicted_normalized"]
        == valid_rows["final_normalized"]
    )

    accuracy = valid_rows["correct"].mean()
    correct_count = int(valid_rows["correct"].sum())
    total_count = len(valid_rows)
    incorrect_count = total_count - correct_count

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Incorrect: {incorrect_count}")

    confidence_values = pd.to_numeric(
        valid_rows[confidence_column],
        errors="coerce",
    )

    correct_confidence = confidence_values[valid_rows["correct"]].dropna()
    incorrect_confidence = confidence_values[~valid_rows["correct"]].dropna()

    if not correct_confidence.empty:
        print(
            "Average confidence when correct: "
            f"{correct_confidence.mean():.3f}"
        )

    if not incorrect_confidence.empty:
        print(
            "Average confidence when incorrect: "
            f"{incorrect_confidence.mean():.3f}"
        )

    mistakes = valid_rows[~valid_rows["correct"]]

    if not mistakes.empty:
        common_mistakes = (
            mistakes.groupby([predicted_column, final_column])
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

        print("Most common mistakes:")

        for (predicted, final), count in common_mistakes.items():
            print(f"  {predicted} -> {final}: {count}")


def main() -> None:
    if not FEEDBACK_FILE.exists():
        print(f"Feedback file not found:\n{FEEDBACK_FILE}")
        return

    feedback = pd.read_csv(FEEDBACK_FILE)

    if feedback.empty:
        print("The feedback file does not contain any data rows.")
        return

    print("=" * 45)
    print("CLOTHING CLASSIFIER EVALUATION")
    print("=" * 45)
    print(f"Feedback entries: {len(feedback)}")

    for attribute_name, columns in ATTRIBUTES.items():
        evaluate_attribute(
            feedback=feedback,
            name=attribute_name,
            predicted_column=columns["predicted"],
            final_column=columns["final"],
            confidence_column=columns["confidence"],
        )

    print("\n" + "=" * 45)
    print("Evaluation complete.")
    print("=" * 45)


if __name__ == "__main__":
    main()