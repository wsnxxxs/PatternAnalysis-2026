"""
Data loading and preprocessing for radiology report lay summarisation.

Dataset: BioLaySumm 2025 LaymanRRG open-source track (Hugging Face).
Each example pairs a chest X-ray report with a lay summary written for patients.

Main steps:
1. Load the official train, validation and test splits.
2. Remove repeated reports from the training set.
3. Remove validation reports that also appear in training (data leakage),
   then remove repeated reports inside the validation set.

Run `python dataset.py` to print the size of each split.
"""

from datasets import load_dataset

DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"


def normalise_report(text):
    """Lowercase the text and squeeze repeated spaces so that copies of a report match."""
    words = text.lower().split()
    return " ".join(words)


def remove_duplicates(examples):
    """Keep the first example of each report and return a list of (report, summary) pairs."""
    pairs = []
    seen_reports = set()
    for example in examples:
        key = normalise_report(example["radiology_report"])
        if key not in seen_reports:
            seen_reports.add(key)
            pairs.append((example["radiology_report"], example["layman_report"]))
    return pairs


def remove_leaked_reports(validation_examples, train_pairs):
    """Drop validation examples whose report also appears in the training set."""
    train_reports = set()
    for report, summary in train_pairs:
        train_reports.add(normalise_report(report))

    kept_examples = []
    for example in validation_examples:
        if normalise_report(example["radiology_report"]) not in train_reports:
            kept_examples.append(example)
    return kept_examples


def prepare_splits():
    """Load the official splits and return leakage-free train, validation and test data."""
    raw_splits = load_dataset(DATASET_NAME)

    # Training set: remove repeated reports.
    train_pairs = remove_duplicates(raw_splits["train"])

    # Validation set: remove reports seen in training, then remove repeated reports.
    validation_examples = remove_leaked_reports(raw_splits["validation"], train_pairs)
    validation_pairs = remove_duplicates(validation_examples)

    # The test set has no reference summaries, so only the reports are kept for demo inference.
    test_reports = list(raw_splits["test"]["radiology_report"])

    leaked_count = len(raw_splits["validation"]) - len(validation_examples)
    print(f"Train: {len(raw_splits['train'])} rows, {len(train_pairs)} after removing duplicates")
    print(f"Validation: {len(raw_splits['validation'])} rows, {leaked_count} leaked into train, "
          f"{len(validation_pairs)} after removing leaks and duplicates")
    print(f"Test: {len(test_reports)} reports (no reference summaries)")

    return train_pairs, validation_pairs, test_reports


if __name__ == "__main__":
    train_pairs, validation_pairs, test_reports = prepare_splits()
