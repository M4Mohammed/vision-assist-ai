import pytest

from app.classification.classifier import DangerClassifier

CASES = [
    ("Road with puddles", "DANGEROUS"),
    ("Road without puddles", "SAFE"),
    ("Street with holes", "DANGEROUS"),
    ("Street with no holes", "SAFE"),
    ("Clear road, free of obstacles", "SAFE"),
    ("Large fire near building", "DANGEROUS"),
    ("A knife on the table", "DANGEROUS"),
    ("There is no knife here", "SAFE"),
    ("Walking down the stairs", "DANGEROUS"),
    ("Level ground, not a step in sight", "SAFE"),
]


@pytest.fixture(scope="module")
def classifier():
    return DangerClassifier()


@pytest.mark.parametrize("caption,expected", CASES)
def test_classify(classifier, caption, expected):
    label, reason = classifier.classify(caption)
    assert label == expected, f"'{caption}' -> {label} (expected {expected}); reason: {reason}"


def test_empty_text_is_safe(classifier):
    label, _ = classifier.classify("")
    assert label == "SAFE"
