import re
from typing import List

def clean_ingredients_text(text: str) -> str:
    """
    Basic cleaning:
    - lowercases
    - removes extra spaces
    - strips weird characters
    """
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # replace newlines / tabs with space
    text = text.replace("\n", " ").replace("\t", " ")

    # remove anything that's not letter, number, comma, space or slash
    text = re.sub(r"[^a-z0-9,/%\s\-]", " ", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_ingredients(text: str) -> List[str]:
    """
    Splits an ingredients string like:
    "aqua, glycerin, niacinamide"
    into a list of ["aqua", "glycerin", "niacinamide"].
    """
    text = clean_ingredients_text(text)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts


def join_ingredients_for_model(text: str) -> str:
    """
    Returns cleaned text for TF-IDF model.
    """
    return clean_ingredients_text(text)
