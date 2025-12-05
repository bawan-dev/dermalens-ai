from src.preprocessing import split_ingredients

# Ingredients known to trigger fungal acne strongly (you can expand this list!)
UNSAFE_KEYWORDS = [
    "lauric acid",
    "myristic acid",
    "stearic acid",
    "oleic acid",
    "isopropyl myristate",
    "cetyl alcohol",
    "glyceryl stearate",
    "polysorbate",
    "sorbitan",
]

# Mild-risk ingredients
NEUTRAL_RISK = [
    "dimethicone",
    "caprylic/capric triglyceride",
    "fragrance"
]


def calculate_safety_score(ingredients_text: str):
    """
    Returns a fungal acne safety score from 0 to 10.
    Based purely on ingredient-level analysis.
    Model prediction is separate.
    """

    ingredients = [i.lower() for i in split_ingredients(ingredients_text)]

    score = 10  # start perfect

    # strong negative ingredients
    for bad in UNSAFE_KEYWORDS:
        if any(bad in ing for ing in ingredients):
            score -= 4  # strong penalty

    # mild-risk ingredients
    for mid in NEUTRAL_RISK:
        if any(mid in ing for ing in ingredients):
            score -= 1

    # keep score within bounds
    score = max(0, min(10, score))

    return score
