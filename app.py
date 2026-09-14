"""
STEP 3 — Serving API.

The API receives ingredient information and detects allergens.

POST /detect

{
    "ingredients": [
        "wheat flour",
        "sugar",
        "peanut butter",
        "milk powder"
    ]
}

Response:

{
    "allergens": [
        "Milk",
        "Peanuts",
        "Wheat/Gluten"
    ]
}

The system uses:
    1. Explicit ingredient matching
    2. ML-based prediction as supporting evidence

The product name itself is not used for allergen detection.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import re


ALLERGEN_COLUMNS = [
    "milk",
    "eggs",
    "fish",
    "shellfish",
    "tree_nuts",
    "peanuts",
    "wheat_gluten",
    "soy",
    "sesame",
]


DISPLAY_NAMES = {
    "milk": "Milk",
    "eggs": "Eggs",
    "fish": "Fish",
    "shellfish": "Shellfish",
    "tree_nuts": "Tree Nuts",
    "peanuts": "Peanuts",
    "wheat_gluten": "Wheat/Gluten",
    "soy": "Soy",
    "sesame": "Sesame",
}


# ML threshold.
#
# The previous 0.50 threshold produced false positives,
# such as detecting Soy when the ingredients did not contain soy.
#
# A higher threshold makes the ML layer more conservative.
ML_THRESHOLD = 0.75


# Explicit ingredient keywords.
#
# These directly identify obvious allergen ingredients.
EXPLICIT_ALLERGENS = {
    "milk": [
        "milk",
        "dairy",
        "whey",
        "casein",
        "caseinate",
        "milk powder",
        "skim milk",
        "whole milk",
        "condensed milk",
        "evaporated milk",
    ],

    "eggs": [
        "egg",
        "eggs",
        "egg white",
        "egg yolk",
        "albumin",
        "albumen",
    ],

    "fish": [
        "fish",
        "tuna",
        "salmon",
        "anchovy",
        "sardine",
        "cod",
        "mackerel",
    ],

    "shellfish": [
        "shellfish",
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "oyster",
        "scallop",
    ],

    "tree_nuts": [
        "almond",
        "almonds",
        "cashew",
        "cashews",
        "walnut",
        "walnuts",
        "hazelnut",
        "hazelnuts",
        "pistachio",
        "pistachios",
        "pecan",
        "pecans",
    ],

    "peanuts": [
        "peanut",
        "peanuts",
        "peanut butter",
    ],

    "wheat_gluten": [
        "wheat",
        "wheat flour",
        "gluten",
        "bread",
        "pasta",
    ],

    "soy": [
        "soy",
        "soya",
        "soybean",
        "soybeans",
        "soy sauce",
        "soy protein",
    ],

    "sesame": [
        "sesame",
        "sesame seeds",
        "tahini",
    ],
}


vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")


app = FastAPI()


class DetectRequest(BaseModel):
    ingredients: List[str]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_explicit_allergens(ingredients: List[str]):
    detected = set()

    for ingredient in ingredients:
        text = normalize_text(ingredient)

        for allergen, keywords in EXPLICIT_ALLERGENS.items():
            for keyword in keywords:
                if keyword in text:
                    detected.add(allergen)
                    break

    return detected


@app.post("/detect")
def detect(req: DetectRequest):
    # Remove empty ingredients.
    ingredients = [
        ingredient.strip()
        for ingredient in req.ingredients
        if ingredient.strip()
    ]

    if not ingredients:
        return {
            "allergens": []
        }

    # Combine all ingredients for the ML model.
    text = ". ".join(ingredients)

    X = vectorizer.transform([text])

    probabilities = model.predict_proba(X)[0]

    # ---------------------------------------------------------
    # STEP 1 — Explicit ingredient detection
    # ---------------------------------------------------------
    #
    # If an ingredient directly contains an allergen keyword,
    # report it immediately.
    #
    # Example:
    # "peanut butter" -> Peanuts
    # "milk powder"  -> Milk
    # "wheat flour"  -> Wheat/Gluten
    #
    detected = detect_explicit_allergens(ingredients)

    # ---------------------------------------------------------
    # STEP 2 — ML supporting detection
    # ---------------------------------------------------------
    #
    # Only add an ML prediction when the probability reaches
    # the higher 0.75 threshold.
    #
    # This prevents weaker predictions from creating too many
    # false positives.
    #
    for allergen, probability in zip(
        ALLERGEN_COLUMNS,
        probabilities,
    ):
        if probability >= ML_THRESHOLD:
            detected.add(allergen)

    # ---------------------------------------------------------
    # STEP 3 — Consistent output order
    # ---------------------------------------------------------

    allergens = [
        DISPLAY_NAMES[allergen]
        for allergen in ALLERGEN_COLUMNS
        if allergen in detected
    ]

    return {
        "allergens": allergens
    }


@app.get("/")
def health():
    return {
        "status": "ok"
    }