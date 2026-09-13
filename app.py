"""
STEP 3 — serving API. Deploy this whole folder as a Hugging Face Space
(free hosting, Docker SDK).

POST /detect  {"name": "...", "notes": "..."}  ->  {"allergens": [...]}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib

ALLERGEN_COLUMNS = ["milk", "eggs", "fish", "shellfish", "tree_nuts", "peanuts", "wheat_gluten", "soy", "sesame"]
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
THRESHOLD = 0.5  # only report a label if the model is at least this confident

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

app = FastAPI()


class DetectRequest(BaseModel):
    name: str
    notes: Optional[str] = None


@app.post("/detect")
def detect(req: DetectRequest):
    text = f"{req.name}. {req.notes or ''}".strip()
    X = vectorizer.transform([text])

    # predict_proba on a OneVsRestClassifier returns one probability
    # column per label, in the same order as ALLERGEN_COLUMNS.
    probs = model.predict_proba(X)[0]

    allergens = [
        DISPLAY_NAMES[col]
        for col, p in zip(ALLERGEN_COLUMNS, probs)
        if p >= THRESHOLD
    ]
    return {"allergens": allergens}


@app.get("/")
def health():
    return {"status": "ok"}
