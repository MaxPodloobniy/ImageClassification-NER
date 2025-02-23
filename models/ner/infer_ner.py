import spacy
import sys

# Завантажуємо модель
MODEL_PATH = "models/ner/best_ner_model/"
nlp = spacy.load(MODEL_PATH)


def extract_animal(text):
    doc = nlp(text)
    animals = [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]
    return animals


if __name__ == "__main__":
    input_text = sys.argv[1]
    animals = extract_animal(input_text)
    if animals:
        print(f"Extracted animals: {', '.join(animals)}")
    else:
        print("No animals found in text.")
