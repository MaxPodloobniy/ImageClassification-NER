import spacy
import argparse

# Шлях до кастомної моделі
CUSTOM_MODEL_PATH = "models/ner/custom_ner_model"

# Список тварин, які нам потрібні
animals = {"chimpanzee", "coyote", "deer", "duck", "eagle", "elephant", "hedgehog", "kangaroo", "rhinocerus", "tiger"}


def extract_animals(text):
    """Функція для отримання тварин з тексту, використовуючи ТІЛЬКИ кастомну модель."""
    nlp = spacy.load(CUSTOM_MODEL_PATH)

    # Обробка тексту
    doc = nlp(text)

    # Вибираємо тільки ті, що мають мітку "ANIMAL"
    found_animals = [ent.text.lower() for ent in doc.ents if ent.label_ == "ANIMAL"]

    return found_animals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER for animal extraction.")
    parser.add_argument("text", type=str, help="Input text for NER")
    # parser.add_argument("--use_custom", action="store_true", help="Use custom trained model")
    args = parser.parse_args()

    animals_found = extract_animals(args.text)
    print("Found animals:", animals_found)
