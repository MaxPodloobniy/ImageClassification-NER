import spacy
import argparse

# Шлях до кастомної моделі
CUSTOM_MODEL_PATH = "custom_ner_model"

# Список тварин, які нам потрібні
animals = {"chimpanzee", "coyote", "deer", "duck", "eagle", "elephant", "hedgehog",
           "hippopotamus", "kangaroo", "rhinocerus", "tiger"}


def extract_animals(text, use_custom_model=False):
    """Функція для отримання тварин з тексту."""

    # Вибір моделі
    if use_custom_model and spacy.util.is_package(CUSTOM_MODEL_PATH):
        print("Using custom NER model.")
        nlp = spacy.load(CUSTOM_MODEL_PATH)
    else:
        print("Using default SpaCy NER model.")
        nlp = spacy.load("en_core_web_sm")

    # Обробка тексту
    doc = nlp(text)
    found_animals = [ent.text for ent in doc.ents if ent.text.lower() in animals]

    return found_animals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER for animal extraction.")
    parser.add_argument("text", type=str, help="Input text for NER")
    parser.add_argument("--use_custom", action="store_true", help="Use custom trained model")
    args = parser.parse_args()

    animals_found = extract_animals(args.text, args.use_custom)
    print("Found animals:", animals_found)
