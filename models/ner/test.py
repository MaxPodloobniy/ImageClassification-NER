from spacy.training import offsets_to_biluo_tags
from spacy.lang.en import English

nlp = English()
text = "Rhinoceroses use their horns for defense."  # Твій текст
entities = [(0, 12, "ANIMAL")]  # Поточні індекси, які ти використовуєш

doc = nlp.make_doc(text)
biluo_tags = offsets_to_biluo_tags(doc, entities)
print(biluo_tags)  # Дивишся, чи правильно працює