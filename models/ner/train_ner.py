import spacy
import random
from spacy.training.example import Example

# Завантажуємо базову модель
nlp = spacy.load("en_core_web_sm")

# Додаємо NER
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Додаємо кастомні сутності (тварини)
animals = ["chimpanzee", "coyote", "deer", "duck", "eagle", "elephant", "hedgehog", "hippopotamus", "kangaroo", "rhinocerus", "tiger"]
for animal in animals:
    ner.add_label(animal.upper())

TRAIN_DATA = [
    ("I saw a chimpanzee climbing a tree.", {"entities": [(7, 17, "CHIMPANZEE")]}),
    ("The chimpanzee was eating bananas.", {"entities": [(4, 14, "CHIMPANZEE")]}),
    ("Chimpanzees are intelligent animals.", {"entities": [(0, 10, "CHIMPANZEE")]}),
    ("A group of chimpanzees was making loud noises.", {"entities": [(10, 21, "CHIMPANZEE")]}),
    ("I watched a documentary about chimpanzees.", {"entities": [(30, 41, "CHIMPANZEE")]}),
    ("A chimpanzee can use tools to get food.", {"entities": [(2, 12, "CHIMPANZEE")]}),
    ("We watched a playful chimpanzee at the zoo.", {"entities": [(19, 29, "CHIMPANZEE")]}),
    ("Chimpanzees are intelligent animals.", {"entities": [(0, 10, "CHIMPANZEE")]}),

    ("A coyote ran across the road.", {"entities": [(2, 8, "COYOTE")]}),
    ("I heard a coyote howling at night.", {"entities": [(9, 15, "COYOTE")]}),
    ("Coyotes live in the wild and hunt for food.", {"entities": [(0, 7, "COYOTE")]}),
    ("The coyote was looking for food.", {"entities": [(4, 10, "COYOTE")]}),
    ("Have you ever seen a coyote in real life?", {"entities": [(21, 27, "COYOTE")]}),
    ("A coyote was hunting in the desert.", {"entities": [(2, 8, "COYOTE")]}),
    ("I heard a coyote howling at night.", {"entities": [(10, 16, "COYOTE")]}),
    ("Coyotes are found in North America.", {"entities": [(0, 7, "COYOTE")]}),
    ("A coyote ran across the road.", {"entities": [(2, 8, "COYOTE")]}),
    ("Coyotes adapt well to urban areas.", {"entities": [(0, 7, "COYOTE")]}),

    ("A deer jumped over the fence.", {"entities": [(2, 6, "DEER")]}),
    ("The deer was grazing in the field.", {"entities": [(4, 8, "DEER")]}),
    ("I spotted a group of deer in the forest.", {"entities": [(19, 23, "DEER")]}),
    ("Deer are common in this area.", {"entities": [(0, 4, "DEER")]}),
    ("A baby deer is called a fawn.", {"entities": [(6, 10, "DEER")]}),
    ("The deer was grazing near the river.", {"entities": [(4, 8, "DEER")]}),
    ("I saw a deer in the forest.", {"entities": [(9, 13, "DEER")]}),
    ("Deer are common in this region.", {"entities": [(0, 4, "DEER")]}),
    ("A deer crossed the road suddenly.", {"entities": [(2, 6, "DEER")]}),
    ("The deer had large antlers.", {"entities": [(4, 8, "DEER")]}),

    ("I saw a duck swimming in the lake.", {"entities": [(7, 11, "DUCK")]}),
    ("The duck quacked loudly.", {"entities": [(4, 8, "DUCK")]}),
    ("Ducks are often found in ponds.", {"entities": [(0, 5, "DUCK")]}),
    ("A mother duck protects her ducklings.", {"entities": [(8, 12, "DUCK")]}),
    ("Have you ever fed ducks at the park?", {"entities": [(19, 24, "DUCK")]}),
    ("A duck was swimming in the pond.", {"entities": [(2, 6, "DUCK")]}),
    ("The duck quacked loudly.", {"entities": [(4, 8, "DUCK")]}),
    ("I saw a duck at the lake.", {"entities": [(9, 13, "DUCK")]}),
    ("Ducks migrate during the winter.", {"entities": [(0, 5, "DUCK")]}),
    ("A mother duck led her ducklings.", {"entities": [(2, 6, "DUCK")]}),

    ("An eagle soared high in the sky.", {"entities": [(3, 8, "EAGLE")]}),
    ("The eagle caught a fish.", {"entities": [(4, 9, "EAGLE")]}),
    ("Eagles have excellent vision.", {"entities": [(0, 6, "EAGLE")]}),
    ("I saw a golden eagle in the mountains.", {"entities": [(10, 15, "EAGLE")]}),
    ("Eagles build nests on tall cliffs.", {"entities": [(0, 6, "EAGLE")]}),
    ("The eagle soared above the mountains.", {"entities": [(4, 9, "EAGLE")]}),
    ("I saw an eagle catching a fish.", {"entities": [(9, 14, "EAGLE")]}),
    ("Eagles have excellent vision.", {"entities": [(0, 6, "EAGLE")]}),
    ("An eagle built a nest on the cliff.", {"entities": [(3, 8, "EAGLE")]}),
    ("The eagle spread its wings wide.", {"entities": [(4, 9, "EAGLE")]}),

    ("An elephant was drinking water.", {"entities": [(3, 11, "ELEPHANT")]}),
    ("Elephants are the largest land animals.", {"entities": [(0, 9, "ELEPHANT")]}),
    ("I saw a herd of elephants in the zoo.", {"entities": [(14, 23, "ELEPHANT")]}),
    ("The elephant used its trunk to grab food.", {"entities": [(4, 12, "ELEPHANT")]}),
    ("Elephants have strong memories.", {"entities": [(0, 9, "ELEPHANT")]}),
    ("The elephant sprayed water with its trunk.", {"entities": [(4, 12, "ELEPHANT")]}),
    ("I saw a baby elephant at the zoo.", {"entities": [(9, 17, "ELEPHANT")]}),
    ("Elephants have strong memories.", {"entities": [(0, 9, "ELEPHANT")]}),
    ("An elephant walked through the jungle.", {"entities": [(3, 11, "ELEPHANT")]}),
    ("The elephant flapped its ears.", {"entities": [(4, 12, "ELEPHANT")]}),

    ("A hedgehog curled into a ball.", {"entities": [(2, 10, "HEDGEHOG")]}),
    ("I saw a hedgehog in my backyard.", {"entities": [(7, 15, "HEDGEHOG")]}),
    ("Hedgehogs have spiky fur.", {"entities": [(0, 9, "HEDGEHOG")]}),
    ("The hedgehog was hiding under the leaves.", {"entities": [(4, 12, "HEDGEHOG")]}),
    ("Hedgehogs are nocturnal animals.", {"entities": [(0, 9, "HEDGEHOG")]}),
    ("A hedgehog rolled into a ball.", {"entities": [(2, 10, "HEDGEHOG")]}),
    ("The hedgehog has sharp spines.", {"entities": [(4, 12, "HEDGEHOG")]}),
    ("Hedgehogs are nocturnal animals.", {"entities": [(0, 9, "HEDGEHOG")]}),
    ("I saw a tiny hedgehog in the garden.", {"entities": [(9, 17, "HEDGEHOG")]}),
    ("The hedgehog was hiding under the leaves.", {"entities": [(4, 12, "HEDGEHOG")]}),

    ("A hippopotamus was resting in the water.", {"entities": [(2, 13, "HIPPOPOTAMUS")]}),
    ("Hippopotamuses spend most of their time in water.", {"entities": [(0, 13, "HIPPOPOTAMUS")]}),
    ("I saw a baby hippopotamus at the zoo.", {"entities": [(10, 21, "HIPPOPOTAMUS")]}),
    ("Hippopotamuses have powerful jaws.", {"entities": [(0, 13, "HIPPOPOTAMUS")]}),
    ("A hippopotamus can run surprisingly fast.", {"entities": [(2, 13, "HIPPOPOTAMUS")]}),
    ("A hippopotamus was bathing in the river.", {"entities": [(2, 14, "HIPPOPOTAMUS")]}),
    ("The hippopotamus is a large mammal.", {"entities": [(4, 16, "HIPPOPOTAMUS")]}),
    ("Hippopotamuses spend most of their time in water.", {"entities": [(0, 13, "HIPPOPOTAMUS")]}),
    ("I saw a baby hippopotamus at the zoo.", {"entities": [(9, 21, "HIPPOPOTAMUS")]}),
    ("The hippopotamus yawned widely.", {"entities": [(4, 16, "HIPPOPOTAMUS")]}),

    ("A kangaroo hopped across the field.", {"entities": [(2, 10, "KANGAROO")]}),
    ("Kangaroos carry their babies in pouches.", {"entities": [(0, 8, "KANGAROO")]}),
    ("I saw a kangaroo at the wildlife park.", {"entities": [(7, 15, "KANGAROO")]}),
    ("The kangaroo was eating grass.", {"entities": [(4, 12, "KANGAROO")]}),
    ("Kangaroos are native to Australia.", {"entities": [(0, 8, "KANGAROO")]}),
    ("A kangaroo hopped across the field.", {"entities": [(2, 10, "KANGAROO")]}),
    ("The kangaroo carried a baby in its pouch.", {"entities": [(4, 12, "KANGAROO")]}),
    ("Kangaroos are strong jumpers.", {"entities": [(0, 8, "KANGAROO")]}),
    ("I saw a kangaroo boxing with another one.", {"entities": [(9, 17, "KANGAROO")]}),
    ("The kangaroo was resting in the shade.", {"entities": [(4, 12, "KANGAROO")]}),

    ("A rhinoceros was drinking water.", {"entities": [(2, 11, "RHINOCERUS")]}),
    ("Rhinoceroses have thick skin.", {"entities": [(0, 11, "RHINOCERUS")]}),
    ("I saw a white rhinoceros at the zoo.", {"entities": [(10, 19, "RHINOCERUS")]}),
    ("Rhinoceroses use their horns for defense.", {"entities": [(0, 11, "RHINOCERUS")]}),
    ("A baby rhinoceros stays close to its mother.", {"entities": [(2, 11, "RHINOCERUS")]}),
    ("A rhinoceros has a thick skin.", {"entities": [(2, 11, "RHINOCEROS")]}),
    ("The rhinoceros was drinking water.", {"entities": [(4, 13, "RHINOCEROS")]}),
    ("Rhinoceroses have a strong horn.", {"entities": [(0, 11, "RHINOCEROS")]}),
    ("I saw a huge rhinoceros at the safari.", {"entities": [(9, 18, "RHINOCEROS")]}),
    ("The rhinoceros charged at the jeep.", {"entities": [(4, 13, "RHINOCEROS")]}),

    ("A tiger was stalking its prey.", {"entities": [(2, 7, "TIGER")]}),
    ("Tigers are powerful hunters.", {"entities": [(0, 6, "TIGER")]}),
    ("I saw a Bengal tiger at the zoo.", {"entities": [(10, 15, "TIGER")]}),
    ("The tiger roared loudly.", {"entities": [(4, 9, "TIGER")]}),
    ("Tigers have distinctive stripes.", {"entities": [(0, 6, "TIGER")]}),
    ("A tiger was stalking its prey.", {"entities": [(2, 7, "TIGER")]}),
    ("The tiger roared loudly in the jungle.", {"entities": [(4, 9, "TIGER")]}),
    ("Tigers are excellent hunters.", {"entities": [(0, 6, "TIGER")]}),
    ("I saw a white tiger at the zoo.", {"entities": [(9, 14, "TIGER")]}),
    ("The tiger pounced on its prey.", {"entities": [(4, 9, "TIGER")]}),
]

# Налаштовуємо пайплайн
optimizer = nlp.begin_training()

# Ініціалізуємо змінні для відстеження найкращої моделі
best_loss = float('inf')
best_model_path = "best_ner_model"

# Тренуємо модель
for i in range(20):
    random.shuffle(TRAIN_DATA)
    losses = {}

    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.3, losses=losses)

    # Обчислюємо загальну втрату для цієї ітерації
    current_loss = sum(losses.values())

    # Перевіряємо, чи це найкраща модель
    if current_loss < best_loss:
        print(f"New best model found! Previous loss: {best_loss}, New loss: {current_loss}")
        best_loss = current_loss

        # Просто зберігаємо нову модель (SpaCy перезапише стару)
        nlp.to_disk(best_model_path)
        print(f"Model saved to {best_model_path}")

print(f"\nTraining completed. Best model saved with loss: {best_loss}")

# Завантаження найкращої моделі для тестування
print("\nLoading best model for testing...")
best_nlp = spacy.load(best_model_path)

# Тестові дані
TEST_DATA = [
    ("There is a chimpanzee in this image."),
    ("I can see a chimpanzee in the photo."),
    ("This picture contains a chimpanzee."),

    ("There is a coyote in this picture."),
    ("I found a coyote in this image."),
    ("This photo clearly shows a coyote."),

    ("A deer is visible in this picture."),
    ("I can spot a deer in the photo."),
    ("This image includes a deer."),

    ("There is a duck in the picture."),
    ("You can see a duck in this image."),
    ("A duck is present in this photograph."),

    ("An eagle appears in this picture."),
    ("This photo has an eagle in it."),
    ("There is an eagle in the image."),

    ("You can see an elephant in this picture."),
    ("An elephant is present in the photo."),
    ("There is an elephant in this image."),

    ("A hedgehog is visible in this picture."),
    ("This photo has a hedgehog in it."),
    ("I can see a hedgehog in this image."),

    ("There is a hippopotamus in this picture."),
    ("A hippopotamus is clearly visible in this image."),
    ("This photo contains a hippopotamus."),

    ("You can find a kangaroo in this picture."),
    ("This image includes a kangaroo."),
    ("A kangaroo appears in this photo."),

    ("There is a rhinoceros in this image."),
    ("A rhinoceros is clearly visible in this picture."),
    ("This photo contains a rhinoceros."),

    ("I see a tiger in this image."),
    ("This picture has a tiger in it."),
    ("A tiger is present in this photo."),
]

# Тестуємо модель
for sentence in TEST_DATA:
    doc = nlp(sentence)
    print(f'\n{sentence}')

    for ent in doc.ents:
        print(f"Знайдено: {ent.text} (категорія: {ent.label_})")
