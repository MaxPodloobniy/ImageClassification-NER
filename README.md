# Named Entity Recognition + Image Classification

## Overview
This project is a full machine learning pipeline that combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** to verify whether a given text description correctly identifies the animal in an image. The system extracts animal names from a sentence using a Named Entity Recognition (NER) model and classifies animals in images using a deep learning model. Finally, it compares the results and determines whether the statement is **true or false**.

## Dataset
I built the dataset from scratch, scraping images from **Pinterest** and **Google Images**, manually filtering noise and irrelevant content. The dataset consists of **10 animal classes**, with approximately **900 images per class**.

## Models
For NER, I fine-tuned the **spaCy en_core_web_sm** model, adding a custom entity type **ANIMALS** and training it on a curated dataset of animal-related sentences. For image classification, I used a **VGG16** model as a backbone, adding two dense layers (**1024 & 512 neurons**) and a final layer with **10 neurons (softmax activation)**.

## Project Structure
The core logic is structured as follows:
- **models/**
  - **ner/**: Contains `train_ner.py`, `infer_ner.py`, and the trained model `custom_ner_model/`.
  - **classifier/**: Contains `train_classifier.py`, `infer_classifier.py`, and the best model `best_classifier_model.keras`.
- **pipeline.py**: The main script that ties everything together.
- **web-scraper.py**: The script used to collect images for training.

## Running the Pipeline
To check if a statement matches the image, run the following command:
```bash
python pipeline.py 'I see a tiger in this picture.' 'eval_dataset/tiger_0.jpeg'
```
### Example Output:
```
Text: I see a tiger in this picture.
Extracted animals: tiger
Image classification: tiger
✅ The statement is TRUE!
```
If the predicted class does not match the extracted entity, the output will indicate **FALSE**.

## Additional Notes
- The NER model is transformer-based but **not a large language model (LLM)**.
- The classifier is based on a fine-tuned **VGG16** model.
- The pipeline is fully automated and expects two inputs: **text** and **image path**.

This project showcases the integration of NLP and CV to solve a multi-modal problem. It can be extended with more sophisticated models and additional robustness improvements.

