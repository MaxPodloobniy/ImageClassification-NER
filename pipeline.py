import subprocess
import sys


def run_classifier(image_path):
    # Execute the image classification script as a subprocess
    result = subprocess.run(
        ["python", "models/classifier/infer_classifier.py", image_path],
        capture_output=True, text=True
    )

    output = result.stdout.strip()
    error_output = result.stderr.strip()

    if error_output:
        print("Classifier error:", error_output)

    # Extract the predicted class from the output
    predicted_line = next((line for line in output.split("\n") if "Predicted class:" in line), None)

    if not predicted_line:
        print("Error: Unexpected classifier output!")
        sys.exit(1)

    predicted_class = predicted_line.split(": ")[1].split(",")[0]
    return predicted_class.lower()


def run_ner(text):
    # Execute the named entity recognition (NER) script as a subprocess
    result = subprocess.run(["python", "models/ner/infer_ner.py", text], capture_output=True, text=True)
    output = result.stdout.strip()
    error_output = result.stderr.strip()

    if error_output:
        print("NER error:", error_output)

    lines = output.strip().split("\n")
    # Extract detected animal names from the last line of output
    if "Found animals:" in lines[-1]:  # Check if the last line contains recognized animals
        animals = lines[-1].split(": ")[1].strip("[]'").split(", ")
        return [animal.lower() for animal in animals]
    return []


if __name__ == "__main__":
    # Ensure the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py '<text>' <image_path>")
        sys.exit(1)

    input_text = sys.argv[1]
    image_path = sys.argv[2]

    # Extract animal names from text using NER model
    extracted_animals = run_ner(input_text)
    # Predict the class of the given image
    predicted_animal = run_classifier(image_path)

    # Display extracted information
    print(f"Text: {input_text}")
    print(f"Extracted animals: {', '.join(extracted_animals) if extracted_animals else 'None'}")
    print(f"Image classification: {predicted_animal}")

    # Check if the classified animal matches any extracted from the text
    if extracted_animals and predicted_animal in extracted_animals:
        print("✅ The statement is TRUE!")
    else:
        print("❌ The statement is FALSE!")
