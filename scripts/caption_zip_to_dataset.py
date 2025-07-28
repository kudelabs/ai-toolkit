"""
caption_zip_to_dataset.py
------------------------
Extracts a zip file of images, captions them using Florence-2, and creates a dataset folder with images and metadata.jsonl.

Usage:
    python caption_zip_to_dataset.py <zip_file> [concept_sentence]

Arguments:
    <zip_file>           Path to the zip file containing images
    [concept_sentence]   (Optional) Concept sentence to append as [trigger] in captions

Example:
    python caption_zip_to_dataset.py my_images.zip "my concept"
"""
import os
import sys
import zipfile
import uuid
import shutil
import json
import yaml
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

def caption_images(image_dir, concept_sentence):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)
    captions = {}
    image_files = [f for f in os.listdir(image_dir) if is_image_file(f)]
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        if concept_sentence:
            caption_text = f"{caption_text} [trigger]"
        captions[image_file] = caption_text
    model.to("cpu")
    del model
    del processor
    return captions

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_dataset_from_images(image_dir, concept_sentence):
    dataset_folder = f"datasets/{uuid.uuid4()}"
    os.makedirs(dataset_folder, exist_ok=True)
    captions = caption_images(image_dir, concept_sentence)
    jsonl_file_path = os.path.join(dataset_folder, "metadata.jsonl")
    image_files = [f for f in os.listdir(image_dir) if is_image_file(f)]
    with open(jsonl_file_path, "w") as jsonl_file:
        for image_file in image_files:
            src = os.path.join(image_dir, image_file)
            dst = os.path.join(dataset_folder, image_file)
            shutil.copy(src, dst)
            caption = captions.get(image_file, "[trigger]" if concept_sentence else "")
            jsonl_file.write(json.dumps({"file_name": image_file, "prompt": caption}) + "\n")
    return dataset_folder

def main(zip_path, concept_sentence):
    temp_dir = f"tmp/unzip_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    extract_zip(zip_path, temp_dir)
    dataset_folder = create_dataset_from_images(temp_dir, concept_sentence)
    print(f"Dataset created at: {dataset_folder}")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caption_zip_to_dataset.py <zip_file> [concept_sentence]")
        sys.exit(1)
    zip_file = sys.argv[1]
    concept_sentence = sys.argv[2] if len(sys.argv) > 2 else ""
    main(zip_file, concept_sentence)

