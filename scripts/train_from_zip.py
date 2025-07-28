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
from toolkit.job import get_job
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

def generate_config(dataset_folder, lora_name, concept_sentence, steps=1000, lr=4e-4, rank=16):
    # 读取模板
    with open("config/examples/train_lora_flux_schnell_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)
    # 替换顶层 name
    config["config"]["name"] = lora_name
    # 替换数据集路径
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    # 替换训练步数和学习率
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    # 替换 LoRA rank
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    # 替换 trigger_word
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    config_path = f"tmp/{uuid.uuid4()}-{lora_name}.yaml"
    os.makedirs("tmp", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def main(zip_path, lora_name, concept_sentence):
    temp_dir = f"tmp/unzip_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    extract_zip(zip_path, temp_dir)
    dataset_folder = create_dataset_from_images(temp_dir, concept_sentence)
    config_path = generate_config(dataset_folder, lora_name, concept_sentence)
    print(f"Starting training with config: {config_path}")
    job = get_job(config_path)
    job.run()
    job.cleanup()
    print("Training completed.")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_from_zip.py <zip_file> <lora_name> [concept_sentence]")
        sys.exit(1)
    zip_file = sys.argv[1]
    lora_name = sys.argv[2]
    concept_sentence = sys.argv[3] if len(sys.argv) > 3 else ""
    main(zip_file, lora_name, concept_sentence)

