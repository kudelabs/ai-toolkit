"""
train_from_dataset.py
---------------------
Runs training from an existing dataset folder using the AI Toolkit job system.

Usage:
    python train_from_dataset.py <dataset_folder> <lora_name> [concept_sentence]

Arguments:
    <dataset_folder>     Path to the dataset folder (with images and metadata.jsonl)
    <lora_name>          Name for the LoRA model/output
    [concept_sentence]   (Optional) Concept sentence/trigger word for training

Example:
    python train_from_dataset.py datasets/1234abcd my_lora "my concept"
"""
import os
import sys
import uuid
import shutil
import yaml
from toolkit.job import get_job

def generate_config(dataset_folder, lora_name, concept_sentence, steps=1000, lr=4e-4, rank=16):
    # Read template
    with open("config/examples/train_lora_flux_schnell_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)
    # Replace top-level name
    config["config"]["name"] = lora_name
    # Replace dataset path
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    # Replace training steps and learning rate
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    # Replace LoRA rank
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    # Replace trigger_word
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    config_path = f"tmp/{uuid.uuid4()}-{lora_name}.yaml"
    os.makedirs("tmp", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

def main(dataset_folder, lora_name, concept_sentence):
    config_path = generate_config(dataset_folder, lora_name, concept_sentence)
    print(f"Starting training with config: {config_path}")
    job = get_job(config_path)
    job.run()
    job.cleanup()
    print("Training completed.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_from_dataset.py <dataset_folder> <lora_name> [concept_sentence]")
        sys.exit(1)
    dataset_folder = sys.argv[1]
    lora_name = sys.argv[2]
    concept_sentence = sys.argv[3] if len(sys.argv) > 3 else ""
    main(dataset_folder, lora_name, concept_sentence)

