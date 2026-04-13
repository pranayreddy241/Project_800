import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

MODEL_PATH = "/mmfs1/home/pbairedd/vlm_project/models/llava-1.5-7b"
DATA_DIR = "/mmfs1/home/pbairedd/vlm_project/data"
IMAGE_DIR = os.path.join(DATA_DIR, "val2014")
RESULTS_DIR = "/mmfs1/home/pbairedd/vlm_project/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=dtype,
    low_cpu_mem_usage=True
).to(device)
model.eval()

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    if "assistant:" in text:
        text = text.split("assistant:")[-1].strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text

def ask_model(image_path, question):
    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question}\nAnswer only yes or no based on the image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    generated = input_ids

    for _ in range(10):
        model_inputs = {"input_ids": generated}
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        adjusted_logits = logits / (1.0 + 0.7 * torch.exp(-entropy))
        next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        eos_token_id = processor.tokenizer.eos_token_id
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return normalize_answer(text)

import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# assumes model, processor, device, dtype, DATA_DIR, IMAGE_DIR, RESULTS_DIR already defined above
model.eval()

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    if "assistant:" in text:
        text = text.split("assistant:")[-1].strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text

def vista_decode(inputs, max_new_tokens=10, alpha=0.7):
    """
    Lightweight VISTA-style decoding:
    - runs greedy decoding manually
    - computes token distribution entropy
    - adjusts logits using entropy-scaled confidence control
    - discourages overly sharp/confident token jumps
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    pixel_values = inputs.get("pixel_values", None)

    generated = input_ids

    for _ in range(max_new_tokens):
        model_inputs = {
            "input_ids": generated
        }

        if attention_mask is not None:
            current_attention = torch.ones_like(generated, device=generated.device)
            model_inputs["attention_mask"] = current_attention

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        with torch.no_grad():
            outputs = model(**model_inputs)

        logits = outputs.logits[:, -1, :]  # next-token logits
        probs = F.softmax(logits, dim=-1)

        # entropy of next-token distribution
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)

        # confidence-aware adjustment
        # lower entropy => more peaked/confident distribution
        # we soften such distributions slightly to reduce overconfident wrong picks
        adjusted_logits = logits / (1.0 + alpha * torch.exp(-entropy))

        next_token = torch.argmax(adjusted_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        # stop if EOS token generated
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated

def ask_model(image_path, question):
    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question}\nAnswer only yes or no based on the image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    output = vista_decode(inputs, max_new_tokens=10, alpha=0.7)

    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return normalize_answer(text)

for split in ["random", "popular", "adversarial"]:
    qfile = os.path.join(DATA_DIR, f"coco_pope_{split}.json")
    outfile = os.path.join(RESULTS_DIR, f"vista_pope_{split}.jsonl")

    with open(qfile, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Running split: {split} | questions: {len(questions)}")

    with open(outfile, "w") as fout:
        for q in tqdm(questions):
            image_path = os.path.join(IMAGE_DIR, q["image"])
            answer = ask_model(image_path, q["text"])
            fout.write(json.dumps({
                "question_id": q["question_id"],
                "image": q["image"],
                "question": q["text"],
                "label": q["label"],
                "text": answer
            }) + "\n")

    print(f"Saved: {outfile}")

print("All splits done.")

    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return normalize_answer(text)

for split in ["random", "popular", "adversarial"]:
    qfile = os.path.join(DATA_DIR, f"coco_pope_{split}.json")
    outfile = os.path.join(RESULTS_DIR, f"baseline_pope_{split}.jsonl")

    with open(qfile, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Running split: {split} | questions: {len(questions)}")

    with open(outfile, "w") as fout:
        for q in tqdm(questions):
            image_path = os.path.join(IMAGE_DIR, q["image"])
            answer = ask_model(image_path, q["text"])
            fout.write(json.dumps({
                "question_id": q["question_id"],
                "image": q["image"],
                "question": q["text"],
                "label": q["label"],
                "text": answer
            }) + "\n")

    print(f"Saved: {outfile}")

print("All splits done.")
