import os
import json
from PIL import Image, ImageFilter
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

def ask_model(image, question):
    prompt = f"USER: <image>\n{question}\nAnswer only yes or no based on the image.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    text = processor.batch_decode(output, skip_special_tokens=True)[0]
    return normalize_answer(text)

def distort_image(image):
    return image.filter(ImageFilter.GaussianBlur(radius=5))

def combine_answers(ans_orig, ans_blur):
    if ans_orig == ans_blur:
        return ans_orig
    if ans_orig in ["yes", "no"] and ans_blur in ["yes", "no"]:
        return "no"
    if ans_orig in ["yes", "no"]:
        return ans_orig
    if ans_blur in ["yes", "no"]:
        return ans_blur
    return "no"

for split in ["random", "popular", "adversarial"]:
    qfile = os.path.join(DATA_DIR, f"coco_pope_{split}.json")
    outfile = os.path.join(RESULTS_DIR, f"vcd_pope_{split}.jsonl")

    with open(qfile, "r") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Running VCD split: {split} | questions: {len(questions)}")

    with open(outfile, "w") as fout:
        for q in tqdm(questions):
            image_path = os.path.join(IMAGE_DIR, q["image"])
            image = Image.open(image_path).convert("RGB")
            blur_image = distort_image(image)

            ans_orig = ask_model(image, q["text"])
            ans_blur = ask_model(blur_image, q["text"])
            final_ans = combine_answers(ans_orig, ans_blur)

            fout.write(json.dumps({
                "question_id": q["question_id"],
                "image": q["image"],
                "question": q["text"],
                "label": q["label"],
                "text": final_ans,
                "orig_answer": ans_orig,
                "blur_answer": ans_blur
            }) + "\n")

    print(f"Saved: {outfile}")

print("All VCD splits done.")
