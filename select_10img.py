import os
import random
import torch
from PIL import Image
from torchvision import transforms
from utils import ImprovedMNISTFeatureExtractor

# =========================
# Configuration
# =========================
INPUT_DIR = "folder_of_samples/samples_soft_baseline"
OUTPUT_PATH = "images_report/samples_soft_baseline.png"
CHECKPOINT_PATH = "checkpoints/cnn_mnist_features_extractor.pkl"

DEVICE = torch.device("cpu")
IMAGES_PER_ROW = 5

# =========================
# Load trained MNIST classifier
# =========================
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = ImprovedMNISTFeatureExtractor(
    feature_dim=checkpoint["feature_dim"]
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================
# Randomized digit selection
# =========================
files = os.listdir(INPUT_DIR)
random.shuffle(files)  # 🔑 ensures different images each run

selected = {}

with torch.no_grad():
    for fname in files:
        if len(selected) == 10:
            break

        path = os.path.join(INPUT_DIR, fname)

        try:
            img = Image.open(path).convert("L")
        except Exception:
            continue  # skip unreadable files

        x = transform(img).unsqueeze(0).to(DEVICE)
        logits = model(x, return_features=False)
        pred = logits.argmax(dim=1).item()

        if pred not in selected:
            selected[pred] = img

if len(selected) < 10:
    raise RuntimeError("Could not find all digits 0–9 in generated samples.")

# =========================
# Build 5×2 grid image
# =========================
w, h = next(iter(selected.values())).size
canvas = Image.new("L", (IMAGES_PER_ROW * w, 2 * h))

for idx, digit in enumerate(range(10)):
    row = idx // IMAGES_PER_ROW
    col = idx % IMAGES_PER_ROW
    canvas.paste(selected[digit], (col * w, row * h))

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
canvas.save(OUTPUT_PATH)

print(f"✅ Combined image saved to: {OUTPUT_PATH}")