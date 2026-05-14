import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr
from PIL import Image
import numpy as np
import os
import sys

# =========================================================
# 1. DYNAMIC PATH SETUP
# =========================================================
script_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(script_path))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import modular architectures
from models.baseline_model import BaselineModel
from models.improved_model import Improved_Model
from models.pretrained_model import get_resnet_model

# =========================================================
# 2. CONFIGURATION & MODEL LOADING
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_PATHS = {
    "Model 1: Baseline Model": os.path.join(src_dir, "notebooks/model1_baseline.pt"),
    "Model 2: Improved Model": os.path.join(src_dir, "notebooks/model2_improved.pt"),
    "Model 3: Pretrained Model": os.path.join(src_dir, "notebooks/model3_pretrained.pt")
}

# Pre-load models into memory
loaded_models = {}
for name, path in MODELS_PATHS.items():
    if os.path.exists(path):
        print(f"Loading {name}...")
        m = torch.load(path, map_location=DEVICE, weights_only=False)
        m.to(DEVICE)
        m.eval()
        loaded_models[name] = m
    else:
        print(f"⚠️ Warning: Model file not found at {path}")

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Standard normalization for EuroSAT/ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================
# 3. REFERENCE GUIDE BUILDER
# =========================================================

# Maps model class keys → display labels used in the reference panel
CLASS_DISPLAY = {
    'Forest':                'Forest',
    'AnnualCrop':            'Annual Crop',
    'PermanentCrop':         'Permanent Crop',
    'Pasture':               'Pasture',
    'Residential':           'Residential',
    'Industrial':            'Industrial',
    'Highway':               'Highway',
    'HerbaceousVegetation':  'Herbaceous Vegetation',
    'River':                 'River',
    'SeaLake':               'Sea Lake',
}

def build_reference_md(top_class=None):
    """Return the reference markdown, highlighting the predicted class if given."""

    def item(class_key, suffix=""):
        display = CLASS_DISPLAY[class_key]
        if class_key == top_class:
            return (
                f'- <span style="background-color:#2FA084; color:#ffffff; '
                f'padding:2px 10px; border-radius:6px; font-weight:bold; '
                f'letter-spacing:0.3px;">✅ {display}</span>{suffix}'
            )
        return f"- {display}{suffix}"

    return f"""
### 📖 Class Reference Guide

**🌳 1. Natural Forest (The Baseline)**
{item('Forest')}

---

**🌾 2. Agricultural Deforestation**
*(Land Cleared for Farming)*
{item('AnnualCrop')}
{item('PermanentCrop')}
{item('Pasture')}

---

**🏗️ 3. Urbanization / Infrastructure**
*(Land Cleared for Construction)*
{item('Residential')}
{item('Industrial')}
{item('Highway')}

---

**🌿 4. Natural Non-Forest**
*(Reference Land Covers)*
{item('HerbaceousVegetation', ' *(Natural Grasslands)*')}
{item('River')}
{item('SeaLake')}
"""

# =========================================================
# 4. INFERENCE LOGIC
# =========================================================

def inference_engine(model_choice, image):
    if model_choice not in loaded_models:
        return None, build_reference_md()

    selected_model = loaded_models[model_choice]

    img_pil = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = selected_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)

    top_class = class_names[predicted.item()]
    probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    return probs, build_reference_md(top_class)

# =========================================================
# 5. GRADIO INTERFACE
# =========================================================

CUSTOM_CSS = """
/* ── Primary button ── */
button.primary                              { background-color: #1F6F5F !important; border-color: #1F6F5F !important; color: #ffffff !important; }
button.primary:hover                        { background-color: #2FA084 !important; border-color: #2FA084 !important; }

/* ── Probability bars ── */
.label-confidence-full                      { background-color: #2FA084 !important; }

/* ── Reference panel text ── */
.reference-panel, .reference-panel *        { color: #1F6F5F !important; }
.reference-panel h3                         { color: #1F6F5F !important; font-weight: 700; }
.reference-panel li                         { color: #1F6F5F !important; line-height: 1.8; }
.reference-panel hr                         { border-color: #6FCF97 !important; }

/* ── Headings ── */
h1                                          { color: #1F6F5F !important; }
"""

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🛰️ AI Deforestation Monitoring System")
    gr.Markdown("Real-time classification using Deep Learning architectures.")

    with gr.Row():
        # LEFT COLUMN: Inputs
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS_PATHS.keys()),
                value="Model 3: Pretrained Model",
                label="Step 1: Choose Intelligence Level"
            )
            img_input = gr.Image(label="Drop Satellite Tile Here")
            run_btn = gr.Button("🚀 Start Intelligence Analysis", variant="primary")

        # MIDDLE COLUMN: Results
        with gr.Column(scale=1):
            prob_output = gr.Label(label="Prediction Probabilities", num_top_classes=5)

        # RIGHT COLUMN: Class Reference Guide (dynamic — highlights predicted class)
        with gr.Column(scale=1):
            reference_panel = gr.Markdown(value=build_reference_md(), elem_classes=["reference-panel"])

    run_btn.click(
        fn=inference_engine,
        inputs=[model_selector, img_input],
        outputs=[prob_output, reference_panel]
    )

    gr.Markdown("---")
    gr.Markdown("*Developed by Ronish Prajapati*")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())