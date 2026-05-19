import torch
import torch.nn as nn
from torchvision import transforms
import gradio as gr
from PIL import Image
import os
import sys
from torchvision import models

# =========================================================
# 1. DYNAMIC PATH SETUP
# =========================================================
script_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(script_path))
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# =========================================================
# 2. ARCHITECTURE DEFINITIONS
# =========================================================
class BaselineModel(nn.Module):
    def __init__(self, num_classes = 10, in_channels = 3):
        super(BaselineModel, self).__init__()
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d (in_channels, 96, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d (96, 256, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d (256, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
        )

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d (384, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
        )

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv2d (384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3,stride = 2)
        )

        # Fully Connected Layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    # Forward Pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x


class Improved_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Improved_Model, self).__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), 
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def get_resnet_model(num_classes=10):
    # Load Pre-Trained Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze Layer 4     
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# =========================================================
# 3. CONFIGURATION & STATE-DICT WEIGHT LOADING
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_PATHS = {
    "Model 1: Baseline Model": os.path.join(src_dir, "notebooks/model1_baseline.pth"),
    "Model 2: Improved Model": os.path.join(src_dir, "notebooks/model2_improved.pth"),
    "Model 3: Pretrained Model": os.path.join(src_dir, "notebooks/model3_pretrained.pth")
}

model_creators = {
    "Model 1: Baseline Model": lambda: BaselineModel(num_classes=10),
    "Model 2: Improved Model": lambda: Improved_Model(num_classes=10),
    "Model 3: Pretrained Model": lambda: get_resnet_model(num_classes=10)
}

loaded_models = {}
for name, path in MODELS_PATHS.items():
    if os.path.exists(path):
        print(f"Loading weights into {name} from {path}...")
        try:
            m = model_creators[name]()
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE)
            m.eval()
            loaded_models[name] = m
        except Exception as e:
            print(f"❌ Error instantiating/loading weights for {name}: {e}")
    else:
        print(f"⚠️ Warning: Model file not found at {path}")

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================
# 4. REFERENCE GUIDE BUILDER
# =========================================================
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
# 5. INFERENCE LOGIC
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
# 6. GRADIO INTERFACE
# =========================================================
CUSTOM_CSS = """
button.primary                              { background-color: #1F6F5F !important; border-color: #1F6F5F !important; color: #ffffff !important; }
button.primary:hover                        { background-color: #2FA084 !important; border-color: #2FA084 !important; }
.label-confidence-full                      { background-color: #2FA084 !important; }
.reference-panel, .reference-panel * { color: #1F6F5F !important; }
.reference-panel h3                         { color: #1F6F5F !important; font-weight: 700; }
.reference-panel li                         { color: #1F6F5F !important; line-height: 1.8; }
.reference-panel hr                         { border-color: #6FCF97 !important; }
h1                                          { color: #1F6F5F !important; }
"""

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🛰️ AI Deforestation Monitoring System")
    gr.Markdown("Real-time classification using Deep Learning architectures.")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS_PATHS.keys()),
                value="Model 3: Pretrained Model",
                label="Step 1: Choose Intelligence Level"
            )
            # CHANGED: Added sources=["upload"] to completely disable the webcam interface
            img_input = gr.Image(label="Drop Satellite Tile Here", sources=["upload"])
            run_btn = gr.Button("🚀 Start Intelligence Analysis", variant="primary")

        with gr.Column(scale=1):
            prob_output = gr.Label(label="Prediction Probabilities", num_top_classes=5)

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