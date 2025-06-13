import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, resnet50
from PIL import Image, UnidentifiedImageError
import io

# ---- Config ----
st.set_page_config(page_title="🧠 MediScan AI", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Models ----
@st.cache_resource
def load_oral_model():
    model = efficientnet_b2(weights=None)  # No pretrained weights
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load("oral_cancer_detector_best_b2.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_pulmo_model():
    model = resnet50(weights=None)  # No pretrained weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("chest_xray_resnet50.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# ---- Preprocessing ----
from torchvision import transforms

oral_transform = transforms.Compose([
    transforms.Resize(288),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

pulmo_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Prediction Functions ----
def predict_oral(image: Image.Image, model):
    image_tensor = oral_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
    if prob > 0.5:
        return 'NON CANCER', prob * 100
    else:
        return 'CANCER', (1 - prob) * 100

def predict_pulmo(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = pulmo_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
    if prob > 0.5:
        return 'PNEUMONIA', prob * 100
    else:
        return 'NORMAL', (1 - prob) * 100

# ---- App Layout ----
st.title("🧠 MediScan AI")
st.subheader("Upload an image and select a model for analysis")
st.markdown("---")

model_choice = st.selectbox("Choose model", ["Oralytics - Oral Cancer", "PulmoScan - Pneumonia Detection"])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing..."):

                    if model_choice.startswith("Oralytics"):
                        model = load_oral_model()
                        pred, conf = predict_oral(image, model)
                    else:
                        model = load_pulmo_model()
                        pred, conf = predict_pulmo(image_bytes, model)

                st.success("Analysis Complete")
                st.markdown("### 🧾 Result")
                st.write(f"**Prediction:** `{pred}`")
                st.write(f"**Confidence:** `{conf:.2f}%`")
                st.progress(int(conf))

    except UnidentifiedImageError:
        st.error("Invalid image file.")

else:
    st.info("Please upload an image to get started.")
st.markdown("""
---
⚠️ **Disclaimer:** This tool is intended for scientific research and educational purposes only.  
It **should NOT** be used for medical diagnosis or treatment decisions.  
Please consult a qualified healthcare professional for any medical concerns.
---
""")
