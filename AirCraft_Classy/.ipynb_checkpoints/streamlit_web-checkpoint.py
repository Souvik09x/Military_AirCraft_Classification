import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from PyTorchLabFlow import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Define the `predict` function as you described
def predict(path, ppl):
    root_dir = "../"  # Root directory (modify based on your structure)
    P = re_train(ppl=ppl)  # Assuming `re_train(ppl)` returns a trained model
    model_weights = torch.load(P.weights_path, weights_only=True)
    P.model.load_state_dict(model_weights, strict=False)

    # Set model to evaluation mode
    P.model.eval()

    # Load and preprocess the image
    try:
        img = Image.open(path).convert("RGB")  # Convert to RGB to ensure consistent color channels
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Define the default transformations (resize, to tensor, and normalization)
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():  # Disable gradient tracking for inference
        pred = F.softmax(P.model(img_tensor), dim=1)

    # Get the predicted class index (index of max probability)
    pred_class_idx = torch.argmax(pred, dim=1).item()

    # Dynamically load class names from the directory (root_dir)
    class_names = sorted([
        'A10', 'A400M', 'AG600', 'AH64', 'An124', 'An22', 'An225', 'An72', 'AV8B', 'B1', 'B2', 'B21', 'B52', 'Be200',
        'C130', 'C17', 'C2', 'C390', 'C5', 'CH47', 'CL415', 'E2', 'E7', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'H6',
        'J10', 'J20', 'JAS39', 'JF17', 'JH7', 'Ka27', 'Ka52', 'KC135', 'KF21', 'KJ600', 'Mi24', 'Mi26', 'Mi28', 'Mig29', 'Mig31', 'Mirage2000',
        'MQ9', 'P3', 'Rafale', 'RQ4', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57', 'TB001', 'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2', 'UH60',
        'US2', 'V22', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'Z19'
    ])  # Replace with your actual class names
    idx_to_class = {idx: class_name for idx, class_name in enumerate(class_names)}

    # Get the predicted class name using the reverse mapping
    pred_class_name = idx_to_class.get(pred_class_idx, "Unknown Class")

    return pred_class_name, pred


# Streamlit UI with HTML and CSS styling
st.markdown("""
    <style>
        .header {
            text-align: center;
            color: #2c3e50;
            font-size: 36px;
            font-weight: bold;
            margin-top: 50px;
        }
        .subheader {
            text-align: center;
            font-size: 18px;
            color: #34495e;
        }
        .upload-box {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            margin-top: 30px;
            padding: 20px;
            border: 2px solid #3498db;
            border-radius: 10px;
            background-color: #ecf0f1;
        }
        .result-box {
            text-align: center;
            margin-top: 30px;
        }
        .result-box h3 {
            font-size: 24px;
            color: #16a085;
        }
        .result-box p {
            font-size: 18px;
            color: #2c3e50;
        }
    </style>
    <div class="header">
        Aircraft Classification Model
    </div>
    <div class="subheader">
        Choose an experiment (ppl) and upload an image for classification.
    </div>
""", unsafe_allow_html=True)

# Initialize selected_ppl in session state if it's not already there
if 'selected_ppl' not in st.session_state:
    st.session_state.selected_ppl = "exp12"  # Default experiment

# Dropdown to select the experiment (ppl)
ppl_options = get_ppls()  # Replace with your actual experiment classes
selected_ppl = st.selectbox("Select Experiment (ppl)", ppl_options, index=ppl_options.index(st.session_state.selected_ppl))

# Update session state when a new experiment is selected
st.session_state.selected_ppl = selected_ppl


# Placeholder image to show before upload (can be a local or online image)
st.subheader("Upload an Image to Classify")


# File uploader for image classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If an image is uploaded, process it and make a prediction
if uploaded_file is not None:
    # Save the uploaded image temporarily to a path
    image_path = "./temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make prediction using the `predict` function
    pred_class_name, pred = predict(image_path, selected_ppl)

    # Show the prediction result
    st.markdown(f"""
    <div class="result-box">
        <h3>Prediction: {pred_class_name}</h3>
        <p>Confidence: {pred.max().item():.2f}</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-box">
        <p>Upload an image to start the classification process.</p>
    </div>
    """, unsafe_allow_html=True)

# Display performance plot and experiment descriptions below prediction
st.subheader("Model Performance ")

# Generate the comparison plot
sx = performance_plot(figsize=(10, 3))  # Adjusted figure size

# Loop through all the plots returned by the performance_plot() function
for idx, (fig, ax) in enumerate(sx):  # Assuming sx is a list of (ax, fig) pairs
    st.pyplot(fig)  # Display each figure

# Display experiment descriptions
ppl_model = {i: re_train(ppl=i).model_name for i in ppl_options}

# Description of each experiment
description_text = """
_40 == training last 40 layers  
_un == training all layers  
The last exp12 uses dynamic learning rate
"""
st.markdown(f"**Description of all experiments:**\n {description_text}")

# Display the model descriptions for each experiment
for ppl, model_name in ppl_model.items():
    st.markdown(f"**{ppl}:** {model_name}")
