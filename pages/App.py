import streamlit as st
import tensorflow as tf
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ✅ important
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import io
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- App Setup ---
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.warning("Please login first.")
    st.stop()

# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.image("logo.jpg", width=200)

st.markdown(
    "<h1 style='text-align: center; font-size: 32px; font-weight: bold;'>"
    "Iqra University – Department of Medical Sciences</h1>",
    unsafe_allow_html=True
)
st.write(f"Welcome, **{st.session_state['username']}**!")
st.title("🧬 Thyroid Disease Detection System")


# --- Load Model ---
# MODEL_PATH = "thyroid_model_finetuned_3class.h5"
MODEL_PATH = "thyroid_model_v4.keras"
CLASS_NAMES = ['Benign', 'Malignant']  # ✅ Make sure order matches training

@st.cache_resource
def load_thyroid_model():
    return load_model(MODEL_PATH)

model = load_thyroid_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ✅ match training preprocessing
    return np.expand_dims(img_array, axis=0)

def generate_gradcam_heatmap(model, image_tensor):
    score = BinaryScore(1)  # Malignant = 1
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    cam = gradcam(score, image_tensor, penultimate_layer=-1)

    heatmap = cam[0]
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(heatmap, cmap='jet')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def generate_diet_plan(prediction: str):
    prompt = f"""
    Act as a certified medical nutritionist. A patient has been diagnosed with '{prediction}' thyroid condition.
    Give a personalized 3-day diet plan including:
    - Breakfast, lunch, dinner, snacks
    - Focus on foods that support or manage {prediction}
    - Explain each day in simple language
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a diet expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content


# --- Form UI ---
st.subheader("👤 Patient Details")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        contact = st.text_input("Contact Number")

    with col2:
        address = st.text_area("Address")
        symptoms = st.multiselect(
            "Symptoms",
            ["Fatigue", "Weight Gain", "Cold Sensitivity", "Depression", "Constipation", "Dry Skin", "Other"]
        )
        date = st.date_input("Date", value=datetime.today())

    uploaded_image = st.file_uploader("📤 Upload Thyroid Scan Image", type=["png", "jpg", "jpeg"])

    predict_btn = st.form_submit_button("🧠 Predict Diagnosis")
    generate_pdf = st.form_submit_button("📝 Generate PDF Report")

# --- Prediction Logic ---
if predict_btn and uploaded_image:
    with st.spinner("Analyzing scan..."):
        image = Image.open(uploaded_image).convert("RGB")
        x = preprocess_image(image)
        prob = model.predict(x)[0][0]  # sigmoid output between 0–1
        idx = int(prob > 0.5)
        prediction = CLASS_NAMES[idx]
        confidence = prob if idx == 1 else 1 - prob
        gradcam_buf = generate_gradcam_heatmap(model, x)
        st.session_state["gradcam_buf"] = gradcam_buf



    # st.write("🔍 Probabilities:", {
    #     "Benign": f"{(1 - prob):.2%}",
    #     "Malignant": f"{prob:.2%}"
    # })
    st.success(f"🩺 Diagnosis: **{prediction}** ({confidence:.2%} confidence)")
    st.subheader("📊 Model Explanation (Grad-CAM)")
    st.image(gradcam_buf, caption="Grad-CAM: What part of the image influenced the decision", use_column_width=True)
    if prediction:
        st.subheader("🍽️ AI-Generated Diet Plan")
        with st.spinner("Creating personalized diet..."):
            diet_plan = generate_diet_plan(prediction)
            st.markdown(diet_plan)
    st.session_state["last_prediction"] = {
        "name": name,
        "age": age,
        "gender": gender,
        "contact": contact,
        "address": address,
        "symptoms": ", ".join(symptoms),
        "date": date.strftime('%Y-%m-%d'),
        "diagnosis": prediction,
        "confidence": f"{confidence:.2%}"
    }

# --- Generate PDF Logic ---
if generate_pdf:
    data = st.session_state.get("last_prediction")
    gradcam_buf = st.session_state.get("gradcam_buf")
    if not data:
        st.warning("⚠️ Please run a prediction first before generating a report.")
    else:
        pdf = FPDF()
        pdf.add_page()
        pdf.image("logo.jpg", x=10, y=8, w=25)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Iqra University - Dept. of Medical Sciences", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Thyroid Diagnosis Report", ln=True, align='C')
        pdf.ln(10)

        for k, v in data.items():
            pdf.cell(200, 10, txt=f"{k.capitalize()}: {v}", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Model Explanation (Grad-CAM):", ln=True)
        # Add Grad-CAM image to PDF
        gradcam_path = "gradcam_temp.png"

        with open(gradcam_path, "wb") as f:
            f.write(gradcam_buf.getbuffer())
        
        pdf.multi_cell(0, 10, f"Diet Plan:\n{diet_plan}")
        # Add Grad-CAM image
        pdf.image(gradcam_path, x=30, y=None, w=140)  # Centered width
        pdf.ln(10)

        filename = f"Thyroid_Report_{data['name'].replace(' ', '_')}_{data['date'].replace('-', '')}.pdf"
        pdf.output(filename)

        st.success(f"✅ PDF Generated: {filename}")
        with open(filename, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name=filename)

