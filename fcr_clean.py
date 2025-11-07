import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="🦐 Shrimp Detection & FCR Dashboard", layout="wide")
st.title("🦐 Shrimp Detection & FCR Dashboard")

# --- CONSTANTS ---
a, b = 0.0046, 2.99   # Species constants
ruler_length_mm = 300  # Reference ruler length

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")
output_folder = os.path.join(BASE_DIR, "shrimp_result")
os.makedirs(output_folder, exist_ok=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ YOLO model not found at: {model_path}")
        return None
    return YOLO(model_path)

model = load_model()
if not model:
    st.stop()
else:
    st.success("✅ YOLO model loaded successfully!")

# --- FEED INPUT ---
feed_input = st.sidebar.number_input("Enter Feed Input (grams)", min_value=1.0, value=100.0, step=10.0)
st.sidebar.info("📦 Feed input will be used for FCR calculation.")

# --- UPLOAD IMAGES ---
uploaded_files = st.file_uploader("📸 Upload Shrimp Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    data_records = []

    for uploaded_file in uploaded_files:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_file.name

        st.write(f"🔍 **Processing:** {image_name}")
        results = model(img)
        result = results[0]
        boxes = result.boxes
        names = result.names

        ruler_boxes, shrimp_boxes = [], []
        for box in boxes:
            cls = int(box.cls[0])
            label = names[cls].lower()
            if label == "ruler":
                ruler_boxes.append(box.xyxy[0])
            elif label == "shrimp":
                shrimp_boxes.append(box.xyxy[0])

        if not ruler_boxes:
            st.warning(f"⚠️ No ruler detected in {image_name}, skipping...")
            continue

        # --- PIXEL TO MM CONVERSION ---
        x1, y1, x2, y2 = ruler_boxes[0]
        ruler_px_len = float(((x2 - x1)**2 + (y2 - y1)**2)**0.5)
        px_per_mm = ruler_px_len / ruler_length_mm

        shrimp_lengths, shrimp_weights = [], []
        for shrimp_box in shrimp_boxes:
            sx1, sy1, sx2, sy2 = map(int, shrimp_box)
            shrimp_px_len = float(((sx2 - sx1)**2 + (sy2 - sy1)**2)**0.5)
            shrimp_mm_len = shrimp_px_len / px_per_mm
            shrimp_cm_len = shrimp_mm_len / 10.0
            shrimp_weight_g = a * (shrimp_cm_len ** b)

            shrimp_lengths.append(round(shrimp_mm_len, 2))
            shrimp_weights.append(round(shrimp_weight_g, 2))

            # Annotate image
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{shrimp_mm_len:.1f}mm | {shrimp_weight_g:.2f}g",
                (sx1, max(30, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # --- CALCULATIONS ---
        shrimp_count = len(shrimp_boxes)
        avg_length = np.mean(shrimp_lengths) if shrimp_lengths else 0
        avg_weight = np.mean(shrimp_weights) if shrimp_weights else 0
        total_biomass = avg_weight * shrimp_count
        weight_gain = total_biomass
        fcr = round(feed_input / weight_gain, 2) if weight_gain > 0 else 0

        # Save to records
        data_records.append({
            "Image": image_name,
            "Individual_Lengths_mm": shrimp_lengths,
            "Individual_Weights_g": shrimp_weights,
            "Avg_Length_mm": round(avg_length, 2),
            "Avg_Weight_g": round(avg_weight, 2),
            "Feed_Input_g": feed_input,
            "Weight_Gain_g": round(weight_gain, 2),
            "FCR": fcr
        })

        # Convert image for Streamlit display
        annotated_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img, caption=f"{image_name} — FCR: {fcr}", use_container_width=True)

    # --- RESULTS TABLE ---
    df = pd.DataFrame(data_records)
    st.subheader("📊 Detection Results")
    st.dataframe(df)

    # --- DOWNLOAD RESULTS ---
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    st.download_button(
        label="📥 Download Results (Excel)",
        data=buffer.getvalue(),
        file_name=f"shrimp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.ms-excel"
    )
else:
    st.info("⬆️ Upload shrimp images to begin detection.")
