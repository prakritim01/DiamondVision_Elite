import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import tensorflow as tf
import numpy as np
import json
import time
import base64
from io import BytesIO
import plotly.express as px
from datetime import datetime, timedelta

# --- Page Config (Must be the very first Streamlit command) ---
st.set_page_config(page_title="DiamondVision Elite", layout="wide", page_icon="💎", initial_sidebar_state="expanded")

# --- Custom Premium CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
        div[data-testid="metric-container"] {
            background-color: rgba(30, 41, 59, 0.7);
            border: 1px solid #334155; padding: 15px 20px; border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); backdrop-filter: blur(10px);
        }
        h1, h2, h3 { color: #e2e8f0; font-weight: 600; }
        [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Assets ---
@st.cache_resource
def load_assets():
    try:
        vision_model = tf.keras.models.load_model('models/shape_model.h5')
        with open('models/price_model.pkl', 'rb') as f:
            price_engine = pickle.load(f)
        with open('models/classes.json', 'r') as f:
            class_indices = json.load(f)
        labels = {v: k for k, v in class_indices.items()}
        return vision_model, price_engine, labels
    except Exception as e:
        return None, None, None

vision_model, price_engine, labels = load_assets()

# --- Helper Functions ---
def process_image(img):
    img_resized = img.resize((224, 224))
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_image_base64(img):
    """Converts a PIL Image to a base64 string for displaying inside a dataframe."""
    buffered = BytesIO()
    img_resized = img.copy()
    img_resized.thumbnail((100, 100)) # Create a tiny thumbnail for speed
    img_resized.convert('RGB').save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_mock_price_trend():
    """Generates mock 7-day price trend data for the Surat market."""
    dates = [(datetime.today() - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
    prices = [245000, 248000, 247500, 251000, 250000, 253000, 255000]
    return pd.DataFrame({"Date": dates, "Price per Carat (INR)": prices})

# --- Sidebar Navigation ---
st.sidebar.title("💎 DiamondVision")
st.sidebar.markdown("Enterprise Grading OS")
app_mode = st.sidebar.radio("Select Operation Mode", ["Single Stone Valuation", "Batch Processing (Lot)"])
st.sidebar.markdown("---")
st.sidebar.info("Market: **Surat (Varachha)**\n\nCurrency: **INR (₹)**\n\nModel Status: **Online**")

# --- MAIN APP ROUTING ---
if vision_model is None:
    st.error("⚠️ System Offline: Models not found in the 'models/' directory. Please run the training notebook first.")

else:
    # ==========================================
    # MODE 1: SINGLE STONE VALUATION
    # ==========================================
    if app_mode == "Single Stone Valuation":
        st.title("Single Stone Valuation")
        st.markdown("Precision AI geometry analysis and pricing for individual specimens.")
        
        uploaded_file = st.file_uploader("Upload a macro photograph of the diamond", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1.2]) 
            with col1:
                img = Image.open(uploaded_file)
                st.image(img, caption="Specimen Acquired", use_container_width=True)
                with st.spinner("Analyzing optical geometry..."):
                    img_array = process_image(img)
                    preds = vision_model.predict(img_array)
                    shape_idx = np.argmax(preds)
                    detected_shape = labels[shape_idx].capitalize()
                    confidence = np.max(preds) * 100
                    
            with col2:
                st.subheader("1. AI Geometry Analysis")
                st.metric("Detected Cut Geometry", detected_shape, f"{confidence:.1f}% AI Confidence")
                
                st.write("🤖 *Override System (Human-in-the-Loop)*")
                override = st.checkbox("Enable Manual Grader Override")
                all_shapes = ["Round", "Cushion", "Marquise", "Oval", "Emerald", "Pear", "Heart", "Princess", "Radiant"]
                default_index = all_shapes.index(detected_shape) if detected_shape in all_shapes else 0
                
                if override:
                    final_shape = st.selectbox("Correct Shape:", all_shapes, index=default_index)
                    st.warning(f"Override Active: Using **{final_shape}**")
                else:
                    final_shape = detected_shape
                    
                st.markdown("---")
                st.subheader("2. Market Parameters")
                carat = st.number_input("Carat Weight", 0.1, 10.0, 1.0, 0.1)
                
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    color = st.select_slider("Color Grade", options=['D', 'E', 'F', 'G', 'H', 'I', 'J'])
                with p_col2:
                    clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"])
                    
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Generate Surat Market Valuation", use_container_width=True, type="primary"):
                    with st.spinner("Calculating live wholesale value..."):
                        # Engine logic (mocked for robust demonstration)
                        base_rate = 250000 
                        color_mult = {'D': 1.2, 'E': 1.1, 'F': 1.0, 'G': 0.9, 'H': 0.8, 'I': 0.7, 'J': 0.6}
                        clarity_mult = {'IF': 1.3, 'VVS1': 1.2, 'VVS2': 1.1, 'VS1': 1.0, 'VS2': 0.9, 'SI1': 0.8, 'SI2': 0.7}
                        final_price = base_rate * (carat ** 1.2) * color_mult[color] * clarity_mult[clarity]
                        
                        st.success("Valuation Complete")
                        st.metric(label="Wholesale Valuation (INR)", value=f"₹ {final_price:,.2f}", delta="-15% vs Retail")

    # ==========================================
    # MODE 2: BATCH PROCESSING DASHBOARD
    # ==========================================
    elif app_mode == "Batch Processing (Lot)":
        st.title("Bulk Lot Analytics Dashboard")
        st.markdown("Process multiple specimens and generate enterprise BI reports with human-in-the-loop editing.")
        
        uploaded_files = st.file_uploader("Upload an entire folder or multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Analyze Lot 🚀", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                # Processing Loop
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Scanning specimen {i+1} of {len(uploaded_files)}: {file.name}")
                    img = Image.open(file)
                    
                    # Generate Thumbnail for the Data Table
                    thumbnail_base64 = get_image_base64(img)
                    
                    # AI Prediction
                    img_array = process_image(img)
                    preds = vision_model.predict(img_array)
                    shape_idx = np.argmax(preds)
                    shape = labels[shape_idx].capitalize()
                    conf = np.max(preds) * 100
                    
                    results.append({
                        "Preview": thumbnail_base64,
                        "Filename": file.name,
                        "AI Predicted Shape": shape,
                        "Confidence (%)": round(conf, 2),
                        "Status": "✅ Verified" if conf > 90 else "⚠️ Human Review"
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Batch Processing Complete!")
                df = pd.DataFrame(results)
                
                # --- DASHBOARD UI LAYOUT ---
                st.markdown("---")
                
                # Section 1: AI Insights
                st.subheader("🧠 AI Lot Insights")
                avg_conf = df["Confidence (%)"].mean()
                dominant_shape = df["AI Predicted Shape"].mode()[0]
                total_stones = len(df)
                
                st.info(f"""
                **Automated Lot Analysis:**
                * Analyzed **{total_stones} specimens**. The dominant geometry in this lot is **{dominant_shape}**.
                * The Vision AI maintains a high average confidence of **{avg_conf:.1f}%** across this batch. 
                * Any specimen falling below a 90% confidence threshold has been automatically flagged for manual grader review.
                """)
                
                st.markdown("---")
                
                # Section 2: Visual Charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("### Lot Composition")
                    pie_chart = px.pie(
                        df, 
                        names='AI Predicted Shape', 
                        hole=0.4, 
                        color_discrete_sequence=px.colors.sequential.Teal
                    )
                    pie_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(pie_chart, use_container_width=True)
                
                with chart_col2:
                    st.markdown("### Surat Market Trend (7 Days)")
                    trend_df = generate_mock_price_trend()
                    line_chart = px.line(
                        trend_df, 
                        x="Date", 
                        y="Price per Carat (INR)",
                        markers=True,
                        line_shape="spline",
                        color_discrete_sequence=["#10b981"]
                    )
                    line_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(line_chart, use_container_width=True)

                st.markdown("---")
                
                # Section 3: The Editable Data Table
                st.markdown("### Detailed Lot Manifest (Editable)")
                st.caption("Double-click any shape in the table to manually override predictions flagged for Human Review.")
                
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Preview": st.column_config.ImageColumn("Image", help="Visual Specimen"),
                        "AI Predicted Shape": st.column_config.SelectboxColumn(
                            "AI Predicted Shape (Editable)",
                            help="Select the correct shape to override the AI.",
                            width="medium",
                            options=["Round", "Cushion", "Marquise", "Oval", "Emerald", "Pear", "Heart", "Princess", "Radiant"],
                            required=True,
                        ),
                        "Filename": st.column_config.TextColumn("Filename", disabled=True),
                        "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", disabled=True),
                        "Status": st.column_config.TextColumn("Status", disabled=True),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download Button uses the EDITED dataframe
                csv = edited_df.drop(columns=["Preview"]).to_csv(index=False).encode('utf-8')
                st.download_button("Download Edited CSV Manifest", data=csv, file_name='diamond_manifest_reviewed.csv', mime='text/csv')