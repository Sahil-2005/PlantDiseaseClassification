import os
import io
import time
import numpy as np
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file automatically
load_dotenv()

# App Configuration
st.set_page_config(
    page_title="FloraDoctor — AI Plant Disease Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Header Gradient & Hero styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981 0%, #34D399 50%, #6EE7B7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        max-width: 750px;
        line-height: 1.6;
    }
    
    /* Modern Glass Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-disease {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
    }
    
    /* Chat & Advisory Container */
    .advisory-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1rem;
        line-height: 1.7;
    }
    
    /* Sidebar styling tweak */
    .sidebar-section-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9CA3AF;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Global model state
BACKEND = None
MODEL = None
TFLITE_INTERPRETER = None
IMG_SIZE = (224, 224)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    # Model Setup
    st.markdown('<div class="sidebar-section-title">Model Architecture</div>', unsafe_allow_html=True)
    use_tflite = st.toggle("Use TFLite Runtime", value=False, help="Enable lightweight TFLite inference if available.")
    
    default_model = "plant_disease_no_finetune.h5" if os.path.exists("plant_disease_no_finetune.h5") else "plant_disease_classifier_final.keras"
    model_path = st.text_input(
        "Model File",
        value=default_model,
        help="Path to trained model file (.h5 or .keras)."
    )
    
    # Diagnostics & Features
    st.markdown('<div class="sidebar-section-title">Diagnostics & Explainability</div>', unsafe_allow_html=True)
    show_gradcam = st.checkbox("Generate Grad-CAM Heatmap", value=True, help="Compute visual class activation heatmap.")
    conf_threshold = st.slider("Confidence Alarm Threshold", 0.0, 1.0, 0.40, 0.05)
    
    # LLM Settings (via .env)
    st.markdown('<div class="sidebar-section-title">AI Agronomist (Groq LLM)</div>', unsafe_allow_html=True)
    enable_llm = st.toggle("Enable AI Prescription", value=True, help="Generate treatments and preventive measures via Groq.")
    
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        st.success("API Key loaded from `.env`", icon="🔑")
        groq_api_key = env_key
    else:
        st.warning("`GROQ_API_KEY` not found in `.env`", icon="⚠️")
        groq_api_key = ""
        st.caption("Add `GROQ_API_KEY=your_key` to `.env` file to enable AI treatment plans.")
    
    # Model selection
    available_models = ["llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]
    if groq_api_key:
        try:
            from groq import Groq
            _client = Groq(api_key=groq_api_key)
            _fetched = [m.id for m in _client.models.list().data if "whisper" not in m.id and "guard" not in m.id]
            if _fetched:
                available_models = sorted(_fetched, key=lambda x: (not x.startswith("llama-3.1"), x))
        except Exception:
            pass

    llm_model = st.selectbox("LLM Reasoner", available_models, index=0)
    
    st.divider()
    st.caption("🌿 **FloraDoctor v2.0** • Deep Learning & Generative AI")

# --- Utilities ---
@st.cache_resource(show_spinner=False)
def load_labels():
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def format_class_name(raw_name: str) -> str:
    cleaned = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    return cleaned.strip()

def preprocess_image(pil_img, size=IMG_SIZE):
    img = pil_img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, axis=0)
    # Scaled to [0, 1] matching MobileNetV2 training
    return x / 255.0

@st.cache_resource(show_spinner=True)
def load_keras_model(path):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)

def predict_with_keras(model, x):
    preds = model.predict(x)
    return preds[0]

def get_topk(probs, labels, k=5):
    idxs = np.argsort(probs)[::-1][:k]
    rows = []
    for i in idxs:
        name = labels[i] if labels and i < len(labels) else f"Class #{i}"
        rows.append((int(i), name, float(probs[i])))
    return rows

def get_groq_advisory(disease_name: str, confidence: float, api_key: str, model_name: str):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        is_healthy = "healthy" in disease_name.lower()
        
        system_prompt = (
            "You are an expert Chief Plant Pathologist and Master Agronomist. "
            "Provide a crisp, actionable, structured advisory for growers and agricultural technicians. "
            "Format the response using clear Markdown with rich emojis and bullet points: "
            "### 📋 Diagnosis Summary\n"
            "### ⚡ Immediate Curative Actions (Organic & Chemical recommendations with dosage & caution)\n"
            "### 🛡️ Cultural & Preventive Protocols (Watering, spacing, pruning)\n"
            "### 🔄 Crop Rotation & Long-Term Field Health\n"
            "If the plant is healthy, celebrate the grower and give maintenance and yield-boosting advice."
        )
        
        user_prompt = (
            f"Diagnosed Leaf Condition: **{format_class_name(disease_name)}**\n"
            f"Classifier Confidence: **{confidence*100:.1f}%**\n\n"
            "Generate the complete clinical advisory now."
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Unable to generate advisory from Groq: {e}"

# --- Model Loading ---
def ensure_model():
    global BACKEND, MODEL
    if os.path.exists(model_path):
        try:
            MODEL = load_keras_model(model_path)
            BACKEND = "keras"
        except Exception as e:
            st.error(f"Failed to load model file: {e}")
            BACKEND = None
    else:
        st.warning(f"Model file `{model_path}` not found. Check repository or adjust sidebar path.")
        BACKEND = None

ensure_model()
labels = load_labels()

# --- Hero Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌿 FloraDoctor AI</div>
    <div class="hero-subtitle">
        Next-generation plant disease detection powered by <strong>MobileNetV2 Transfer Learning</strong> 
        with <strong>Grad-CAM explainability</strong> and <strong>Groq LLM agronomic treatment planning</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Layout: Two Column Interactive Dashboard ---
col_upload, col_analysis = st.columns([1, 1.2], gap="large")

with col_upload:
    st.markdown("### 📸 Input Plant Leaf")
    uploaded = st.file_uploader(
        "Upload leaf image (PNG, JPG, JPEG)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear close-up picture of an infected or healthy leaf."
    )
    
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)
    else:
        st.info("👆 Upload a leaf photo to initiate automated diagnostics.")
        
        # Quick samples guide
        with st.expander("💡 Recommended test classes:"):
            st.markdown("""
            - **Tomato**: Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Yellow Leaf Curl, Healthy
            - **Potato**: Early Blight, Late Blight, Healthy
            - **Bell Pepper**: Bacterial Spot, Healthy
            """)

with col_analysis:
    if uploaded is not None and BACKEND is not None:
        x = preprocess_image(image)
        
        with st.spinner("Analyzing botanical features & computing class probabilities..."):
            probs = predict_with_keras(MODEL, x)
            topk = get_topk(probs, labels, k=min(5, len(probs)))
            best_idx, best_name, best_conf = topk[0]
            display_name = format_class_name(best_name)
            is_healthy = "healthy" in best_name.lower()

        st.markdown("### 🔬 Clinical Diagnosis")
        
        # Diagnosis Banner
        status_class = "status-healthy" if is_healthy else "status-disease"
        status_text = "HEALTHY SPECIMEN" if is_healthy else "DISEASE DETECTED"
        
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-badge {status_class}">{status_text}</span>
            <h2 style="margin: 0.6rem 0 0.2rem 0; font-size: 1.8rem; font-weight: 700;">{display_name}</h2>
            <p style="color: #9CA3AF; margin-bottom: 0.8rem;">Classifier Confidence</p>
            <div style="font-size: 2rem; font-weight: 800; color: #10B981;">{best_conf*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        if best_conf < conf_threshold:
            st.warning("⚠️ Low confidence warning: The model is below the alarm threshold. Ensure good lighting and a clear close-up of the leaf.", icon="⚠️")

        # Probability Distribution
        st.markdown("#### 📊 Confidence Breakdown (Top-5)")
        for rank, (idx, name, score) in enumerate(topk, 1):
            col_lbl, col_bar = st.columns([2, 3])
            with col_lbl:
                st.caption(f"**{rank}.** {format_class_name(name)}")
            with col_bar:
                st.progress(float(score), text=f"{score*100:.1f}%")

        # Grad-CAM Explainability
        if show_gradcam:
            st.markdown("#### 🧠 Explainable AI: Grad-CAM Saliency")
            try:
                import tensorflow as tf
                import cv2
                
                last_conv = None
                for layer in reversed(MODEL.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer.name
                        break
                if last_conv is None:
                    last_conv = "top_conv"

                @st.cache_resource(show_spinner=False)
                def compute_gradcam(img_array, _model, last_conv_layer_name):
                    grad_model = tf.keras.models.Model(
                        [_model.inputs],
                        [_model.get_layer(last_conv_layer_name).output, _model.output]
                    )
                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(img_array)
                        pred_index = tf.argmax(predictions[0])
                        class_channel = predictions[:, pred_index]
                    grads = tape.gradient(class_channel, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs = conv_outputs[0]
                    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                    return heatmap.numpy()

                heatmap = compute_gradcam(x, MODEL, last_conv)
                img_display = np.array(image.convert("RGB").resize(IMG_SIZE))
                hm = cv2.resize((heatmap * 255).astype("uint8"), IMG_SIZE)
                hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(hm, 0.45, img_display, 0.55, 0)

                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.image(hm, caption="Focus Heatmap", use_container_width=True)
                with g_col2:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.info(f"Grad-CAM visualization skipped: {e}")

    elif uploaded is None:
        st.markdown("### 🔬 Clinical Diagnosis")
        st.caption("Waiting for leaf specimen upload to begin analysis...")

# --- Full Width AI Agronomist Prescription & Chat ---
if uploaded is not None and BACKEND is not None and enable_llm:
    st.markdown("---")
    st.markdown("## 🤖 AI Agronomist Clinical Advisory")
    
    if not groq_api_key:
        st.info("💡 **AI Advisory is currently offline**: Add your Groq API key to `.env` (`GROQ_API_KEY=your_key`) to unlock real-time treatment protocols, organic remedies, and interactive Q&A.")
    else:
        # Cache key based on diagnosis
        if "last_diagnosed" not in st.session_state or st.session_state.last_diagnosed != best_name:
            st.session_state.last_diagnosed = best_name
            st.session_state.advisory_text = None
            st.session_state.chat_history = []

        if st.session_state.advisory_text is None:
            with st.spinner("AI Agronomist is analyzing symptoms and formulating care protocol..."):
                st.session_state.advisory_text = get_groq_advisory(best_name, best_conf, groq_api_key, llm_model)

        with st.container():
            st.markdown(st.session_state.advisory_text)

        # Interactive Q&A
        st.markdown("### 💬 Ask the AI Plant Doctor")
        st.caption("Have specific questions about fungicides, natural organic solutions, watering routines, or soil health? Ask below:")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a follow-up question (e.g. 'Can I use baking soda or neem oil for this?')...")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                system_qa = (
                    f"You are an empathetic, highly knowledgeable Senior Plant Pathologist consulting a grower whose plant was identified as {format_class_name(best_name)}. "
                    "Provide practical, concise, and safe advice regarding remedies, dosage, weather effects, and care."
                )
                msgs = [{"role": "system", "content": system_qa}]
                for m in st.session_state.chat_history:
                    msgs.append({"role": m["role"], "content": m["content"]})

                with st.chat_message("assistant"):
                    with st.spinner("Formulating response..."):
                        resp = client.chat.completions.create(
                            model=llm_model,
                            messages=msgs,
                            temperature=0.4,
                            max_tokens=600
                        )
                        reply = resp.choices[0].message.content
                        st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error communicating with AI Agronomist: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6B7280; font-size: 0.85rem;'>FloraDoctor AI • Powered by MobileNetV2, TensorFlow, Streamlit & Groq Cloud</div>", unsafe_allow_html=True)
