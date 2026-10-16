
import os
import io
import time
import numpy as np
from PIL import Image
import streamlit as st

# Optional: Keras + TFLite support
BACKEND = None
MODEL = None
TFLITE_INTERPRETER = None
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")

st.title("🌿 Plant Disease Classification")
st.write("Upload a leaf photo and get the predicted disease class using a transfer-learned CNN.")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Model & Labels")
    use_tflite = st.toggle("Use TFLite if available", value=False, help="Prefer TFLite model if present.")
    
    # Auto-detect best existing model file
    default_model = "plant_disease_no_finetune.h5" if os.path.exists("plant_disease_no_finetune.h5") else "plant_disease_classifier_final.keras"
    model_path = st.text_input(
        "Model filename",
        value=default_model,
        help="Put the model file in the same folder as this app."
    )
    st.caption("If you change model file name, ensure the file is present on the server.")
    st.divider()
    st.subheader("Class Labels")
    labels_source = st.radio("Provide labels via:", ["Default / labels.txt file", "Paste list"], horizontal=True)
    pasted_labels = ""
    label_file = None
    if labels_source == "Paste list":
        pasted_labels = st.text_area(
            "Paste comma-separated labels (in the training order)",
            value="",
            placeholder="e.g. Pepper__bell___Bacterial_spot, Potato___Early_blight, ...",
            height=80
        )
    else:
        label_file = st.file_uploader("Upload custom labels.txt (optional)", type=["txt"])

    show_gradcam = st.checkbox("Show Grad-CAM (Keras only)", value=True, help="Requires Keras model, not supported for TFLite.")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)

    st.divider()
    st.subheader("🤖 AI Agronomist (Groq)")
    enable_llm = st.toggle("Enable AI Treatment Advisory", value=True, help="Generate treatment and prevention plans using Groq LLM.")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Enter your Groq API key. You can get one free at console.groq.com."
    )
    llm_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0
    )

# --- Utilities ---
@st.cache_resource(show_spinner=False)
def load_labels_from_text(text: str):
    labs = [x.strip() for x in text.split(",") if x.strip()]
    return labs

@st.cache_resource(show_spinner=False)
def load_labels_from_file(file_obj_or_path):
    try:
        if isinstance(file_obj_or_path, str):
            with open(file_obj_or_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
        else:
            content = file_obj_or_path.read().decode("utf-8")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
        return lines
    except Exception as e:
        st.warning(f"Failed to read labels file: {e}")
        return []

def _resolve_labels():
    if labels_source == "Paste list" and pasted_labels.strip():
        return load_labels_from_text(pasted_labels)
    elif labels_source == "Default / labels.txt file":
        if label_file is not None:
            return load_labels_from_file(label_file)
        elif os.path.exists("labels.txt"):
            return load_labels_from_file("labels.txt")
    return []

def preprocess_image(pil_img, size=IMG_SIZE, backend="keras"):
    img = pil_img.convert("RGB").resize(size)
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, axis=0)
    # The training notebook used rescale=1./255 for MobileNetV2
    x = x / 255.0
    return x

@st.cache_resource(show_spinner=True)
def load_keras_model(path):
    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(path, compile=False)
    return model

@st.cache_resource(show_spinner=True)
def load_tflite_model(path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_with_keras(model, x):
    preds = model.predict(x)
    return preds[0]

def predict_with_tflite(interpreter, input_details, output_details, x):
    # Assume single input/output
    interpreter.set_tensor(input_details[0]['index'], x.astype(input_details[0]['dtype']))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return out[0]

def get_topk(probs, labels, k=5):
    idxs = np.argsort(probs)[::-1][:k]
    rows = []
    for i in idxs:
        name = labels[i] if labels and i < len(labels) else f"class_{i}"
        rows.append((int(i), name, float(probs[i])))
    return rows

def format_class_name(raw_name: str) -> str:
    # Convert 'Tomato___Early_blight' or 'Tomato_Bacterial_spot' to 'Tomato - Early Blight'
    cleaned = raw_name.replace("___", " - ").replace("__", " - ").replace("_", " ")
    return cleaned.strip()

def get_groq_advisory(disease_name: str, confidence: float, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        system_prompt = (
            "You are an expert AI Agronomist and Plant Pathologist. "
            "Your job is to provide concise, scientifically accurate, and actionable guidance for farmers and gardeners. "
            "Format your advice cleanly using Markdown with distinct sections: "
            "1. Overview & Diagnosis Summary, "
            "2. Immediate Curative Actions (organic & chemical options with safety precautions), "
            "3. Preventive & Cultural Practices (watering, spacing, soil health), "
            "4. Long-Term Crop Rotation & Management. "
            "If the plant is diagnosed as healthy, congratulate the grower and provide maintenance/growth tips."
        )
        
        user_prompt = (
            f"The image classifier diagnosed the plant leaf as: **{format_class_name(disease_name)}** "
            f"with a confidence of {confidence*100:.1f}%.\n\n"
            "Please provide a comprehensive treatment, prevention, and advisory guide for this condition."
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Groq API: {e}"

# --- Model loader ---
def ensure_model():
    global BACKEND, MODEL, TFLITE_INTERPRETER
    if use_tflite and os.path.exists(model_path) and model_path.endswith(".tflite"):
        try:
            interpreter, in_det, out_det = load_tflite_model(model_path)
            BACKEND = "tflite"
            TFLITE_INTERPRETER = (interpreter, in_det, out_det)
            MODEL = None
            return
        except Exception as e:
            st.warning(f"Failed to load TFLite model: {e}. Falling back to Keras.")
    if os.path.exists(model_path):
        try:
            MODEL = load_keras_model(model_path)
            BACKEND = "keras"
            TFLITE_INTERPRETER = None
            return
        except Exception as e:
            st.error(f"Failed to load Keras model: {e}")
            BACKEND = None
    else:
        st.info("Model file not found yet. Please place the model file next to app.py or adjust the filename in the sidebar.")
        BACKEND = None

ensure_model()

# --- Main UI ---
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    labels = _resolve_labels()
    if not labels:
        st.info("No labels provided yet. You can still run predictions; class names will be shown as indices (class_#).")

    if BACKEND is None:
        st.stop()

    backend_name = "TensorFlow Lite" if BACKEND == "tflite" else "Keras"
    st.caption(f"Using backend: {backend_name}")

    x = preprocess_image(image, backend=BACKEND)

    with st.spinner("Predicting..."):
        if BACKEND == "keras":
            probs = predict_with_keras(MODEL, x)
        else:
            interpreter, in_det, out_det = TFLITE_INTERPRETER
            probs = predict_with_tflite(interpreter, in_det, out_det, x)

    topk = get_topk(probs, labels, k=min(5, len(probs)))
    best_idx, best_name, best_conf = topk[0]

    st.subheader("Prediction")
    st.metric(label="Top class", value=best_name, delta=f"{best_conf*100:.2f}%")

    st.write("Top-5 probabilities:")
    st.table(
        {"rank": list(range(1, len(topk)+1)),
         "index": [r[0] for r in topk],
         "label": [r[1] for r in topk],
         "confidence": [f"{r[2]*100:.2f}%" for r in topk]}
    )

    if best_conf < conf_threshold:
        st.warning("Low confidence prediction. Try a clearer image or different lighting/angle.")

    # Optional Grad-CAM (Keras only)
    if BACKEND == "keras" and show_gradcam:
        try:
            import tensorflow as tf
            last_conv = None
            for layer in reversed(MODEL.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            if last_conv is None:
                last_conv = "top_conv"  # common for EfficientNet

            @st.cache_resource(show_spinner=False)
            def compute_gradcam(img_array, model, last_conv_layer_name):
                grad_model = tf.keras.models.Model([model.inputs],
                                                   [model.get_layer(last_conv_layer_name).output, model.output])
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
            # Resize & overlay
            import cv2
            img_display = np.array(image.convert("RGB").resize(IMG_SIZE))
            hm = cv2.resize((heatmap*255).astype("uint8"), IMG_SIZE)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(hm, 0.4, img_display, 0.6, 0)

            st.subheader("Grad-CAM")
            st.image([img_display, hm, overlay], caption=["Image", "Heatmap", "Overlay"], use_column_width=True)
        except Exception as e:
            st.info(f"Grad-CAM not available: {e}")

    # --- AI Agronomist Advisory & Chat (Groq LLM) ---
    if enable_llm:
        st.markdown("---")
        st.subheader("🤖 AI Agronomist Advisory")

        if not groq_api_key.strip():
            st.info("💡 Enter your free **Groq API Key** in the sidebar to generate custom treatment, organic remedies, and prevention plans for this diagnosis.")
        else:
            # Generate Advisory
            if "last_diagnosed" not in st.session_state or st.session_state.last_diagnosed != best_name:
                st.session_state.last_diagnosed = best_name
                st.session_state.advisory_text = None
                st.session_state.chat_history = []

            if st.session_state.advisory_text is None:
                with st.spinner("AI Agronomist is analyzing disease and preparing treatment plan..."):
                    st.session_state.advisory_text = get_groq_advisory(best_name, best_conf, groq_api_key.strip(), llm_model)

            st.markdown(st.session_state.advisory_text)

            # Interactive Q&A about this diagnosis
            st.markdown("#### 💬 Ask the AI Plant Doctor")
            st.caption("Have questions about organic pesticides, dosage, or care? Ask below:")

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_question = st.chat_input("Ask a question about this plant disease...")
            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                try:
                    from groq import Groq
                    client = Groq(api_key=groq_api_key.strip())
                    system_qa = (
                        f"You are an AI Agronomist assisting a grower whose plant leaf was classified as {format_class_name(best_name)}. "
                        "Answer their questions concisely, practically, and empathetically with safe recommendations."
                    )
                    messages = [{"role": "system", "content": system_qa}]
                    for m in st.session_state.chat_history:
                        messages.append({"role": m["role"], "content": m["content"]})

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            resp = client.chat.completions.create(
                                model=llm_model,
                                messages=messages,
                                temperature=0.4,
                                max_tokens=600
                            )
                            answer = resp.choices[0].message.content
                            st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error answering question: {e}")
else:
    st.info("Upload an image to get started.")

st.markdown("---")
st.caption("Tip: Paste your class labels in the sidebar or upload a labels.txt file so the app can display disease names.")
