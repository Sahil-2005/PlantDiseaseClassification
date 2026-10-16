# 🌿 FloraDoctor AI — Plant Disease Diagnostics & AI Agronomist

An end-to-end Deep Learning and Generative AI application for real-time plant leaf disease detection, explainable CNN attention mapping (**Grad-CAM**), and personalized clinical treatment prescriptions powered by **Groq Cloud LLMs (LLaMA 3)**.

---

## 📌 Key Highlights

- **Deep Learning Classifier**: MobileNetV2 transfer learning architecture fine-tuned on the PlantVillage dataset for rapid edge-level image classification.
- **15 Supported Conditions**: Accurate identification across Tomato, Potato, and Bell Pepper crops (including healthy specimens).
- **Explainable AI (Grad-CAM)**: Gradient-weighted Class Activation Mapping displays visual saliency heatmaps highlighting the exact leaf lesions influencing predictions.
- **AI Agronomist Advisory**: Automated clinical prescriptions detailing diagnosis summaries, organic & chemical treatments with dosages, and cultural sanitation protocols.
- **Interactive Botanical Chat**: In-app conversational assistant to answer follow-up queries regarding fungicides, watering, and crop protection.
- **Modern Glassmorphic UI**: High-end two-column diagnostic dashboard with real-time confidence telemetry.
- **Secure Environment Management**: API credentials managed securely via `.env` file.

---

## 🗂 Project Structure

```text
Plant-Disease-Classification/
├── app.py                                # Streamlit interactive web dashboard
├── plant_disease_no_finetune.h5          # Pre-trained MobileNetV2 Keras model
├── labels.txt                            # 15 class labels in model order
├── plant-disease-transfer-learning.ipynb # Model training & evaluation notebook
├── requirements.txt                      # Project dependencies
├── sample_images/                        # Test photos & evaluation guide
│   └── README.md
├── .env.example                          # Environment template
├── .gitignore                            # Excludes venv, secrets, and caches
└── README.md                             # Project documentation
```

---

## 🏷 Supported Crop Classes (15 Categories)

| Crop | Condition / Disease |
| :--- | :--- |
| **Pepper Bell** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-spotted Spider Mite, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🚀 Setup & Execution Guide

Follow these steps to set up and run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Sahil-2005/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

### 2. Set Up a Python Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Configure Your Environment Variables (`.env`)
1. Copy the example environment template:
   ```bash
   cp .env.example .env
   ```
   *(Or on Windows PowerShell: `Copy-Item .env.example .env`)*
2. Obtain a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys).
3. Open `.env` and paste your key:
   ```env
   GROQ_API_KEY=gsk_your_actual_api_key_here
   ```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧠 Model Architecture & Technical Specs

- **Backbone**: MobileNetV2 (Pre-trained on ImageNet)
- **Input Dimension**: `(224, 224, 3)`
- **Preprocessing**: Normalized to `[0, 1]` range (`x / 255.0`)
- **Top Layers**:
  - `GlobalAveragePooling2D`
  - `Dense(512, activation='relu')` + `Dropout(0.5)`
  - `Dense(256, activation='relu')` + `Dropout(0.3)`
  - `Dense(15, activation='softmax')`
- **Saliency Mapping**: Custom TensorFlow GradientTape Grad-CAM extractor targeting the final `Conv2D` layer (`Conv_1`).
- **LLM Engine**: Groq Cloud (`llama-3.1-8b-instant` / `llama3-70b-8192`).

---

## 👨‍💻 Author

**Sahil Gawade**

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://sahil-gawade.vercel.app/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Sahil-2005)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahil-gawade-920a0a242/)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/sahilgawade4321/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:gawadesahil.dev@gmail.com)

---

## 📜 License
This project is open-source and distributed under the [MIT License](LICENSE).
