# 🌿 Plant Disease Classification — Deep Learning Web App

An end-to-end Computer Vision and Deep Learning application to detect and classify plant leaf diseases from images using Transfer Learning with **MobileNetV2** and an interactive **Streamlit** dashboard with **Grad-CAM** visual interpretability.

---

## 📌 Features

- **Transfer Learning with MobileNetV2**: Pre-trained on ImageNet and fine-tuned on the PlantVillage dataset.
- **15 Supported Plant Conditions**: Classifies diseases across Bell Pepper, Potato, and Tomato crops.
- **Top-5 Confidence Scores**: View model prediction rankings and confidence percentages.
- **Explainable AI (Grad-CAM)**: Heatmap overlay on leaf images to visualize which regions the CNN focused on during classification.
- **Streamlit Web Interface**: Fast, responsive, and easy-to-use user interface for instant inference.

---

## 🗂 Project Structure

```text
Plant-Disease-Classification/
├── app.py                                # Streamlit dashboard application
├── plant_disease_no_finetune.h5          # Trained MobileNetV2 Keras model
├── labels.txt                            # 15 class labels in model order
├── plant-disease-transfer-learning.ipynb # Model training & evaluation notebook
├── requirements.txt                      # Project dependencies
├── sample_images/                        # Test leaf photos & evaluation guide
│   └── README.md
├── .gitignore                            # Git ignore rules for Python & venv
└── README.md                             # Documentation
```

---

## 🏷 Supported Disease Classes

The model classifies leaves into 15 specific categories:

| Crop | Disease / Condition |
| :--- | :--- |
| **Pepper Bell** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted spider mite), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Plant-Disease-Classification
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model & Architecture Details

- **Base Backbone**: MobileNetV2 (ImageNet weights)
- **Input Resolution**: 224 × 224 × 3
- **Classification Head**:
  - GlobalAveragePooling2D
  - Dense (512 units, ReLU activation) + Dropout (0.5)
  - Dense (256 units, ReLU activation) + Dropout (0.3)
  - Dense (15 units, Softmax output)
- **Interpretability**: Grad-CAM (Gradient-weighted Class Activation Mapping) implemented via TensorFlow GradientTape.

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
This project is open-source and available under the [MIT License](LICENSE).

