# 🧴 Fungal Acne Ingredient Classifier  
A lightweight machine-learning tool that analyses skincare ingredient lists and detects their fungal-acne safety, ingredient risks, and category classification — powered by a custom **10-class TF-IDF model** and optional **expert-mode explanations** using LIME.

The app provides both:
- **Human Mode** → simple, easy-to-understand results  
- **Expert Mode** → probabilities, bar charts, and LIME explanation  

👉 **Live Demo:** *[Your Streamlit link here]*  
👉 **GitHub Repo:** *[Your repo link here]*  

---

## 🚀 Features

### 🧠 Machine Learning  
- Custom **TF-IDF + Logistic Regression multiclass model**  
- 10 label taxonomy (safe, neutral, fungal-trigger, comedogenic, irritant, etc.)  
- Expert probability breakdown + bar chart  
- LIME local explanation showing which words influenced the prediction  

### 🧴 Fungal Acne Safety  
- Automatic rating from **0–10**  
- Ingredient-level breakdown:  
  - 🟢 Safe  
  - 🟡 Mild / questionable  
  - 🔴 Known fungal-acne triggers  

### 💄 Beautiful UI  
- Clean, Apple-style interface  
- Toggle between Human & Expert mode  
- Soft cards, colour-coded badges, ingredient chips  

---

## 🖼️ Screenshots

### 🏠 Home Page
<img src="assets/screenshots/homepage.png" width="750"/>

---

### 📊 Ingredient Analysis (Normal Mode)
<img src="assets/screenshots/result_page.png" width="750"/>

---

### 🧪 Expert Mode (Probabilities + LIME)
<img src="assets/screenshots/expert_mode.png" width="750"/>

---

## 🔍 How It Works

1. User pastes a skincare ingredient list  
2. Ingredients are cleaned, normalised, and tokenised  
3. They are fed into the **TF-IDF model**  
4. The model assigns one of 10 label categories  
5. The app generates:
   - Fungal acne risk score  
   - Explanation of risk  
   - Colour-coded ingredient chips  
6. (Expert mode) Probabilities + LIME explanation  

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/bawan-dev/fungal-acne-classifier.git
cd fungal-acne-classifier
Install dependencies:

pip install -r requirements.txt

▶️ Run locally
streamlit run src/app.py

Your app will open at:

http://localhost:8501

🧠 Model Training (Optional)

If you want to retrain the 10-class TF-IDF model:

python -m src.train_tfidf


The trained model will be saved to:

/models/tfidf_multiclass_model.joblib

☁️ Deployment
▶️ Streamlit Cloud

Upload your repo

Set Main file = src/app.py

Add requirements.txt

Deploy

▶️ HuggingFace Spaces (recommended)

Use Streamlit template

Upload model + code

Deploy instantly

🛠️ Tech Stack
Component	Used
ML Model	TF-IDF + Logistic Regression
Language	Python 3.x
Framework	Streamlit
Explainability	LIME
Data Handling	Pandas, NumPy
Visualization	Streamlit native charts
Deployment	Streamlit Cloud / HuggingFace
🧩 Project Structure
fungal-acne-classifier/
│
├── data/
│   └── ingredients_multilabel.csv
│
├── models/
│   └── tfidf_multiclass_model.joblib
│
├── src/
│   ├── app.py
│   ├── train_tfidf.py
│   ├── predict_tfidf.py
│   ├── preprocessing.py
│   ├── safety_score.py
│   └── analytics.py
│
├── assets/
│   └── screenshots/
│       ├── homepage.png
│       ├── result_page.png
│       └── expert_mode.png
│
├── requirements.txt
└── README.md

🧭 Roadmap / Future Improvements

🔍 Add ingredient search engine

🧬 Add deep learning model (BERT or DistilBERT fine-tuned)

📲 Export results as PDF

💾 User accounts & product history

📦 Add API endpoint

📊 Dashboard of most common triggers

📜 License

MIT License — free to use, modify, and share.

❤️ Acknowledgements

Built by Bawan — inspired by the need for clearer ingredient transparency and better fungal acne education.


---

If you'd like:

🔥 A **README banner** with gradient  
🔥 A **project logo**  
🔥 A **demo GIF**  
🔥 A version written like a startup product page  

Just tell me — *“make the README look like a real SaaS product”* or *“make it more aesthetic”*.