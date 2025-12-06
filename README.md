🧴 Fungal Acne Ingredient Classifier

A clean, modern machine-learning app that analyses skincare ingredient lists and predicts their fungal-acne safety, overall risk, and ingredient-level breakdown.
Powered by a custom 10-class TF-IDF Logistic Regression model and optional Expert Mode with probabilities + LIME explainability.

✔️ Designed for both beginners and skincare experts
✔️ Beautiful Apple-style UI
✔️ Works entirely offline once deployed

🔗 Live Links

👉 Live Demo: [Add your Streamlit link here]
👉 GitHub Repo: [Add your repo link here]

🚀 Features
🧠 Machine Learning Model

Custom TF-IDF Vectorizer + Logistic Regression (10 classes)

Supports multi-class categories:

safe

neutral

malassezia_trigger

comedogenic

irritant

fatty_acid

preservative

surfactant

emollient_heavy

fragrance_heavy

🧪 Ingredient Safety Engine

Computes a fungal-acne score from 0–10

Highlights risky ingredients:

🟢 Safe

🟡 Mild/questionable

🔴 Known fungal-acne triggers

🧬 Expert Mode (Optional)

Includes:

Class probability distribution

Bar chart visualisation

LIME explanation for model interpretability

💄 UI & Experience

Apple-style clean design

Smooth cards, soft shadows, rounded chips

Toggle button for beginner/expert views

🖼️ Screenshots
🏠 Home Page
<img src="assets/screenshots/homepage.png" width="750"/>
📊 Result Page (Normal Mode)
<img src="assets/screenshots/result_page.png" width="750"/>
🧪 Expert Mode (Probabilities + LIME)
<img src="assets/screenshots/expert_mode.png" width="750"/>
🔍 How It Works

User pastes their ingredient list

Ingredients are cleaned + normalised

Text is passed through the TF-IDF model

Model predicts a class label

The app generates:

Fungal acne score

Risk explanation

Ingredient-level tags

Expert Mode shows probabilities + LIME explanation

📦 Installation

Clone the repository:

git clone https://github.com/bawan-dev/fungal-acne-classifier.git
cd fungal-acne-classifier


Install dependencies:

pip install -r requirements.txt

▶️ Run locally
streamlit run src/app.py


Your app will open automatically at:

http://localhost:8501

🧠 Training the Model (Optional)

If you want to retrain the TF-IDF model:

python -m src.train_tfidf


The model updates here:

/models/tfidf_multiclass_model.joblib

☁️ Deployment
🚀 Deploy to Streamlit Cloud

Push your repo to GitHub

Go to share.streamlit.io

Select your repo

Set Main file = src/app.py

Deploy

🚀 Deploy to HuggingFace Spaces (Recommended)

Create a new Space → Streamlit template

Upload your entire repo

Add requirements.txt

Deploy instantly

HuggingFace is faster and handles ML models better.

🛠️ Tech Stack
Component	Technology Used
ML Model	TF-IDF + Logistic Regression
Language	Python 3.x
Frontend	Streamlit
Explainability	LIME
Data Processing	Pandas, NumPy
Deployment	Streamlit Cloud / HuggingFace
📂 Project Structure
fungal-acne-classifier/
│
├── assets/
│   └── screenshots/
│       ├── homepage.png
│       ├── result_page.png
│       └── expert_mode.png
│
├── data/
│   └── ingredients_multilabel.csv
│
├── logs/
│   └── analysis_log.csv
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
├── requirements.txt
├── LICENSE
└── README.md

🧭 Roadmap (Future Improvements)

🔍 Ingredients search engine

🧬 Upgrade to BERT/DistilBERT deep-learning model

📲 Export results as PDF

🧑‍🤝‍🧑 User accounts + saved analysis history

📊 Dashboard of common ingredients + triggers

💾 REST API endpoint

📜 License

This project is licensed under the MIT License — free to modify, use, and share.

❤️ Acknowledgements

Built by Bawan — inspired by the need for clearer ingredient transparency and better fungal-acne education using interpretable machine learning.