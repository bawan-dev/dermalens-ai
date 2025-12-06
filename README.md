🧬 DermaLens — AI Skincare Ingredient Analyzer

A project by Bawan Sabah

DermaLens is an AI-powered skincare tool that helps people instantly understand what’s inside their products.
It reads ingredient lists, scans barcodes, checks store availability, finds safer alternatives, and explains everything in clean, simple language.

The goal was to build something useful in real life, not just another ML demo — and DermaLens has grown into a full mini-ecosystem of machine learning, NLP, embeddings, scraping and a polished Streamlit app.

🚀 What DermaLens Can Do
🔍 Ingredient Analysis

Paste an ingredients list or upload a product label — DermaLens breaks everything down:

Classifies the formula using a trained ML model (10 labels)

Gives a fungal acne safety score

Highlights safe / risky ingredients

Explains why the model predicted what it did (LIME)

🧠 AI Similar Products

Using sentence-transformer embeddings, DermaLens can:

Find products that “feel” similar based on ingredients

Compare the user’s product to known formulas

Suggest safer or cleaner alternatives

📷 Image & Barcode Support

You can upload:

A photo of the ingredients

A screenshot

A barcode

DermaLens tries to extract the text, decode the barcode, and fetch the correct ingredients in the background.

🛒 Store Availability

It also checks major UK retailers (Boots, Superdrug) to show:

Where the product is sold

Whether it’s in stock

Direct links to buy it

⭐ Favourites

Users can save products into a personal “favourites” list.
They stay stored locally and can be reloaded instantly.

🌓 Beautiful UI With Dark Mode

The app now has a clean, modern design inspired by real consumer skincare apps:

Dark/light mode toggle

Ingredient chips

Tabs for Overview / Ingredients / Similar Products / Expert Mode

Smooth animations and a minimal look

💡 Why I Built This

I wanted a project that wasn’t just “train a model and stop”.
DermaLens let me combine everything I’ve been learning:

Machine learning

NLP

Embeddings

Model explainability

Data scraping

OCR + barcode decoding

Streamlit UI/UX

File persistence and caching

Deployment and testing

It turned into a proper full-stack ML product — not just an assignment.

🧱 Tech Stack

Machine Learning / NLP

Scikit-learn (TF-IDF + Logistic Regression)

Sentence-transformers (BERT embeddings)

LIME (model explainability)

Backend

Python

Ingredient lookup service

Store availability scraper

Barcode scanning (pyzbar, Pillow)

OCR stub (ready for Tesseract)

Frontend

Streamlit UI

Dark/light mode

PDF generation (FPDF)

Storage

Product memory (CSV + JSONL)

Embedding storage (PyTorch tensor)

▶️ How to Run Locally
git clone https://github.com/bawan-dev/Fungal-acne-classifier
cd fungal-acne-classifier
pip install -r requirements.txt

# optional: regenerate embeddings if you update the dataset
python src/ingredient_embeddings.py

streamlit run src/app.py

📂 Project Structure
src/
  app.py                 # main UI
  analysis_engine.py     # predictions, scoring, similarity, PDF generation
  embeddings_utils.py    # embedding loader + similarity helpers
  ingredient_lookup.py   # ingredient fetcher
  store_availability.py  # Boots/Superdrug scraper
  barcode_scanner.py     # barcode + image decoding
  user_favourites.py     # save/load favourite products
  ingredient_embeddings.py # regenerate embedding tensor
data/
  product_memory.csv
  user_product_memory.jsonl
models/
  tfidf_multiclass_model.joblib
  ingredient_embeddings.pt

🧪 Testing
pytest


Covers similarity functions, ingredient parsing, analysis pipeline, OCR stub, etc.

🌟 Future Features (Roadmap)

Real OCR (Tesseract + preprocessing)

Brand database + automatic product match

Live barcode scanning from webcam

More retailers (Sephora, Amazon, LookFantastic)

Mobile app version

Fine-tuned transformer model instead of TF-IDF

📬 Contact

If you want to collaborate or have a feature idea:

LinkedIn: https://www.linkedin.com/in/bawan-sabah-84281b371

GitHub: https://github.com/bawan-dev