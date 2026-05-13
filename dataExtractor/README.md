# Sentiment Analysis Web Application (FYP)

A comprehensive full-stack platform designed to analyze, compare, and visualize text sentiment using a multi-model approach. This project integrates rule-based logic, traditional machine learning, and deep learning architectures to provide a robust evaluation of textual data.

## 🚀 Features
- **Multi-Model Comparison:** Compare results across Rule-based (VADER, TextBlob), Machine Learning (Bernoulli Naive Bayes), and Deep Learning (BERT) models.
- **Dynamic Model Loading:** High-performance BERT models are hosted on **Hugging Face Hub** and downloaded on-demand to maintain a lightweight repository.
- **Batch Processing:** Integrated CSV upload functionality for large-scale dataset analysis.
- **Data Persistence:** Real-time logging of analysis results and confidence scores to **Firebase Realtime Database**.
- **Multilingual Support:** Automatic language detection and translation to English for global text inputs.

## 🛠️ Tech Stack
- **Backend:** Flask (Python 3.12)
- **Deep Learning:** PyTorch, Transformers (BERT)
- **Machine Learning:** Scikit-learn
- **NLP Tools:** TextBlob, VADER, Langdetect, Googletrans
- **Database:** Firebase Realtime Database
- **Model Hosting:** Hugging Face Hub

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmBai193/sentiment-analysis-fyp.git
cd sentiment-analysis-fyp
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Configuration (Required)
- **Firebase:** Generate a `credentials.json` from your Firebase Console (Project Settings > Service Accounts) and place it in the root directory.
- **NLTK Data:** Download the necessary linguistic corpora for TextBlob:
```bash
python -m textblob.download_corpora
```

## 🏃 Usage

Run the application locally:
```bash
python app.py
```

Access the interface at `http://localhost:8080`.

> **Note:** On the first run using a BERT model, the application will download approximately 1.5GB of model weights from Hugging Face. This occurs only once and requires an active internet connection.

## 📂 Project Structure

| File/Folder | Description |
|---|---|
| `app.py` | Main Flask server and sentiment analysis routing logic |
| `upload_models.py` | Utility script for managing Hugging Face repository uploads |
| `templates/` | HTML front-end components |
| `static/` | Custom CSS and JavaScript for interactive visualizations |
| `requirements.txt` | Comprehensive list of Python dependencies |

