import json
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
import joblib
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pickle
import os
import io
import csv
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from langdetect import detect, LangDetectException
from googletrans import Translator, constants

# Initialize translator
translator = Translator()

# Dense transformer for GaussianNB
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Firebase setup
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://sentiment-analysis-fyp-befbd-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# Initialize basic analyzers
basic_sia = SentimentIntensityAnalyzer()


def classify_sentiment(text, threshold=0.1):
    """TextBlob sentiment classifier with configurable threshold"""
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    if abs(score) < threshold or analysis.sentiment.subjectivity < 0.3:
        return "Neutral"
    return "Positive" if score > 0 else "Negative"


def vader_predict(text, analyzer, threshold_pos=0.05, threshold_neg=-0.05):
    if not isinstance(text, str) or not text.strip():
        return 'neutral'
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    return 'positive' if compound >= threshold_pos else 'negative' if compound <= threshold_neg else 'neutral'


# Load all models
textblob_config = None
if os.path.exists('textblob_sentiment_model.pkl'):
    try:
        textblob_config = pickle.load(open('textblob_sentiment_model.pkl', 'rb'))
    except (AttributeError, pickle.UnpicklingError) as e:
        print(f"Error loading TextBlob config: {e}")
        # Create default config if pickle loading fails
        textblob_config = {
            "predict_function": classify_sentiment,
            "metadata": {
                "threshold": 0.1,
                "subjectivity_cutoff": 0.3,
                "version": "1.0",
                "model_type": "TextBlob"
            }
        }

models = {
    # Rule-based
    'textblob': {
        'name': 'TextBlob (Rule-based)',
        'type': 'rule_based',
        'analyzer': None,
        'config': textblob_config
    },
    'vader': {
        'name': 'VADER (Rule-based)',
        'type': 'rule_based',
        'analyzer': pickle.load(open('vader_sentiment_model.pkl', 'rb')) if os.path.exists(
            'vader_sentiment_model.pkl') else None,
        'config': None
    },

    # Machine learning: TextBlob-labeled
    'bernoullinb_textblob': {
        'name': 'BernoulliNB (TextBlob labels)',
        'type': 'ml',
        'model': joblib.load('model_bernoullinb_textblob.joblib') if os.path.exists(
            'model_bernoullinb_textblob.joblib') else None
    },
    # Baseline BernoulliNB: TextBlob
    'bernoullinb_textblob_baseline': {
        'name': 'BernoulliNB Baseline (TextBlob labels)',
        'type': 'ml',
        'model': joblib.load('bernoulli_nb_model_textblob_baseline.joblib') if os.path.exists(
            'bernoulli_nb_model_textblob_baseline.joblib') else None
    },
    # Baseline BernoulliNB: VADER
    'bernoullinb_vader_baseline': {
        'name': 'BernoulliNB Baseline (VADER labels)',
        'type': 'ml',
        'model': joblib.load('bernoulli_nb_model_vader_baseline.joblib') if os.path.exists(
            'bernoulli_nb_model_vader_baseline.joblib') else None
    },
    # Baseline BernoulliNB: ChatGPT
    'bernoullinb_chatgpt_baseline': {
        'name': 'BernoulliNB Baseline (ChatGPT labels)',
        'type': 'ml',
        'model': joblib.load('bernoulli_nb_model_chatgpt_baseline.joblib') if os.path.exists(
            'bernoulli_nb_model_chatgpt_baseline.joblib') else None
    },
    # BernoulliNB: ChatGPT (non-baseline)
    'bernoullinb_chatgpt': {
        'name': 'BernoulliNB (ChatGPT labels)',
        'type': 'ml',
        'model': joblib.load('model_bernoullinb_chatgpt.joblib') if os.path.exists(
            'model_bernoullinb_chatgpt.joblib') else None
    },
    # Machine learning: VADER-labeled
    'bernoullinb_vader': {
        'name': 'BernoulliNB (VADER labels)',
        'type': 'ml',
        'model': joblib.load('model_bernoullinb_vader.joblib') if os.path.exists(
            'model_bernoullinb_vader.joblib') else None
    },

    # BERT: TextBlob-labeled
    'bert_textblob': {
        'name': 'BERT (TextBlob labels)',
        'type': 'bert',
        'model': BertForSequenceClassification.from_pretrained('./bert_finetuned_textblob_model') if os.path.exists(
            './bert_finetuned_textblob_model') else None,
        'tokenizer': BertTokenizer.from_pretrained('./bert_finetuned_textblob_model') if os.path.exists(
            './bert_finetuned_textblob_model') else None
    },
    # BERT: VADER-labeled
    'bert_vader': {
        'name': 'BERT (VADER labels)',
        'type': 'bert',
        'model': BertForSequenceClassification.from_pretrained('./bert_finetuned_vader_model') if os.path.exists(
            './bert_finetuned_vader_model') else None,
        'tokenizer': BertTokenizer.from_pretrained('./bert_finetuned_vader_model') if os.path.exists(
            './bert_finetuned_vader_model') else None
    },
    # BERT: ChatGPT-labeled
    'bert_chatgpt': {
        'name': 'BERT (ChatGPT labels)',
        'type': 'bert',
        'model': BertForSequenceClassification.from_pretrained('./bert_finetuned_chatgpt_model') if os.path.exists(
            './bert_finetuned_chatgpt_model') else None,
        'tokenizer': BertTokenizer.from_pretrained('./bert_finetuned_chatgpt_model') if os.path.exists(
            './bert_finetuned_chatgpt_model') else None
    }

}

# Put BERT models in eval mode
for model_key in ['bert_textblob', 'bert_vader', 'bert_chatgpt']:
    if models[model_key]['model']:
        models[model_key]['model'].eval()

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}


def analyze_textblob(text):
    """Analyze text using TextBlob with saved configuration"""
    if not text or not isinstance(text, str):
        return 1, 0.0  # Return neutral for invalid input

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        # If no config, use basic thresholding
        if not models['textblob']['config']:
            return (
                2 if polarity > 0.05 else
                0 if polarity < -0.05 else
                1
            ), abs(polarity)

        # Use configured model if available
        config = models['textblob']['config']
        if not config or 'predict_function' not in config:
            raise ValueError("Invalid TextBlob configuration")

        predict_func = config['predict_function']
        threshold = config['metadata'].get('threshold', 0.1)
        result = predict_func(text, threshold=threshold)

        sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        return sentiment_map.get(result.lower(), 1), abs(polarity)

    except Exception as e:
        print(f"TextBlob analysis error: {str(e)}")
        return 1, 0.0  # Fallback to neutral


def analyze_vader(text):
    """Analyze text using VADER with saved configuration"""
    if not models['vader']['analyzer']:
        scores = basic_sia.polarity_scores(text)
        return (
            2 if scores["compound"] >= 0.05 else
            0 if scores["compound"] <= -0.05 else
            1
        ), abs(scores["compound"])

    # Use saved model configuration
    predict_func = models['vader']['analyzer']['predict_function']
    analyzer = models['vader']['analyzer']['analyzer']
    result = predict_func(text, analyzer=analyzer)
    scores = analyzer.polarity_scores(text)
    sentiment_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    return sentiment_map.get(result.lower(), 1), abs(scores["compound"])


def analyze_nb(text, model_key):
    """Analyze text using any saved Naive Bayes model"""
    if model_key not in models or not models[model_key].get('model'):
        return 1, 0.5  # Default to neutral if model not loaded

    try:
        # The pipeline saved includes vectorizer => model
        proba = models[model_key]['model'].predict_proba([text])[0]
        pred = models[model_key]['model'].predict([text])[0]
        confidence = float(max(proba))
        return int(pred), confidence
    except Exception:
        pred = models[model_key]['model'].predict([text])[0]
        return int(pred), 0.7


def analyze_bert(text, model_key):
    """Analyze text using BERT model"""
    if model_key not in models or not models[model_key]['model']:
        return 1, 0.5  # Default to neutral if model not loaded

    inputs = models[model_key]['tokenizer'](text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = models[model_key]['model'](**inputs)
        probs = softmax(outputs.logits, dim=1)
        sentiment = int(torch.argmax(probs))
        confidence = float(torch.max(probs))
    return sentiment, confidence


def detect_and_translate(text, preferred_lang=None):
    """
    Detect language and translate to English if not already English
    Returns tuple of (processed_text, original_language)
    """
    if not text or not isinstance(text, str):
        return text, "en"

    try:
        # Use preferred language if specified
        if preferred_lang and preferred_lang != 'auto':
            if preferred_lang == 'en':
                return text, "en"
            translation = translator.translate(text, dest='en')
            return translation.text, preferred_lang

        # Auto-detect language
        lang = detect(text)
        if lang == 'en':
            return text, "en"

        # Translate non-English text to English
        translation = translator.translate(text, dest='en')
        return translation.text, lang
    except LangDetectException:
        return text, "en"
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text, "en"

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/batch')
def batch_analysis():
    return render_template('batch_analysis.html')


@app.route('/classify', methods=['GET'])
def classify_text():
    try:
        model_key = request.args.get('model', '').lower()
        text = request.args.get('text', '').strip()
        preferred_lang = request.args.get('lang', 'auto')

        if not text or not model_key:
            return jsonify({"error": "Text or model missing."}), 400

        # Handle language detection and translation
        if preferred_lang == 'auto':
            processed_text, original_lang = detect_and_translate(text)
        else:
            # For manual language selection, skip detection and use specified language
            if preferred_lang == 'en':
                processed_text, original_lang = text, 'en'
            else:
                try:
                    translation = translator.translate(text, dest='en')
                    processed_text, original_lang = translation.text, preferred_lang
                except Exception as e:
                    print(f"Translation error: {str(e)}")
                    processed_text, original_lang = text, preferred_lang

        words = text.split()  # Keep original words for word-level analysis
        compare_all = (model_key == "all")
        models_to_run = models.keys() if compare_all else [model_key]

        results = {}
        valid_models = []

        for key in models_to_run:
            if key not in models:
                continue

            try:
                # Full-text sentiment (using processed/translated text)
                if key == 'textblob':
                    sentiment, confidence = analyze_textblob(processed_text)
                elif key == 'vader':
                    sentiment, confidence = analyze_vader(processed_text)
                elif key.startswith(('bernoullinb', 'complementnb', 'gaussiannb')):
                    sentiment, confidence = analyze_nb(processed_text, key)
                elif key.startswith('bert'):
                    sentiment, confidence = analyze_bert(processed_text, key)
                else:
                    raise ValueError("Invalid model: " + key)

                word_sentiments = []
                for w in words:
                    # For word-level analysis, process each word with same language handling
                    if preferred_lang == 'auto':
                        processed_word, _ = detect_and_translate(w)
                    else:
                        if preferred_lang == 'en':
                            processed_word = w
                        else:
                            try:
                                translation = translator.translate(w, dest='en')
                                processed_word = translation.text
                            except Exception:
                                processed_word = w

                    if key == 'textblob':
                        w_sent, w_conf = analyze_textblob(processed_word)
                    elif key == 'vader':
                        w_sent, w_conf = analyze_vader(processed_word)
                    elif key.startswith(('bernoullinb', 'complementnb', 'gaussiannb')):
                        w_sent, w_conf = analyze_nb(processed_word, key)
                    elif key.startswith('bert'):
                        w_sent, w_conf = analyze_bert(processed_word, key)
                    else:
                        continue

                    if w_conf > 0.5:
                        word_sentiments.append(w_sent)

                total = len(word_sentiments) or 1
                distribution = {
                    'Negative': word_sentiments.count(0) / total,
                    'Neutral': word_sentiments.count(1) / total,
                    'Positive': word_sentiments.count(2) / total
                }

                label = sentiment_labels.get(sentiment, "Unknown")
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                result_json = {
                    "text": text,
                    "processed_text": processed_text if original_lang != 'en' else None,
                    "original_language": original_lang,
                    "model_used": key,
                    "model_name": models[key]['name'],
                    "classification": {
                        "tag_name": label,
                        "confidence": round(confidence, 4)
                    },
                    "word_level_distribution": distribution,
                    "timestamp": now,
                    "language_preference": preferred_lang  # Track user's language selection
                }

                db.reference("results").push(result_json)
                valid_models.append(key)

                results[key] = {
                    "model_name": models[key]['name'],
                    "sentiment": int(sentiment),
                    "sentiment_label": label,
                    "confidence": float(round(confidence, 4)),
                    "distribution": [
                        distribution['Negative'],
                        distribution['Neutral'],
                        distribution['Positive']
                    ],
                    "json_preview": result_json,
                    "language_info": {
                        "original_language": original_lang,
                        "was_translated": original_lang != 'en',
                        "user_preference": preferred_lang
                    }
                }

            except Exception as e:
                results[key] = {
                    "error": str(e),
                    "model_name": models[key]['name']
                }

        # Historical distribution (unchanged)
        distribution = {}
        ref = db.reference("results")
        all_results = ref.get()

        for key in valid_models:
            model_results = []
            if all_results:
                for rec in all_results.values():
                    if rec.get('model_used') == key:
                        word_dist = rec.get('word_level_distribution', {})
                        if word_dist:
                            model_results.append(word_dist)
                        else:
                            tag = rec.get('classification', {}).get('tag_name')
                            if tag:
                                dist = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                                dist[tag] = 1
                                model_results.append(dist)

            if model_results:
                avg = {
                    'Negative': sum(r.get('Negative', 0) for r in model_results) / len(model_results),
                    'Neutral': sum(r.get('Neutral', 0) for r in model_results) / len(model_results),
                    'Positive': sum(r.get('Positive', 0) for r in model_results) / len(model_results)
                }
            else:
                avg = {'Negative': 0.33, 'Neutral': 0.34, 'Positive': 0.33}

            distribution[key] = {
                'name': models[key]['name'],
                'distribution': [
                    avg['Negative'], avg['Neutral'], avg['Positive']
                ],
                'counts': avg
            }

        return jsonify({
            "success": True,
            "text_analyzed": text,
            "results": results,
            "distribution": distribution,
            "all_models": {k: v['name'] for k, v in models.items()},
            "language_info": {
                "original_language": original_lang,
                "was_translated": original_lang != 'en',
                "user_preference": preferred_lang
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/distribution', methods=['GET'])
def get_distribution():
    try:
        model_filter = request.args.get('model', '').lower()
        ref = db.reference("results")
        results_data = ref.get()
        count = {'Negative': 0, 'Neutral': 0, 'Positive': 0}

        if results_data:
            for rec in results_data.values():
                if model_filter and rec.get('model_used', '').lower() != model_filter:
                    continue
                tag = rec.get('classification', {}).get('tag_name')
                if tag in count:
                    count[tag] += 1

        total = sum(count.values()) or 1
        distribution = [
            count['Negative'] / total,
            count['Neutral'] / total,
            count['Positive'] / total
        ]

        return jsonify({
            "success": True,
            "distribution": distribution,
            "total": total,
            "counts": count
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Process uploaded CSV file and perform sentiment analysis on each text entry"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.csv'):
            model_key = request.form.get('model', 'vader')

            try:
                df = pd.read_csv(file, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file, encoding='latin1')
                except Exception as e:
                    return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

            text_column = None
            for col in df.columns:
                if col.lower() in ['text', 'content', 'message', 'review', 'comment', 'tweet']:
                    text_column = col
                    break

            if text_column is None:
                if len(df.columns) > 0:
                    text_column = df.columns[0]
                else:
                    return jsonify({"error": "CSV file has no columns"}), 400

            results = []
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()
                if not text or text.lower() == 'nan':
                    continue

                result = process_text(text, model_key)
                entry = {
                    "original_row": idx,
                    "text": text,
                    "sentiment": result.get("sentiment_label", "Unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "model_used": model_key
                }
                for col in df.columns:
                    if col != text_column:
                        entry[col] = str(row[col])
                results.append(entry)

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                db_entry = {
                    "text": text,
                    "model_used": model_key,
                    "model_name": models[model_key]['name'],
                    "classification": {
                        "tag_name": result.get("sentiment_label", "Unknown"),
                        "confidence": round(result.get("confidence", 0.0), 4)
                    },
                    "timestamp": now,
                    "source": "batch_upload"
                }
                db.reference("results").push(db_entry)

            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            for r in results:
                s = r["sentiment"]
                if s in sentiment_counts:
                    sentiment_counts[s] += 1

            total = sum(sentiment_counts.values()) or 1
            distribution = {
                "Positive": sentiment_counts["Positive"] / total,
                "Neutral": sentiment_counts["Neutral"] / total,
                "Negative": sentiment_counts["Negative"] / total
            }

            return jsonify({
                "success": True,
                "results": results,
                "stats": {
                    "total_processed": len(results),
                    "sentiment_counts": sentiment_counts,
                    "distribution": distribution
                }
            })
        else:
            return jsonify({"error": "File must be a CSV"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def process_text(text, model_key):
    """Process a single text entry with the specified model"""
    if model_key not in models:
        return {"error": f"Unknown model: {model_key}"}

    try:
        # Detect language and translate if necessary
        processed_text, original_lang = detect_and_translate(text)

        if model_key == 'textblob':
            sentiment, confidence = analyze_textblob(processed_text)
        elif model_key == 'vader':
            sentiment, confidence = analyze_vader(processed_text)
        elif model_key.startswith(('bernoullinb', 'complementnb', 'gaussiannb')):
            sentiment, confidence = analyze_nb(processed_text, model_key)
        elif model_key.startswith('bert'):
            sentiment, confidence = analyze_bert(processed_text, model_key)
        else:
            return {"error": "Invalid model selection"}

        label = sentiment_labels.get(sentiment, "Unknown")
        return {
            "sentiment": sentiment,
            "sentiment_label": label,
            "confidence": float(confidence),
            "language_info": {
                "original_language": original_lang,
                "was_translated": original_lang != 'en'
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.route('/download_results', methods=['POST'])
def download_results():
    """Generate and download CSV file of analysis results"""
    try:
        data = request.json
        if not data or 'results' not in data:
            return jsonify({"error": "No results data provided"}), 400

        results = data['results']
        if not results or len(results) == 0:
            return jsonify({"error": "Empty results"}), 400

        output = io.StringIO()
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=sentiment_analysis_results.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/result')
def result():
    sort_order = request.args.get('sort', 'newest')
    filter_sentiment = request.args.get('filter', 'All')

    ref = db.reference("results")
    results_data = ref.get()
    formatted_results = []

    if results_data:
        for key, value in results_data.items():
            classification = value.get("classification", {})
            tag = classification.get("tag_name", "N/A")
            if filter_sentiment != "All" and tag != filter_sentiment:
                continue
            formatted_results.append({
                "id": key,
                "text": value.get("text", "N/A"),
                "model": value.get("model_used", "N/A"),
                "tag": tag,
                "confidence": classification.get("confidence", "N/A"),
                "timestamp": value.get("timestamp", "N/A")
            })

    if sort_order == "newest":
        formatted_results.sort(key=lambda x: x['timestamp'], reverse=True)
    else:
        formatted_results.sort(key=lambda x: x['timestamp'])

    return render_template('result.html',
                           results=formatted_results,
                           current_sort=sort_order,
                           current_filter=filter_sentiment)


@app.route('/get_results')
def get_results():
    sort_order = request.args.get('sort', 'newest')
    filter_sentiment = request.args.get('filter', 'All')

    ref = db.reference("results")
    results_data = ref.get()
    formatted_results = []

    if results_data:
        for key, value in results_data.items():
            classification = value.get("classification", {})
            tag = classification.get("tag_name", "N/A")
            if filter_sentiment != "All" and tag.lower() != filter_sentiment.lower():
                continue
            formatted_results.append({
                "id": key,
                "text": value.get("text", "N/A"),
                "model": value.get("model_used", "N/A"),
                "tag": tag,
                "confidence": classification.get("confidence", "N/A"),
                "timestamp": value.get("timestamp", "N/A")
            })

    formatted_results.sort(key=lambda x: x['timestamp'], reverse=(sort_order == "newest"))

    return jsonify(formatted_results)


@app.route('/delete/<result_id>', methods=['POST'])
def delete_result(result_id):
    try:
        db.reference(f"results/{result_id}").delete()
        return redirect(url_for('result'))
    except Exception as e:
        return f"<h1>Error deleting result</h1><p>{str(e)}</p>"


@app.route('/delete_all', methods=['POST'])
def delete_all_results():
    try:
        db.reference("results").delete()
        return redirect(url_for('result'))
    except Exception as e:
        return f"<h1>Error deleting all results</h1><p>{str(e)}</p>"


@app.route('/evaluation_report')
def evaluation_report():
    return render_template('evaluation_report.html')


@app.route('/classification_report/<model_name>')
def get_classification_report(model_name):
    try:
        model_name = model_name.lower()

        # Special case for bert training reports
        if model_name in ['bert_textblob.json', 'bert_vader.json', 'bert_chatgpt.json']:
            path = f'bert_training_report_{model_name.replace("bert_", "").replace(".json", "")}.json'
            if not os.path.exists(path):
                return jsonify({"error": "Report not found"}), 404
            with open(path, 'r') as f:
                report = json.load(f)
            return jsonify(report)

        # Handle baseline keys with (baseline) in filename
        baseline_map = {
            'bernoullinb_test_textblob': 'classification_report_bernoullinb_test_textblob(baseline).json',
            'complementnb_test_textblob': 'classification_report_complementnb_test_textblob(baseline).json',
            'gaussiannb_test_textblob': 'classification_report_gaussiannb_test_textblob(baseline).json',
            'bernoullinb_test_vader': 'classification_report_bernoullinb_test_vader(baseline).json',
            'complementnb_test_vader': 'classification_report_complementnb_test_vader(baseline).json',
            'gaussiannb_test_vader': 'classification_report_gaussiannb_test_vader(baseline).json',
            'bernoullinb_test_chatgpt': 'classification_report_bernoullinb_test_chatgpt(baseline).json',
            'complementnb_test_chatgpt': 'classification_report_complementnb_test_chatgpt(baseline).json',
            'gaussiannb_test_chatgpt': 'classification_report_gaussiannb_test_chatgpt(baseline).json',
        }
        if model_name in baseline_map:
            path = baseline_map[model_name]
            if not os.path.exists(path):
                return jsonify({"error": "Report not found"}), 404
            with open(path, 'r') as f:
                report = json.load(f)
            return jsonify(report)

        # Normal classification reports
        path = f'classification_report_{model_name}.json'
        if not os.path.exists(path):
            return jsonify({"error": "Report not found"}), 404
        with open(path, 'r') as f:
            report = json.load(f)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)