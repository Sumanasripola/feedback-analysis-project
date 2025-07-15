import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg') # This line is CRUCIAL: Set Matplotlib backend to Agg (non-interactive)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64

# Download NLTK data (only needs to be run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# --- Configuration ---
DATA_DIR = 'data'
MODELS_DIR = 'models'
DATASET_PATH = os.path.join(DATA_DIR, 'feedback_data.csv') # The app expects this name
AUTHENTICITY_MODEL_PATH = os.path.join(MODELS_DIR, 'authenticity_model.joblib')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.joblib')
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Global variables for models and vectorizer ---
authenticity_model = None
sentiment_model = None
tfidf_vectorizer = None
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and preprocesses text for model input.
    - Converts to lowercase
    - Removes non-alphanumeric characters
    - Tokenizes
    - Removes stop words
    - Stems words
    """
    text = str(text).lower() # Ensure text is string and lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Model Training Function ---
def train_models():
    """
    Trains or retrains the authenticity and sentiment models.
    Loads data from DATASET_PATH, preprocesses it, and saves the trained models
    and TF-IDF vectorizer.
    """
    global authenticity_model, sentiment_model, tfidf_vectorizer

    # Initialize models to None before attempting to train
    authenticity_model = None
    sentiment_model = None
    tfidf_vectorizer = None

    print("Starting model training...")

    # Create a dummy dataset if it doesn't exist for initial run
    if not os.path.exists(DATASET_PATH):
        print(f"Creating dummy dataset at {DATASET_PATH}")
        dummy_data = {
            'feedback_text': [
                "This product is absolutely amazing! Highly recommend it to everyone.",
                "Terrible service, completely unhelpful staff. I'm very disappointed.",
                "It's okay, nothing special, just average performance.",
                "Fake review, this product is a scam. Do not trust it.",
                "I love this, it's perfect for my needs and works flawlessly.",
                "This is spam, don't trust this feedback. It's clearly malicious.",
                "Neutral comment about the delivery time. It arrived on schedule.",
                "The quality is poor and it broke quickly. Very frustrating.",
                "Excellent customer support, very responsive and solved my issue quickly.",
                "This feedback seems fabricated and unnatural. It reads like a bot.",
                "Fantastic experience, everything went smoothly from start to finish.",
                "Worst purchase ever, completely regret buying this. A total waste of money.",
                "It functions as expected, no complaints, no praise.",
                "This review is definitely not real, too generic and promotional.",
                "So happy with this! Exceeded my expectations in every way.",
                "Unsatisfactory, the features advertised are not present.",
                "The instructions were clear and the setup was easy.",
                "This is a fraudulent claim, the user doesn't exist.",
                "Decent product for the price, could be better but not bad.",
                "Absolutely dreadful, I want a refund immediately.",
                "The design is sleek and modern, very appealing.",
                "This feedback is suspicious, repetitive phrases used.",
                "Average product, nothing to write home about.",
                "Highly satisfied, will definitely buy again.",
                "This is a bot-generated review, ignore it.",
                "It's neither good nor bad, just functional.",
                "A truly positive experience, very impressed.",
                "Complete rubbish, avoid at all costs.",
                "The interface is intuitive and easy to navigate.",
                "This feedback is clearly automated, not from a real person."
            ],
            'is_fake': [ # Now using 'is_fake' as the authenticity column
                'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes',
                'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no',
                'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes'
            ],
            'sentiment': [ # Ensure lowercase for consistency
                'positive', 'negative', 'neutral', 'negative', 'positive', 'negative', 'neutral', 'negative', 'positive', 'negative',
                'positive', 'negative', 'neutral', 'negative', 'positive', 'negative', 'positive', 'negative', 'neutral', 'negative',
                'positive', 'negative', 'neutral', 'positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative'
            ]
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(DATASET_PATH, index=False)
        print("Dummy dataset created.")

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Initial DataFrame size: {len(df)} rows") # Added print
        print(f"Columns detected in CSV: {df.columns.tolist()}")
        print(f"First 5 rows of DataFrame:\n{df.head()}")
        df.columns = df.columns.str.strip()
        print(f"Columns after stripping spaces: {df.columns.tolist()}")

        # Drop rows where 'feedback_text' is NaN
        initial_rows = len(df)
        df.dropna(subset=['feedback_text'], inplace=True)
        rows_after_nan_drop = len(df)
        print(f"Rows after dropping NaN in feedback_text: {rows_after_nan_drop} (Dropped {initial_rows - rows_after_nan_drop} rows)")

        # Drop rows where 'feedback_text' is empty or only whitespace
        initial_rows_for_empty_check = len(df)
        df = df[df['feedback_text'].astype(str).str.strip() != '']
        rows_after_empty_drop = len(df)
        print(f"Rows after dropping empty/whitespace feedback_text: {rows_after_empty_drop} (Dropped {initial_rows_for_empty_check - rows_after_empty_drop} rows)")


        if df.empty:
            print("Dataset is empty after cleaning. Cannot train models.")
            return

        # Preprocess text for TF-IDF vectorizer
        df['processed_text'] = df['feedback_text'].apply(preprocess_text)
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        X = tfidf_vectorizer.fit_transform(df['processed_text'])

        # --- Authenticity Model Training ---
        if 'is_fake' in df.columns:
            df['is_genuine'] = df['is_fake'].astype(str).str.lower().apply(lambda x: 1 if x == 'no' else 0)
            if 'is_genuine' in df.columns and len(df['is_genuine'].unique()) >= 2:
                try:
                    y_authenticity = df['is_genuine']
                    X_train_auth, X_test_auth, y_train_auth, y_test_auth = train_test_split(
                        X, y_authenticity, test_size=0.2, random_state=42, stratify=y_authenticity
                    )
                    authenticity_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42, class_weight='balanced', alpha=0.0001)
                    authenticity_model.fit(X_train_auth, y_train_auth)

                    # Evaluate authenticity model
                    y_pred_auth = authenticity_model.predict(X_test_auth)
                    print(f"Authenticity Model Metrics:")
                    print(f"   Accuracy: {accuracy_score(y_test_auth, y_pred_auth):.4f}")
                    print(f"   Precision: {precision_score(y_test_auth, y_pred_auth):.4f}")
                    print(f"   Recall: {recall_score(y_test_auth, y_pred_auth):.4f}")
                    print(f"   F1-Score: {f1_score(y_test_auth, y_pred_auth):.4f}")
                except Exception as e:
                    print(f"Error fitting authenticity model: {e}. Authenticity model will not be available.")
                    authenticity_model = None
            else:
                print("Warning: 'is_genuine' column could not be created or lacks diversity. Authenticity model training skipped.")
                authenticity_model = None
        else:
            print("Warning: 'is_fake' column not found in dataset. Authenticity model training skipped.")
            authenticity_model = None


        # --- Sentiment Model Training ---
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype(str).str.lower()
            print(f"Unique values in 'sentiment' column (after lowercasing): {df['sentiment'].unique().tolist()}")
            if len(df['sentiment'].unique()) > 1:
                try:
                    y_sentiment = df['sentiment']
                    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
                        X, y_sentiment, test_size=0.2, random_state=42, stratify=y_sentiment
                    )
                    sentiment_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                    sentiment_model.fit(X_train_sent, y_train_sent)

                    # Evaluate sentiment model
                    y_pred_sent = sentiment_model.predict(X_test_sent)
                    print(f"Sentiment Model Metrics:")
                    print(f"   Accuracy: {accuracy_score(y_test_sent, y_pred_sent):.4f}")
                except Exception as e:
                    print(f"Error fitting sentiment model: {e}. Sentiment model will not be available.")
                    sentiment_model = None
            else:
                print("Warning: 'sentiment' column lacks diversity (only one unique value found). Sentiment model training skipped.")
                sentiment_model = None
        else:
            print("Warning: 'sentiment' column not found in dataset. Sentiment model training skipped.")
            sentiment_model = None

        # Save models and vectorizer
        joblib.dump(authenticity_model, AUTHENTICITY_MODEL_PATH)
        joblib.dump(sentiment_model, SENTIMENT_MODEL_PATH)
        joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
        print("Models and vectorizer saved successfully.")

    except Exception as e:
        print(f"Critical error during model training setup: {e}")
        # Ensure models are None if a critical error occurs during setup
        authenticity_model = None
        sentiment_model = None
        tfidf_vectorizer = None

# --- Load Models on Startup ---
def load_models():
    """
    Loads pre-trained models and TF-IDF vectorizer if they exist.
    If not, it triggers training.
    """
    global authenticity_model, sentiment_model, tfidf_vectorizer
    try:
        # Load models. If a model file is missing or corrupted,
        # the corresponding global variable will remain None.
        authenticity_model = joblib.load(AUTHENTICITY_MODEL_PATH)
        sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        print("Models and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("One or more model files not found. Training new models...")
        train_models()
    except Exception as e:
        print(f"Error loading models: {e}. Retraining...")
        train_models()

# Load models when the app starts
with app.app_context():
    load_models()

# --- Status Endpoint ---
@app.route('/status', methods=['GET'])
def status():
    """
    Provides a simple status check for the backend.
    """
    return jsonify({"status": "online"}), 200

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles both single text input and batch CSV file input for prediction.
    Stores input and results in the dataset.
    """
    if not tfidf_vectorizer: # TF-IDF vectorizer is essential for any prediction
        return jsonify({"error": "TF-IDF Vectorizer not loaded. Cannot perform predictions."}), 500

    new_feedback_entries = []
    response_data = []

    if 'file' in request.files:
        # Batch prediction from CSV file
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

        try:
            df_batch = pd.read_csv(file)
            if df_batch.empty:
                return jsonify({"error": "CSV file is empty or contains no valid data."}), 400

            if 'feedback_text' not in df_batch.columns:
                return jsonify({"error": "CSV must contain a 'feedback_text' column."}), 400

            # Ensure 'is_fake' and 'sentiment' columns are handled for saving back to dataset
            if 'is_fake' in df_batch.columns:
                df_batch['is_genuine'] = df_batch['is_fake'].astype(str).str.lower().apply(lambda x: 1 if x == 'no' else 0)
            else:
                df_batch['is_genuine'] = 1 # Default to genuine if column is missing for batch input

            if 'sentiment' in df_batch.columns:
                df_batch['sentiment'] = df_batch['sentiment'].astype(str).str.lower()
            else:
                df_batch['sentiment'] = 'neutral' # Default to 'neutral' if column is missing for batch input


            for index, row in df_batch.iterrows():
                feedback_text = str(row['feedback_text'])
                if not feedback_text.strip(): # Skip empty texts
                    continue
                processed_text = preprocess_text(feedback_text)
                vectorized_text = tfidf_vectorizer.transform([processed_text])

                # Authenticity prediction
                authenticity_label = "Unknown"
                authenticity_proba = 0.0
                is_genuine_pred = 0 # Default to false for storage if no model
                if authenticity_model:
                    is_genuine_pred = int(authenticity_model.predict(vectorized_text)[0])
                    authenticity_label = "Genuine" if is_genuine_pred == 1 else "False"
                    authenticity_proba = authenticity_model.predict_proba(vectorized_text)[0][is_genuine_pred]
                else:
                    # If authenticity model is not available, use the value from the batch file for storage
                    # and set label based on that.
                    is_genuine_pred = row['is_genuine'] # Use the 1/0 from batch
                    authenticity_label = "Genuine" if is_genuine_pred == 1 else "False"
                    authenticity_proba = 1.0 # Assume full confidence if using batch value directly


                # Sentiment prediction
                sentiment_label = "Neutral" # Default if sentiment model is not available
                sentiment_proba = 0.0
                if sentiment_model:
                    sentiment_pred = sentiment_model.predict(vectorized_text)[0]
                    sentiment_label = str(sentiment_pred)
                    if sentiment_pred in sentiment_model.classes_:
                        sentiment_proba = sentiment_model.predict_proba(vectorized_text)[0][list(sentiment_model.classes_).index(sentiment_pred)]
                else:
                    sentiment_label = row['sentiment'] # Use default from batch if no model

                entry = {
                    'feedback_text': feedback_text,
                    'is_fake': 'yes' if is_genuine_pred == 0 else 'no', # Store as 'yes'/'no' for consistency with CSV
                    'sentiment': sentiment_label
                }
                new_feedback_entries.append(entry)
                response_data.append({
                    'feedback_text': feedback_text,
                    'authenticity': authenticity_label,
                    'authenticity_confidence': round(authenticity_proba, 4),
                    'sentiment': sentiment_label,
                    'sentiment_confidence': round(sentiment_proba, 4)
                })

            # Append new entries to the dataset
            if new_feedback_entries:
                df_new = pd.DataFrame(new_feedback_entries)
                df_existing = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame(columns=['feedback_text', 'is_fake', 'sentiment'])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(DATASET_PATH, index=False)
                print(f"Appended {len(new_feedback_entries)} new entries to {DATASET_PATH}")

            return jsonify({"predictions": response_data, "message": f"Processed {len(response_data)} entries."})

        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return jsonify({"error": f"Error processing CSV file: {e}"}), 500

    else:
        # Single text input
        data = request.get_json()
        feedback_text = data.get('feedback_text')

        if not feedback_text or not feedback_text.strip():
            return jsonify({"error": "No feedback_text provided"}), 400

        processed_text = preprocess_text(feedback_text)
        vectorized_text = tfidf_vectorizer.transform([processed_text])

        # Authenticity prediction
        authenticity_label = "Unknown"
        authenticity_proba = 0.0
        is_genuine_pred = 0 # Default for storage if no model
        if authenticity_model:
            is_genuine_pred = int(authenticity_model.predict(vectorized_text)[0])
            authenticity_label = "Genuine" if is_genuine_pred == 1 else "False"
            authenticity_proba = authenticity_model.predict_proba(vectorized_text)[0][is_genuine_pred]
        else:
            authenticity_label = "Genuine" # Default to genuine in response if no model
            authenticity_proba = 1.0 # Assume full confidence


        # Sentiment prediction
        sentiment_label = "Neutral" # Default if sentiment model is not available
        sentiment_proba = 0.0
        if sentiment_model:
            sentiment_pred = sentiment_model.predict(vectorized_text)[0]
            sentiment_label = str(sentiment_pred)
            if sentiment_pred in sentiment_model.classes_:
                sentiment_proba = sentiment_model.predict_proba(vectorized_text)[0][list(sentiment_model.classes_).index(sentiment_pred)]
        else:
            sentiment_label = "Neutral" # Explicitly set to Neutral if model not available

        # Store the new entry in the dataset
        new_entry = {
            'feedback_text': feedback_text,
            'is_fake': 'yes' if is_genuine_pred == 0 else 'no', # Store as 'yes'/'no' for consistency with CSV
            'sentiment': sentiment_label
        }
        try:
            df_existing = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else pd.DataFrame(columns=['feedback_text', 'is_fake', 'sentiment'])
            df_combined = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
            df_combined.to_csv(DATASET_PATH, index=False)
            print(f"Appended 1 new entry to {DATASET_PATH}")
        except Exception as e:
            print(f"Error appending single entry to CSV: {e}")

        return jsonify({
            'feedback_text': feedback_text,
            'authenticity': authenticity_label,
            'authenticity_confidence': round(authenticity_proba, 4),
            'sentiment': sentiment_label,
            'sentiment_confidence': round(sentiment_proba, 4)
        })

# --- Retrain Endpoint ---
@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Triggers the retraining of models with the current dataset.
    """
    try:
        train_models()
        return jsonify({"message": "Models retrained successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrain models: {e}"}), 500

# --- Report Data Endpoint ---
@app.route('/report_data', methods=['GET'])
def get_report_data():
    """
    Generates structured report data including trends, keyword analysis,
    and data for visualizations.
    """
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found for reporting."}), 404

    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Initial DataFrame size for report: {len(df)} rows") # Added print
        df.dropna(subset=['feedback_text'], inplace=True)
        print(f"DataFrame size after dropping NaN in feedback_text: {len(df)} rows") # Added print
        df = df[df['feedback_text'].astype(str).str.strip() != '']
        print(f"DataFrame size after dropping empty/whitespace feedback_text: {len(df)} rows") # Added print


        if df.empty:
            return jsonify({"error": "Dataset is empty after cleaning. No report to generate."}), 400

        # Convert 'is_fake' from 'yes'/'no' to 1/0 for 'is_genuine' logic
        if 'is_fake' in df.columns:
            df['is_genuine'] = df['is_fake'].astype(str).str.lower().apply(lambda x: 1 if x == 'no' else 0)
        else:
            print("Warning: 'is_fake' column not found for reporting. Authenticity data might be incomplete.")
            df['is_genuine'] = 0 # Default to 0 if missing for reporting purposes

        # Convert 'sentiment' to lowercase to ensure consistency for reporting
        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype(str).str.lower()
        else:
            print("Warning: 'sentiment' column not found for reporting. Sentiment data might be incomplete.")
            df['sentiment'] = 'neutral' # Default if missing for reporting purposes


        # Re-apply preprocessing for word cloud and keyword analysis,
        # as the original 'processed_text' might not exist if loaded from CSV directly
        df['processed_text'] = df['feedback_text'].apply(preprocess_text)

        # --- Authenticity Distribution ---
        authenticity_counts = df['is_genuine'].value_counts().rename(index={1: 'Genuine', 0: 'False'}).to_dict()
        authenticity_distribution = [
            {'label': 'Genuine', 'count': authenticity_counts.get('Genuine', 0)},
            {'label': 'False', 'count': authenticity_counts.get('False', 0)}
        ]

        # --- Sentiment Distribution (only for genuine feedback) ---
        genuine_feedback_df = df[df['is_genuine'] == 1]
        sentiment_counts = genuine_feedback_df['sentiment'].value_counts().to_dict()
        sentiment_distribution = [
            {'label': 'Positive', 'count': sentiment_counts.get('positive', 0)},
            {'label': 'Negative', 'count': sentiment_counts.get('negative', 0)},
            {'label': 'Neutral', 'count': sentiment_counts.get('neutral', 0)}
        ]

        # --- Frequent Keywords (from genuine feedback) ---
        all_genuine_text = " ".join(genuine_feedback_df['processed_text'].dropna().tolist())
        words = re.findall(r'\b\w+\b', all_genuine_text.lower())
        # Filter out short words and common stop words again (though already stemmed)
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        word_freq = pd.Series(filtered_words).value_counts().head(20).to_dict() # Top 20 keywords

        # --- Word Cloud Image (base64 encoded) ---
        wordcloud_img_base64 = None
        if all_genuine_text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_genuine_text)
            img_buffer = io.BytesIO()
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            img_buffer.seek(0)
            wordcloud_img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # --- Trends (e.g., average sentiment over time if timestamps were available) ---
        total_feedback = len(df)
        genuine_percentage = (authenticity_counts.get('Genuine', 0) / total_feedback * 100) if total_feedback else 0
        false_percentage = (authenticity_counts.get('False', 0) / total_feedback * 100) if total_feedback else 0

        report = {
            "total_feedback": total_feedback,
            "genuine_percentage": round(genuine_percentage, 2),
            "false_percentage": round(false_percentage, 2),
            "authenticity_distribution": authenticity_distribution,
            "sentiment_distribution": sentiment_distribution,
            "frequent_keywords": [{"word": k, "count": v} for k, v in word_freq.items()],
            "wordcloud_image": f"data:image/png;base64,{wordcloud_img_base64}" if wordcloud_img_base64 else None,
            "summary": "This report provides an overview of feedback authenticity and sentiment. "
                       "The authenticity distribution shows the proportion of genuine vs. false feedback. "
                       "Sentiment analysis is performed on genuine feedback, categorizing it into positive, negative, and neutral. "
                       "Frequent keywords highlight common themes in genuine feedback."
        }
        return jsonify(report)

    except Exception as e:
        print(f"Error generating report data: {e}")
        return jsonify({"error": f"Failed to generate report data: {e}"}), 500

# --- Run the Flask app ---
if __name__ == '__main__':
    # Initial training or loading models when the script is run directly
    # This also handles creating the dummy dataset if it doesn't exist
    load_models()
    app.run(debug=True, port=5000)
