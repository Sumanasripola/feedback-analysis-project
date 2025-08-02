# Feedback Authenticity & Sentiment Analysis

An AI-powered Flask system designed to classify textual feedback for authenticity (genuine vs. false) and sentiment (positive, negative, neutral). This project provides a robust API for real-time analysis, batch processing, model retraining, and generating insightful analytics reports.

## Project Description

This project delivers a robust **Feedback Authenticity & Sentiment Analysis system**, powered by machine learning. Designed as a Flask backend, it provides critical insights into user feedback by automatically classifying submissions as either **genuine or false** and determining their **sentiment (positive, negative, or neutral)**.

In today's digital landscape, distinguishing authentic feedback from spam, bot-generated content, or malicious reviews is crucial for businesses and platforms. Simultaneously, understanding the emotional tone of customer input allows for data-driven improvements and better customer engagement.

This system addresses these challenges by:
* **Automating the detection of fake feedback**, helping to maintain data integrity and trustworthiness.
* **Providing immediate sentiment analysis**, enabling quick responses to customer needs and identifying areas for improvement.
* **Offering comprehensive analytics reports**, including key trends, word clouds, and keyword analysis, to transform raw feedback into actionable business intelligence.

Whether you're looking to enhance customer service, protect brand reputation, or gain deeper insights from user data, this project offers a powerful and extensible solution for intelligent feedback management.

## Table of Contents

-   [Features](#features)
-   [Algorithms Used](#algorithms-used)
-   [Technologies Used](#technologies-used)
-   [Setup and Installation](#setup-and-installation)
-   [Usage](#usage)
-   [API Endpoints](#api-endpoints)
-   [Data Format](#data-format)
-   [Contributing](#contributing)
-   [License](#license)

## Features

* **Real-time Prediction**: Analyze individual feedback texts for authenticity and sentiment instantly.
* **Batch Analysis**: Process multiple feedback entries simultaneously by uploading a CSV file.
* **Model Retraining**: Dynamically retrain the authenticity and sentiment models with new or updated data to improve performance.
* **Analytics Reports**: Generate comprehensive reports including authenticity distribution, sentiment breakdown, frequent keywords, and word clouds for genuine feedback.
* **Data Persistence**: Automatically saves new feedback and predictions to a CSV dataset, allowing continuous learning and analysis.
* **Robust Preprocessing**: Includes text cleaning, stop-word removal, and stemming for effective NLP.

## Algorithms Used

This project leverages the following machine learning algorithms and techniques:

* **TF-IDF Vectorization (`TfidfVectorizer`)**: Converts text data into numerical feature vectors, weighting words based on their importance in a document and across the corpus.
* **Authenticity Model (`SGDClassifier`)**: A linear classifier (Stochastic Gradient Descent) trained to distinguish between genuine and false feedback. It's efficient for large-scale text classification.
* **Sentiment Model (`LogisticRegression`)**: A linear model used for multi-class classification to categorize feedback into positive, negative, or neutral sentiments.

## Technologies Used

* **Python 3.x**
* **Flask**: Web framework for building the API.
* **Flask-CORS**: Enables Cross-Origin Resource Sharing for frontend integration.
* **Pandas**: Data manipulation and analysis (for CSV handling).
* **scikit-learn**: Machine learning library for TF-IDF, `SGDClassifier`, and `LogisticRegression`.
* **NLTK (Natural Language Toolkit)**: For text preprocessing (stopwords, stemming).
* **Joblib**: For efficient saving and loading of Python objects (models, vectorizer).
* **Matplotlib & Seaborn**: For generating visualizations (e.g., word clouds).
* **WordCloud**: Library for creating word cloud images.
* **Node.js & npm**: (Assumed for frontend development, typically with React, Vue, or Angular)

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment for the backend (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the backend virtual environment:**
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is not provided, you can create one with `pip freeze > requirements.txt` after manually installing the listed dependencies, or simply install them one by one: `pip install Flask Flask-Cors pandas scikit-learn nltk joblib matplotlib seaborn wordcloud`)

5.  **Download NLTK data:**
    The application will attempt to download `stopwords` and `punkt` automatically on first run if not found. If you encounter issues, you can manually download them:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## Usage

### Running the Backend

Assuming your backend code (`app.py`) is located in a directory named `backend` at the root of your project:

1.  **Ensure your backend virtual environment is activated.**
2.  **Navigate into the backend directory:**
    ```bash
    cd backend
    ```
3.  **Run the Flask backend server:**
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000` (or `http://localhost:5000`).

    * Upon the first run, the application will automatically create a `data` folder with a `feedback_data.csv` (if it doesn't exist) and train the machine learning models, saving them in the `models` folder. This might take a moment.
    * Subsequent runs will load the saved models, which is much faster.

### Running the Frontend

Assuming your frontend code is located in a directory named `frontend` at the root of your project:

1.  **Navigate into the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install frontend dependencies (if you haven't already):**
    ```bash
    npm install
    ```
3.  **Start the frontend development server:**
    ```bash
    npm run dev
    ```
    This command will typically start a development server (e.g., on `http://localhost:3000` or `http://localhost:5173`, depending on the frontend framework) and open the application in your browser.

### Interacting with the API

You can use tools like Postman, curl, or integrate with your running frontend application to send requests to the API endpoints.

## API Endpoints

The backend exposes the following API endpoints:

### 1. `/status` (GET)

* **Description**: Checks if the backend server is running.
* **Response**:
    ```json
    {
      "status": "online"
    }
    ```

### 2. `/predict` (POST)

* **Description**: Analyzes feedback text for authenticity and sentiment. Supports single text input or batch CSV file upload.
* **Request Body (Single Text)**:
    ```json
    {
      "feedback_text": "This product is amazing and works perfectly!"
    }
    ```
* **Request Body (CSV File Upload)**:
    * Send a `multipart/form-data` request with a file field named `file`.
    * The CSV file **must** contain a column named `feedback_text`. It can optionally contain `is_fake` and `sentiment` columns for initial data.
    * Example `test.csv` content:
        ```csv
        feedback_text
        "This is a great product."
        "Terrible experience, very slow delivery."
        "Limited time offer! Buy now!"
        ```
* **Response (Single Text)**:
    ```json
    {
      "feedback_text": "This product is amazing and works perfectly!",
      "authenticity": "Genuine",
      "authenticity_confidence": 0.98,
      "sentiment": "positive",
      "sentiment_confidence": 0.95
    }
    ```
* **Response (Batch CSV)**:
    ```json
    {
      "predictions": [
        {
          "feedback_text": "This is a great product.",
          "authenticity": "Genuine",
          "authenticity_confidence": 0.97,
          "sentiment": "positive",
          "sentiment_confidence": 0.93
        },
        // ... more predictions
      ],
      "message": "Processed X entries."
    }
    ```

### 3. `/retrain` (POST)

* **Description**: Triggers a retraining of both the authenticity and sentiment models using the current `feedback_data.csv`.
* **Request Body**: Empty JSON object `{}`.
* **Response**:
    ```json
    {
      "message": "Models retrained successfully!"
    }
    ```
    or
    ```json
    {
      "error": "Failed to retrain models: <error_details>"
    }
    ```

### 4. `/report_data` (GET)

* **Description**: Generates an analytics report based on the `feedback_data.csv`.
* **Response**:
    ```json
    {
      "total_feedback": 5000,
      "genuine_percentage": 70.25,
      "false_percentage": 29.75,
      "authenticity_distribution": [
        {"label": "Genuine", "count": 3512},
        {"label": "False", "count": 1488}
      ],
      "sentiment_distribution": [
        {"label": "Positive", "count": 1800},
        {"label": "Negative", "count": 1000},
        {"label": "Neutral", "count": 712}
      ],
      "frequent_keywords": [
        {"word": "great", "count": 150},
        {"word": "product", "count": 120}
        // ... more keywords
      ],
      "wordcloud_image": "data:image/png;base64,<base64_encoded_image_data>",
      "summary": "This report provides an overview of feedback authenticity and sentiment..."
    }
    ```

## Data Format

The core dataset for this project is `feedback_data.csv`, located in the `data/` directory. It should have the following columns:

* `feedback_text`: (String) The actual text content of the feedback.
* `is_fake`: (String) Label indicating authenticity. Expected values: `"yes"` or `"no"`.
* `sentiment`: (String) Label indicating sentiment. Expected values: `"positive"`, `"negative"`, or `"neutral"`.

**Example `feedback_data.csv` entry:**

```csv
feedback_text,is_fake,sentiment
"The product arrived quickly and was exactly as described. Very happy with the purchase.",no,positive
"Customer service was terrible, they never responded to my emails. Very frustrating experience.",no,negative
"It's an average device, nothing groundbreaking, but it gets the job done.",no,neutral
"This is an amazing offer! Limited time only! Buy now and get a discount!",yes,positive
