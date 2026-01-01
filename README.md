# ⚖️ Hoax vs Real News Classifier

This repository contains a Machine Learning-based web application designed to detect and classify news articles as either **Hoax (Fake)** or **Real (Fact)**. Built using **Streamlit**, the application integrates a Natural Language Processing (NLP) pipeline to provide accurate credibility assessment, complete with linguistic analysis and confidence scoring.

---

## 📌 Problem Statement

The spread of misinformation in United States becomes massve.Thousands of hoax news articles are shared daily on social media platforms, leading to confusion and mistrust among the public. for this reason, it is crucial to develop an automated system that can accurately classify news articles as hoax or real.

This project aims to:

- Develop a machine learning model that can classify news articles as hoax or real based on their content.
- Analyze linguistic and contextual features that differentiate hoax news from real news.
- Evaluate the model's performance using **recall** as the primary metric to ensure that hoax news articles are correctly identified. 

---

## 🧠 Features (Webapp)

- Real-Time Classification: Instantly predicts whether a news article is likely a Hoax or Real using a trained ML model.
- Interactive Dashboard: Built with Streamlit for a smooth, responsive, and modern user interface.
- Confidence Meter: Visualizes the model's certainty level using an interactive gauge chart (Plotly).
- Linguistic Analysis: Provides deep insights such as word frequency distribution and text statistics using NLTK.

---

## 📊 Features (Dataset)

The dataset contains the following features:

| Feature Name        | Description                                                   | Type        |
|---------------------|---------------------------------------------------------------|-------------|
| `title`             | The title of the news article                                 | Text|
| `text`              | the body content of the news article                          | Text |
| `subject`           | The subject category of the news article                      | Categorical |
| `date`              | The publication date of the news article                      | DateTime |
| `label`             | The label indicating whether the news is hoax or real         | Categorical |

---

## 🛠️ Tech Stack

### Frontend:

- **Language:** Python
- **Framework:** Streamlit

### Data Science & ML:

- **Data Handling:** Pandas
- **Numerical Computing:** Numpy
- **Text Processing:** NLTK
- **Data Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning Algorithms:** scikit-learn, XGBoost, LightGBM, CatBoost

### Experiments:

- **Experimentation:** Jupyter Notebook

---

## 📁 Project Structure

```bash
hoax-news-detection-webapp/
│
├── artifacts/              # Serialized model and preprocessors
│   ├── model.pkl           # Trained Classification Model
│   ├── scaler.pkl          # Scaler object for numerical features
│   └── vectorizer.pkl      # Vectorizer for text features
│
├── data/                   # Dataset directory
│   ├── fake.csv            # Raw dataset containing fake news
│   └── true.csv            # Raw dataset containing real news
│
├── Notebooks/              # Data Science workspace
│   ├── eda.ipynb           # Exploratory Data Analysis & Visualization
│   └── modelling.ipynb     # Model training, tuning, and evaluation
│
├── services/               # Backend logic modules
│   ├── predictor.py        # Logic to load artifacts and generate predictions
│   └── preprocessing.py    # Pipeline for text preprocessing and feature extraction
│
├── .gitignore
├── app.py                  # Main Streamlit application entry point
├── README.md
└── requirements.txt        # Python dependencies list
```

---

## 🔁 Workflow

1. **Data Collection**: The raw datasets (`fake.csv` and `true.csv`) are loaded and merged into a balanced dataset.
2. **Exploratory Data Analysis (EDA)**: Conducted in `eda.ipynb` to understand data distributions and text and label characteristics
3. **Data Preprocessing**: Handled in `eda.ipynb` and `preprocessing.py` to clean text data,vectorize text using TF-IDF, and scale numerical features.
4. **Model Training**: Experiments with different algorithms in `model.ipynb`, leading to the selection of a CatBoost Classifier based on recall performance.
5. **Model Evaluation**: Performance metrics and confusion matrix visualizations are generated to assess model effectiveness.
6. **Model Deployment**: The trained model and preprocessors are saved in the `artifacts/` directory and integrated into the Streamlit app (`app.py`) for real-time predictions.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[Fake vs Real News](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data)

We would like to acknowledge and thanks to the dataset creator for making this resource publicly available for research and educational use.

---

## 🚀 How to Run

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/abidalfrz/hoax-news-detection-webapp.git
cd hoax-news-detection-webapp
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. Run the Application

Go to the project directory and run the following command in your terminal:

```bash
streamlit run app.py
```

### 5. Access the Web App

Open your web browser and navigate to `http://localhost:8501` to access the Hoax vs Real News Classifier web app.

1. Input a news article in the provided text area.
2. Click the "ANALYZE CONTENT" button to see the classification result along with confidence scores and linguistic analysis.

---


