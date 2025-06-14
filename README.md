# Fake News Detection using BERT and FastAPI

This project is a part of the COMP8420 (Artificial Intelligence for Text and Vision) Major Assignment at Macquarie University, Session 1, 2025.
The Full project Can acceess Google Drive link - https://drive.google.com/drive/folders/1OdapM4QAEdtdHXGU4ehxR9b5_fxAnAnM?usp=sharing
## 🔍 Project Overview

Fake news and misinformation have become major threats in modern society. This project develops a binary text classification system to identify whether a news claim is **True** or **False** using deep learning techniques.

Key Components:
- **BERT (bert-base-uncased)** for classification
- **LIAR Dataset** for political fact-checking
- **FastAPI** for real-time prediction through an HTTP endpoint
- **(Optional)** ChatGPT API used for limited comparison with LLMs

## 📁 Project Structure

```
.
├── fake-news-detection.ipynb    # Jupyter notebook with data prep, training, evaluation
├── main.py                      # FastAPI app for prediction API
├── saved_bert_model/            # Saved model and tokenizer
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## ⚙️ Setup Instructions

1. **Clone this repo**
```bash
git clone https://github.com/yourusername/fake-news-bert-api.git
cd fake-news-bert-api
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the API**
```bash
uvicorn main:app --reload
```

Visit the Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🧪 Sample API Request

```json
POST /predict
{
  "text": "The economy grew 10% last quarter according to official statistics."
}
```

## 🧠 Model Performance

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 63%    |
| Precision     | 60%    |
| Recall        | 63%    |
| F1 Score      | 61%    |

## 📦 Dataset

- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- Filtered for binary classification: `True` vs `False`

## 🤝 Contribution

Developed by:
- **Fardin Haque**
- **Ruijie Du**

## 📄 License

This project is for educational purposes under Macquarie University’s COMP8420 coursework.
