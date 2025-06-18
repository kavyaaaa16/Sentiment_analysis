**Project Overview**
The goal is to build a reliable sentiment classifier for book reviews, handling noisy, unlabeled real-world data. We begin with data collection and weak supervision, followed by iterative experimentation across ML and DL models, culminating in fine-tuned transformer-based architectures.

🧾 Pipeline Summary

1. 🕸️ Web Scraping + Dataset Collection
   
-Book reviews were scraped from Goodreads using requests + BeautifulSoup
-Also included a Kaggle dataset (~12,000 book reviews)
-Metadata extracted: review summary, number of votes, and genre

2. 🏷️ Weak Supervision for Labeling

-Since the data lacked sentiment labels, we used:
TextBlob
VADER
Hugging Face sentiment classifiers

A hybrid strategy was applied:
-Ensemble voting across models
-Confident scores boosted the label reliability

3. 📊 Classical ML Models
I used TF-IDF and CountVectorizer with:

-Logistic Regression
-Naive Bayes
-SVM

🔎 Performance (approx):

Model	Accuracy
Logistic Regression	~60–62%
SVM	~63–65%
Random Forest	~60%
Naive Bayes	~58–60%

These models struggled with subtle sentiment and sarcasm in long-form reviews.

4. 🔁 Deep Learning Models

✅ LSTM (Vanilla)
-Basic embedding + LSTM with padding
-Trained on weakly labeled reviews
-Accuracy: ~66–68%

✅ LSTM + GloVe
-100D GloVe vectors from Stanford
-Added dropout and bidirectional LSTM
-Accuracy: ~70%, improved contextual handling

5. 🤖 Transformer Models
✅ BERT (bert-base-uncased)
-Fine-tuned using Hugging Face Trainer
-More context-aware, but overfit slightly due to small dataset
-Accuracy: ~50%

✅ ELECTRA (bhadresh-savani/electra-base-emotion)
-Replaced classification head for 3-class task
-Trained for 4 epochs using manually cleaned dataset
-Final validation accuracy: 💥 78.54%

🧪 Final Evaluation
Model	Validation Accuracy
TF-IDF + SVM	~64%
LSTM + GloVe	~70%
BERT	~74%
ELECTRA (final)	78.54%


📁 Dataset Details
Combined reviews from:
-Goodreads (scraped)
-Kaggle open dataset

Final size: ~12,000 reviews

Labels: assigned via weak supervision

Held-out test set: 200 rows manually labeled

💾 Model Artifacts
Model: ./electra-bookreview-model

Tokenizer: ./electra-bookreview-tokenizer

Compatible with Hugging Face's from_pretrained()

🤝 Credits
Hugging Face Transformers

ELECTRA Emotion Model

Kaggle community

Goodreads data for real-world context


