from flask import Flask, render_template, request, jsonify
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch
import csv
app = Flask(__name__)


books = []
with open('C:/Users/njsha/OneDrive/Desktop/Book Review/goodreads_books.csv', newline='', encoding='utf-8') as csvfile:

    reader = csv.DictReader(csvfile)
    for row in reader:
        books.append(row)


model_path = "C:/Users/njsha/OneDrive/Desktop/Book Review/model"
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraForSequenceClassification.from_pretrained(model_path)
model.eval()

# Routes for pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyse_review')
def analyse_review():
    return render_template('analyse_review.html')

@app.route('/find_book')
def find_book():
    return render_template('find_book.html')

# Sentiment prediction API
@app.route('/predict_review', methods=['POST'])
def predict_review():
    data = request.get_json()
    review_text = data.get('review', '')

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    labels = ['Positive', 'Neutral', 'Negative']  
    prediction = labels[pred.item()]
    confidence = conf.item()

    return jsonify({'prediction': prediction, 'confidence': confidence})

# Sample book search API (mock)
@app.route('/search_books', methods=['POST'])
def search_books():
    data = request.get_json()
    query = data.get('query', '').lower()

    filtered = [book for book in books if query in book['title'].lower()]
    
    return jsonify({'books': filtered})


# Sample book review sentiment API (mock)
@app.route('/get_book_review', methods=['POST'])
def get_book_review():
    data = request.get_json()
    title = data.get('title', '')

    # Find the book in your dataset
    book = next((b for b in books if b['title'].lower() == title.lower()), None)
    if not book:
        return jsonify({'error': 'Book not found'}), 404

    # Get reviews from the actual dataset fields
    sample_reviews = [book.get('review1', ''), book.get('review2', ''), book.get('review3', '')]
    sample_reviews = [r.strip() for r in sample_reviews if r.strip()]  # clean and skip empty

    if not sample_reviews:
        return jsonify({'sentiment': 'Unknown', 'recommendation': 'No reviews available for this book.'})

    # Predict sentiment for each review using your model
    sentiments = []
    for review_text in sample_reviews:
        inputs = tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        labels = ['Negative', 'Neutral', 'Positive']  # adjust this to match your model's training
        sentiment = labels[pred.item()]
        sentiments.append(sentiment)

    # Aggregate the sentiment results
    from collections import Counter
    sentiment_count = Counter(sentiments)
    top_sentiment = sentiment_count.most_common(1)[0][0]

    # Generate recommendation
    if top_sentiment == 'Positive':
        recommendation = "This book is worth reading!"
    elif top_sentiment == 'Neutral':
        recommendation = "This book has mixed reviews. You may want to check it out yourself."
    else:
        recommendation = "This book might not be for everyone."

    return jsonify({'sentiment': top_sentiment, 'recommendation': recommendation})



if __name__ == '__main__':
    app.run(debug=True)
