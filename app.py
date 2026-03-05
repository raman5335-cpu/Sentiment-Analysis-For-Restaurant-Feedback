from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import os
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import joblib
import torch
from langdetect import detect, LangDetectException
from googletrans import Translator
import emoji

# Initialize sentiment analysis pipeline
sentiment_classifier = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment", 
    device=0 if torch.cuda.is_available() else -1, 
    truncation=True
)

# Initialize Google Translator
translator = Translator()

# Flask app setup
app = Flask(__name__)

# Caching setup using joblib
translation_cache = {}
# Define keywords for categorization
categories = {
    'Food': ['food', 'taste', 'dish', 'meal', 'flavor', 'spice', 'cuisine'],
    'Interior': ['interior', 'decor', 'ambiance', 'atmosphere', 'design', 'furniture'],
    'Service': ['service', 'staff', 'waiter', 'waitress', 'customer service', 'hospitality'],
    'Price': ['price', 'cost', 'expensive', 'cheap', 'value for money', 'affordable', 'worth'],
    'Hygiene': ['clean', 'hygiene', 'sanitation', 'dirty', 'neat', 'tidy'],
    'Menu': ['menu', 'variety', 'selection', 'options', 'dishes', 'specials', 'item'],
    'Location': ['location', 'parking', 'accessibility', 'nearby', 'distance', 'spot'],
    'Drinks': ['drink', 'beverage', 'wine', 'cocktail', 'juice', 'beer', 'alcohol'],
}

def translate_review(review):
    review_hash = hashlib.md5(review.encode()).hexdigest()

    # Return cached translation if it exists
    if review_hash in translation_cache:
        return translation_cache[review_hash]
    
    try:
        # Detect the language of the review
        detected_lang = detect(review)
    except LangDetectException:
        # If detection fails, assume it's English
        detected_lang = 'en'
    
    # Skip translation for English reviews
    if detected_lang == 'en':
        translation_cache[review_hash] = review
        return review

    # Translate if the detected language is not English
    try:
        translated_review = translator.translate(review, src=detected_lang, dest='en').text
        translation_cache[review_hash] = translated_review
        return translated_review
    except Exception as e:
        print(f"Translation failed for {review} in {detected_lang}: {e}")
    
    # Fallback: If translation fails, return original review
    translation_cache[review_hash] = review
    return review

positive_emojis = {
    "😊", "😍", "👍", "🎉", "😁", "😀", "😃", "😄", "😆", "😇", "🥰", "😘", "😋", "😜", "😎",
    "🤩", "🥳", "💖", "💗", "💓", "💞", "💕", "💙", "💚", "💛", "💜", "❤️", "🤗", "👏", "🙌",
    "🌟", "✨", "🔥", "🎊", "🎈", "🏆", "🥂", "🍾", "🎶", "💃", "🕺", "🎵", "😊", "😁", "😺",
    "😻", "😽", "🥇", "🥰", "🤝"
}

negative_emojis = {
    "😢", "😡", "👎", "💔", "😭", "😞", "😖", "😣", "😩", "😤", "😠", "😾", "😿", "🙀", "☹️",
    "🙁", "😕", "😔", "😓", "🤦", "🤷", "🤨", "🥵", "🥶", "🤒", "🤕", "😰", "😨", "😧", "😱",
    "💢", "🚫", "⚠️", "❌", "💀", "👀", "😶", "😬", "😵", "😳", "🥀", "🖤", "🤢", "🤮"
}


def analyze_emoji_sentiment(review):
    extracted_emojis = [c for c in review if c in emoji.EMOJI_DATA]

    if not extracted_emojis:
        return "Neutral", 0.5  # Default to Neutral with a neutral score

    emoji_scores = []

    for em in extracted_emojis:
        if em in positive_emojis:
            emoji_scores.append(0.8)  # Assign a strong positive score
        elif em in negative_emojis:
            emoji_scores.append(0.2)  # Assign a strong negative score

    if not emoji_scores:
        return "Neutral", 0.5  # No sentiment-related emojis found

    avg_score = sum(emoji_scores) / len(emoji_scores)

    # Determine sentiment based on the average score
    if avg_score >= 0.6:
        return "Positive", avg_score
    elif avg_score <= 0.4:
        return "Negative", avg_score
    else:
        return "Neutral", avg_score


def analyze_text_sentiment(review):
    result = sentiment_classifier([review])[0]
    label = result['label']
    score = result['score']
    if '4 stars' in label or '5 stars' in label:
        sentiment = "Positive"
    elif '1 star' in label or '2 stars' in label:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, score


def categorize_review(review):
    review_lower = review.lower()
    matched_categories = [category for category, keywords in categories.items() if any(keyword in review_lower for keyword in keywords)]
    return matched_categories if matched_categories else ['General']

def process_reviews(reviews):
    results_dict = {category: [] for category in categories.keys()}
    results_dict['General'] = []

    batch_size = 20
    with ThreadPoolExecutor(max_workers=4) as executor:
        translated_batches = executor.map(lambda r: [translate_review(review) for review in r],
                                          [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)])

        for i, translated_batch in enumerate(translated_batches):
            original_batch = reviews[i * batch_size: (i + 1) * batch_size]
            for original_review, translated_review in zip(original_batch, translated_batch):
                emoji_sentiment, emoji_score = analyze_emoji_sentiment(original_review)

                # If the review contains only emojis, rely solely on emoji sentiment
                if all(c in emoji.EMOJI_DATA for c in original_review.strip()):
                    sentiment = emoji_sentiment
                    text_score = None  # No text score for emoji-only reviews
                    final_score = emoji_score  # Only emoji score
                else:
                    text_sentiment, text_score = analyze_text_sentiment(translated_review)

                    # Final decision: Weigh emoji sentiment when text is neutral or conflicting
                    if text_sentiment == "Neutral":
                        sentiment = emoji_sentiment
                        final_score = emoji_score
                    elif text_sentiment == "Positive" and emoji_sentiment == "Negative":
                        sentiment = "Negative"  # Prioritize negative emoji sentiment
                        final_score = min(text_score, emoji_score)
                    else:
                        sentiment = text_sentiment
                        final_score = text_score  # Keep text sentiment score

                review_categories = categorize_review(translated_review)
                for category in review_categories:
                    results_dict[category].append({
                        "Review": original_review,
                        "Translated Review": translated_review if original_review != translated_review else None,
                        "Sentiment": sentiment,
                        "Score": round(final_score, 2)  # Ensure score is rounded for readability
                    })

    return results_dict




def plot_sentiment_distribution(sentiment_counts, title, chart_type='bar'):
    plt.figure(figsize=(10, 6))
    
    if chart_type == 'bar':
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
    elif chart_type == 'pie':
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
        plt.ylabel('')  # Hide y-axis label for pie chart
    
    plt.title(f'{title} Sentiment Distribution')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'review' in request.form:
        review = request.form['review']
        
        # Analyze both text and emoji sentiment
        text_sentiment, text_score = analyze_text_sentiment(review)
        emoji_sentiment = analyze_emoji_sentiment(review)

        # Improve sentiment decision-making for negative emojis
        if text_sentiment == "Neutral":
            sentiment = emoji_sentiment
        elif text_sentiment == "Positive" and emoji_sentiment == "Negative":
            sentiment = "Negative"  # Prioritize negative emoji sentiment if present
        else:
            sentiment = text_sentiment

        # Render results with sentiment score and detected emoji sentiment
        return render_template(
            'index.html',
            review=review,
            sentiment=sentiment,
            text_score=round(text_score, 2),  # Round score for display
            emoji_sentiment=emoji_sentiment  # Show emoji sentiment separately
        )

    elif 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if 'Review' not in df.columns:
                return render_template('index2.html', error="CSV file must contain a 'Review' column.")

            reviews = df['Review'].tolist()
            categorized_results = process_reviews(reviews)

            visualizations = {}
            summary = {}

            for category, results in categorized_results.items():
                if results:
                    results_df = pd.DataFrame(results)
                    sentiment_counts = results_df['Sentiment'].value_counts()

                    bar_chart = plot_sentiment_distribution(sentiment_counts, category.capitalize(), 'bar')
                    pie_chart = plot_sentiment_distribution(sentiment_counts, category.capitalize(), 'pie')

                    visualizations[category] = {'bar': bar_chart, 'pie': pie_chart}
                    summary[category] = {
                        'total': len(results),
                        'positive': sentiment_counts.get('Positive', 0),
                        'neutral': sentiment_counts.get('Neutral', 0),
                        'negative': sentiment_counts.get('Negative', 0),
                    }

            return render_template(
                'index.html',
                visualizations=visualizations,
                summary=summary,
                categorized_results=categorized_results
            )

    return redirect(url_for('index'))


if __name__ == '__main__':
    try:
        if os.path.exists('translation_cache.pkl'):
            translation_cache = joblib.load('translation_cache.pkl')
        else:
            translation_cache = {}
    except (EOFError, FileNotFoundError):
        translation_cache = {}  # Fallback to an empty cache if loading fails

    try:
        app.run(debug=True)
    finally:
        joblib.dump(translation_cache, 'translation_cache.pkl')


