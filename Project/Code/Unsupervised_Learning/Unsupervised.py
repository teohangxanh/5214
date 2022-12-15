import pandas as pd
import numpy as np
from collections import Counter
import spacy
import seaborn as sb
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def not_outliers(data, threshold=3):
    '''Returns a list of boolean values indicating if a datum is not an outlier'''
    mean = np.mean(data)
    std = np.std(data)
    return [(datum - mean) / std <= abs(threshold) for datum in data]


# Get n-grams token
def get_most_commons(text, n_gram=1):
    '''Return a Counter of n_grams-model tokens'''
    blobs = TextBlob(text)
    n_grams = [' '.join(i) for i in blobs.ngrams(n=n_gram)]
    counter = Counter(n_grams).most_common(20)
    return counter


def to_df(c, col1, col2):
    '''Returns a DataFrame of most-common-tokens with two columns'''
    return pd.DataFrame(data=c, columns=[col1, col2])


def extract_text(t: list):
    '''Extracts a joined text from a list of tuples'''
    return ' '.join(i[0] for i in t)


def criteria_ratios(tokens, criteria):
    '''Return a list of percentage of criteria based on the frequency of words in tokens'''
    nlp = spacy.load('en_core_web_md')
    criteria_ratio = [0] * len(criteria.split())
    tokens = nlp(tokens)
    criteria_tokens = nlp(criteria)
    for i, criterion in enumerate(criteria_tokens):
        for token in tokens:
            criteria_ratio[i] += token.similarity(criterion)
    # Get percentage
    sum_ = sum(criteria_ratio)
    for i, _ in enumerate(criteria_ratio):
        criteria_ratio[i] = round(criteria_ratio[i] / sum_ * 100, 2)
    return sorted(zip(criteria.split(), criteria_ratio), key=lambda x: x[1], reverse=True)


# Generate a word cloud image of document
def generate_wordcloud(text: str) -> None:
    wordcloud = WordCloud(width=780, height=450, max_font_size=50,
                          background_color="#ffde59").generate(' '.join(text))

    # Display the generated image:
    fig = plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    fig.savefig('Word cloud.png')


def generate_report(all_reviews, visualized: bool, criteria=None) -> None:
    if not criteria:
        criteria = 'default,criteria,separated,by,commas'
    # File name should be of pickle
    most_commons = get_most_commons(all_reviews, 2)
    # Remove "New York" from the bi_most_commons
    most_commons.pop(1)
    bi_ratios = to_df(criteria_ratios(extract_text(most_commons), criteria), 'criterion', 'percentage')
    if visualized:
        fig = plt.figure(figsize=(10, 6))
        sb.barplot(data=bi_ratios, x='criterion', y='percentage').set(title='Criteria ratios - Bigrams model')
        fig.savefig('Criteria percentage.png')
    else:
        bi_ratios.to_csv('Criteria percentage.csv', sep=',', encoding='utf-8', index=False)
