import re
import numpy as np
import contractions
import spacy
import pickle
import os


def clean_text(text):
    nlp = spacy.load('en_core_web_md')
    stopwords = nlp.Defaults.stop_words
    # Contraction
    expanded_words = []
    for word in text.split():
        # using contractions to expand the shortened words
        expanded_words.append(contractions.fix(word))
    text = ' '.join(expanded_words)
    text = re.sub(r'[^\w\s]', '', text)  # Symbols removal
    text = re.sub(r'\\s{2,}', r'\.', text)  # knowwwwwww -> know
    text = text.strip()
    # lemmatization
    text = ' '.join(token.lemma_.lower() for token in nlp(text) if token.lemma_.lower() not in stopwords)
    return text


def pick_rest(df, given_id):
    '''Filter the entire dataset to return only rows of the restaurant id'''
    rest = df[df['restaurant_id'] == given_id].reset_index(drop=True)
    # Drop unnecessary columns
    rest = rest[['review_full', 'rating_review']]
    # Vectorization
    rest['review_full'] = np.vectorize(clean_text)(rest['review_full'])
    # Save cleaned data
    folder = 'Cleaned data/'
    try:
        os.makedirs(folder)
    except OSError:
        # print(f'The folder Cleaned data already exists')
        pass
    with open(f'{folder}preprocessed restaurant {given_id}.pkl', 'wb') as f:
        pickle.dump(rest, f)
    return rest
