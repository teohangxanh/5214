import pandas as pd
import pickle
import spacy_sentence_bert
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import os


def finalize(file_name: str, model_class_choice: int) -> None:
    '''This does everything from A to Z given the file name'''
    df = pd.read_pickle(file_name)
    # load the BERT sentence transformer and create the vector embedding
    nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')
    df['vector'] = df['review_full'].apply(lambda x: nlp(x).vector)
    y = df['rating_review']
    X = df['vector']
    X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y.tolist(), test_size=0.20, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    if model_class_choice == 1:
        model = GaussianNB()
    elif model_class_choice == 2:
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    # Save model
    model_folder = 'Saved model/'
    try:
        os.makedirs(model_folder)
    except OSError:
        # print(f'The folder {model_folder} already exists')
        pass

    # Save accuracy
    acc_folder = 'Accuracy/'
    try:
        os.makedirs(acc_folder)
    except OSError:
        # print(f'The folder Accuracy already exists')
        pass

    # Save model
    pickle.dump(model, open(f'{model_folder}{model.__class__.__name__}', 'wb'))
    with open(f"{acc_folder}{model.__class__.__name__} accuracy score.txt", "w") as text_file:
        text_file.write(str(accuracy_score(y_preds, y_test)))

# file_name = r'../preprocessed restaurant 332.pkl'
