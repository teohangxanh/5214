import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def clean_data(file_name: str):
    '''Cleand and save data into pkl'''
    data = pd.read_csv(file_name)
    cleaned_data = data.copy()
    # Clean review_id and author_id
    cleaned_data['review_id'] = cleaned_data['review_id'].str.split('_').str[-1]
    cleaned_data['author_id'] = cleaned_data['author_id'].str.split('_').str[-1]

    # Convert date to datetime
    cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])

    # Convert rating_review to numeric
    cleaned_data['rating_review'] = pd.to_numeric(cleaned_data['rating_review'], errors='coerce', downcast='integer')

    # Convert string to categorical data to save memory
    cleaned_data["sample"] = cleaned_data["sample"].astype("category")

    # Create a column restaurant_id
    enc = LabelEncoder()
    enc.fit(cleaned_data['restaurant_name'])
    cleaned_data['restaurant_id'] = enc.transform(cleaned_data['restaurant_name'])

    # save encoder object to pickle file in case we want to recover the restaurant name
    # with open('restaurant_encoder.pkl', 'wb') as output:
    #     pickle.dump(enc, output, pickle.HIGHEST_PROTOCOL)

    # Drop redundant columns
    cleaned_data.drop(columns=['Unnamed: 0', 'parse_count', 'city', 'review_preview', 'restaurant_name'], inplace=True)

    # Save cleaned data
    folder = 'Cleaned data/'
    try:
        os.makedirs(folder)
    except OSError:
        # print(f'The folder Cleaned data already exists')
        pass
    cleaned_data_path = f'{folder}cleaned data.pkl'
    cleaned_data.to_pickle(cleaned_data_path)
    return cleaned_data_path
