import pandas as pd
import Cleaning.Cleaning_data as cleaning
import Cleaning.Preprocessing as preprocessing
import Unsupervised_Learning.Unsupervised as ul
import Supervised_Learning.ML_Supervised as svml
import Supervised_Learning.DL_Supervised as svdl

'''
This will be used in the real case
file_name = 'New_York_reviews.csv'
df = pd.read_csv(file_name)
cleaning.clean_data(df)
'''

file_name = 'cleaned data.pkl'
cleaned_data = pd.read_pickle(file_name)
rest_list = cleaned_data['restaurant_id'].unique().tolist()


def ask_id():
    rest_id = input('Please enter your restaurant id.')
    while int(rest_id) not in rest_list:
        # 332
        rest_id = input('Your restaurant is not in the list. Please choose again')
    '''
    This will be used in the real case to filer reviews on the restaurant of the given id to the system
    preprocessing.pick_rest(cleaned_data, rest_id)
    '''


ask_id()

file_name = 'preprocessed restaurant 332.pkl'

# Supervised
svml.finalize(file_name)
svdl.finalize(file_name)

# Unsupervised
cleaned_data = pd.read_pickle(file_name)
reviews = pd.Series(cleaned_data['review_full']).tolist()

'''
This will be used in the real case
cleaned_data['review_full'] = cleaned_data['review_full'].apply(lambda row: preprocessing.clean_text(row))
'''

ul.generate_wordcloud(reviews)

# Popularly accepted criteria for restaurants from https://owlcation.com/humanities/How-to-Write-a-Restaurant-Review-With-Examples
criteria = 'food service interior exterior ambiance cleanliness value'
ul.generate_report(' '.join(reviews), criteria)
