from tkinter import Tk, Label, Button, IntVar, BooleanVar, Radiobutton, messagebox, simpledialog
import pandas as pd
import Cleaning.Cleaning_data as cleaning
import Cleaning.Preprocessing as preprocessing
import Unsupervised_Learning.Unsupervised as ul
import Supervised_Learning.ML_Supervised as svml
import Supervised_Learning.DL_Supervised as svdl
import sys

#########################    Setting up part   #########################
'''
This will be used in the real scenario
file_name = 'New_York_reviews.csv'
cleaned_data_path = cleaning.clean_data(file_name)

'''
cleaned_data_path = 'cleaned data.pkl'
cleaned_data = pd.read_pickle(cleaned_data_path)
rest_list = cleaned_data['restaurant_id'].unique().tolist()


# https://pythonguides.com/python-tkinter-messagebox/

def save_btn():
    messagebox.showinfo(f'{answered_rest_id} restaurant',
                        f'Thank you for your input! We will email you the report when it\'s ready! Please close this app after done.')


app = Tk()
model_choice = IntVar()
vis_choice = BooleanVar()
vis_choice.set(1)
model_y = 70
report_y = 200

app.geometry('500x500')
app.title('Welcome')

# Verify if the right user
answered_rest_id = simpledialog.askinteger("Input", "What is your restaurant id?",
                                           parent=app, minvalue=min(rest_list), maxvalue=max(rest_list))
trial_count = 1
while trial_count < 3 and int(answered_rest_id) not in rest_list:
    trial_count += 1
    answered_rest_id = simpledialog.askinteger("Input", f"Wrong. You have {3 - trial_count} attemps!",
                                               parent=app, minvalue=min(rest_list), maxvalue=max(rest_list))

Label(app, text='Welcome to the restaurant app!', bg='yellow', relief='solid', font=18).pack()

Label(app, text='Please choose the type of model', font=10).place(x=30, y=model_y)
model1_btn = Radiobutton(app, text='Highest accuracy, long training time', variable=model_choice, value=1)
model1_btn.place(x=30, y=model_y + 30)
model2_btn = Radiobutton(app, text='High accuracy, medium training time', variable=model_choice, value=2)
model2_btn.place(x=30, y=model_y + 50)
model3_btn = Radiobutton(app, text='Accepted accuracy, fast training time', variable=model_choice, value=3)
model3_btn.place(x=30, y=model_y + 70)

Label(app, text='Which report do you want?', font=10).place(x=30, y=report_y)
vis1_btn = Radiobutton(app, text='Visualized report', variable=vis_choice, value=True)
vis1_btn.place(x=30, y=report_y + 30)
vis1_btn = Radiobutton(app, text='Plain report', variable=vis_choice, value=False)
vis1_btn.place(x=200, y=report_y + 30)

Button(app, text='Submit', command=save_btn).place(x=200, y=300)
app.mainloop()

#########################    Machine Learning part   #########################
'''
    This will be used in the real case to filer reviews on the restaurant of the given id to the system
    preprocessing.pick_rest(cleaned_data, rest_id)
'''
# For example, the id is 332
file_name = f'preprocessed restaurant {answered_rest_id}.pkl'

# Supervised
if model_choice.get() in [1, 2]:
    svml.finalize(file_name, model_choice.get())
elif model_choice.get() == 3:
    svdl.finalize(file_name)
else:
    print('There are only three model choices. Please run this app again.')
    sys.exit()

# Unsupervised
cleaned_data = pd.read_pickle(file_name)
reviews = pd.Series(cleaned_data['review_full']).tolist()

'''
This will be used in the real case
cleaned_data['review_full'] = cleaned_data['review_full'].apply(lambda row: preprocessing.clean_text(row))
'''

# Popularly accepted criteria for restaurants from https://owlcation.com/humanities/How-to-Write-a-Restaurant-Review-With-Examples
criteria = 'food service interior exterior ambiance cleanliness value'
if vis_choice.get():
    ul.generate_wordcloud(reviews)
    ul.generate_report(' '.join(reviews), criteria, visualized=True)
else:
    ul.generate_report(' '.join(reviews), criteria, visualized=False)
