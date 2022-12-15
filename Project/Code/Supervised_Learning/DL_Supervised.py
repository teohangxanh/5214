import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# tokenization
tok = spacy.load('en_core_web_md')


def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]


def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# Pytorch Dataset
class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


def train_model(model, train_dl, valid_ds, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    val_accs = []
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, valid_ds)
        val_accs.append(val_acc)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
                sum_loss / total, val_loss, val_acc, val_rmse))
    return model, sum(val_accs) / len(val_accs)


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.clone().detach().to(torch.long)
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1))) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total


# LSTM with fixed length input
class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])


def fit_and_save(file_name):
    reviews = pd.read_pickle(file_name)
    reviews = reviews.rename({'review_full': 'review', 'rating_review': 'rating'}, axis=1)

    # changing ratings to 0-numbering
    zero_numbering = {1.0: 0, 2.0: 1, 3.0: 2, 4.0: 3, 5.0: 4}
    reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

    # count number of occurences of each word
    counts = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['review']))

    # deleting infrequent words
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))

    X = list(reviews['encoded'])
    y = list(reviews['rating'])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = ReviewsDataset(X_train, y_train)
    valid_ds = ReviewsDataset(X_valid, y_valid)

    batch_size = 5000
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)

    model_fixed = LSTM_fixed_len(vocab_size, 50, 50)
    trained_model, accuracy = train_model(model_fixed, train_dl, val_dl, epochs=30, lr=0.01)
    # Save model
    folder = 'Saved model/'
    try:
        os.makedirs(folder)
    except OSError:
        # print('The folder Saved model already exists')
        pass
    torch.save(trained_model.state_dict(), f'{folder}LSTM_fixed_length.pth')
    return accuracy


def finalize(file_name):
    accuracy = fit_and_save(file_name)
    folder = 'Accuracy/'
    try:
        os.makedirs(folder)
    except OSError:
        # print('The folder Accuracy already exists')
        pass
    with open(f'{folder}LSTM fixed length.txt', "w") as text_file:
        text_file.write(str(accuracy))

# file_name = r'../preprocessed restaurant 332.pkl'
