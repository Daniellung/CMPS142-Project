import pandas as pd
import numpy as np
import nltk, re, string, collections
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.utils import to_categorical

train = pd.read_csv("/notebooks/storage/final_project/train.csv")
test = pd.read_csv("/notebooks/storage/final_project/real_test.csv")

max_len = 50
full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
tk = Tokenizer(lower = True, filters="")
tk.fit_on_texts(full_text)
train_tokenized = tk.texts_to_sequences(train["Phrase"])
test_tokenized = tk.texts_to_sequences(test["Phrase"])
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)
y_ohe = to_categorical(train["Sentiment"])


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype="float32")

embedding_path = "/notebooks/storage/final_project/crawl-300d-2M.vec"
embed_size = 300
max_features = 30000
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, "r+", encoding="utf-8"))
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

lr = 1e-3
lr_d = 1e-10
units = 64
spatial_dr = 0.3
kernel_size1 = 3
kernel_size2 = 2
dense_units = 32
dr = 0.1
conv_size = 32

inp = Input(shape = (max_len,))
x = Embedding(16530, embed_size, weights = [embedding_matrix], trainable = False)(inp)
x1 = SpatialDropout1D(spatial_dr)(x)
x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding="valid", kernel_initializer="he_uniform")(x_lstm)
avg_pool1_lstm = GlobalAveragePooling1D()(x1)
max_pool1_lstm = GlobalMaxPooling1D()(x1)
x2 = Conv1D(conv_size, kernel_size=kernel_size2, padding="valid", kernel_initializer="he_uniform")(x_lstm)
avg_pool2_lstm = GlobalAveragePooling1D()(x2)
max_pool2_lstm = GlobalMaxPooling1D()(x2)
x = concatenate([avg_pool1_lstm, max_pool1_lstm, avg_pool2_lstm, max_pool2_lstm, avg_pool1_lstm, max_pool1_lstm])
x = Dense(5, activation = "softmax")(x)

model = Model(inputs = inp, outputs = x)
model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
history = model.fit(X_train, y_ohe, batch_size = 16, epochs = 16, validation_split=0.1, verbose = 1)
preds = model.predict(X_test, batch_size = 1024, verbose = 1)
predictions = np.round(np.argmax(preds, axis=1)).astype(int)
test = test.drop(["SentenceId", "Phrase"], axis=1)
test["Sentiment"] = predictions
test.to_csv("/notebooks/storage/final_project/out/LungPetruscaSarkar_predictions_FINAL.csv", index=False)
