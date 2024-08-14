from model.model import *
import nltk
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import dataset
DATASET_PATH = './dataset/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)
messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()

# Preprocessing
# a) Label
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')

# b) Feature
messages = [preprocess_text(message) for message in messages]
dictionary = create_dictionary(messages)
X = np.array([create_features(tokens, dictionary) for tokens in messages])

# c) Split the dataset
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=TEST_SIZE,
                                                    shuffle=True,
                                                    random_state=SEED)

# Train model
model = train(X_train, y_train)

# Evaluate model
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Val accuracy: {val_accuracy}')
print(f'Test accuracy: {test_accuracy}')

# Predict
test_input = 'I am actually thinking a way of doing something useful'
prediction_cls = predict(test_input, model, dictionary, le)
print(f'Prediction: {prediction_cls}')
