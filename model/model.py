from sklearn.naive_bayes import GaussianNB
from model.preprocessing import preprocess_text, create_features
import numpy as np


def train(x_train, y_train):
    model = GaussianNB()
    print('Start training...')
    model = model.fit(x_train, y_train)
    print('Training completed!')
    return model


def predict(text, model, dictionary, le):
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = le.inverse_transform(prediction)[0]
    return prediction_cls
