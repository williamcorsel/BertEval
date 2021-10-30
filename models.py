from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.naive_bayes import GaussianNB #, MultinomialNB
from sklearn.svm import SVR, SVC

#TODO: We can vary the classification models here

bayes = GaussianNB()
#bayes = MultinomialNB()

svm = SVC()

# Sequential model to deal with bert features. 
# Src: https://medium.com/swlh/understand-tweets-better-with-bert-sentiment-analysis-3b054c4b802a
model = Sequential([
    Dense(512, activation='tanh', input_shape=(768,)),
    Dropout(0.5),
    Dense(128, activation='tanh'),
    Dropout(0.5),
    Dense(32, activation='tanh'),
    Dropout(0.5),
    Dense(3, activation='softmax'),
], name='seq')


model_large = Sequential([
    Dense(1024, activation='tanh', input_shape=(768,)),
    Dropout(0.5),
    Dense(256, activation='tanh'),
    Dropout(0.5),
    Dense(64, activation='tanh'),
    Dropout(0.5),
    Dense(3, activation='softmax'),
], name='seq_l')


model_extra_large = Sequential([
    Dense(1024, activation='tanh', input_shape=(768,)),
    Dense(1024, activation='tanh'),
    Dense(256, activation='tanh'),
    Dense(256, activation='tanh'),
    Dropout(0.5),
    Dense(64, activation='tanh'),
    Dropout(0.5),
    Dense(64, activation='tanh'),
    Dropout(0.5),
    Dense(3, activation='softmax'),
], name='seq_xl')

model.summary()
model_large.summary()

def get_models():
    # TODO: find some nice way to deal with both TF and sklearn models
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adagrad(),
                  metrics=['accuracy'])

    model_large.compile(loss='categorical_crossentropy',
                  optimizer=Adagrad(),
                  metrics=['accuracy'])
    
    model_extra_large.compile(loss='categorical_crossentropy',
                  optimizer=Adagrad(),
                  metrics=['accuracy'])

    models = (model_extra_large, model_large, model, bayes, svm)
    return models

