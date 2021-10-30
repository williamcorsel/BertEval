import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model

from dataloader import Dataloader
from models import get_models

'''
Example bert models
tfidf                                       
bert-base-uncased                           basic BERT model
cardiffnlp/twitter-roberta-base             BERT model trained on twitter data
cardiffnlp/twitter-roberta-base-sentiment   BERT model trained on twitter data and further on sentiment
'''

parser = argparse.ArgumentParser(description='Test BERT models on SemEval dataset')
parser.add_argument('op', type=str, choices=['train', 'test'],
                    help="What operation to perform")
parser.add_argument('--preprocess_model', type=str, default='bert-base-uncased',
                    help="Which preprocessing model to use, e.g.: tfidf, bert-base-uncased, cardiffnlp/twitter-roberta-base, cordiffnlp/twitter-roberta-base-sentiment")
parser.add_argument('--data_path', type=str, default='data/tweets/',
                    help="Path were data is located")
parser.add_argument('--processed_path', type=str, default='data/processed/',
                    help="Path where pre-processed data will be stored")
parser.add_argument('--output', type=str, default='output/',
                    help="Path to store output")
parser.add_argument('--preprocess', action='store_true', default=False,
                    help="Forces preprocessing of data")
parser.add_argument('--sentence_embedding_averaged', action='store_true', default=False,
                    help="Whether to average the outputs of all token embeddings instead of just using the embedding of the [CLS] token")
parser.add_argument('--results_path', type=str, default='results/',
                    help="Path where results will be stored")
parser.add_argument("--year_dataset", type=int, default=2013, 
                    help="Year of dataset to be used, 2013 is default")        
args = parser.parse_args()

if not os.path.exists(args.data_path):
    print("Data path not found!")
    exit(1)

if not os.path.exists(args.processed_path):
    os.mkdir(args.processed_path)

if not os.path.exists(args.output):
    os.mkdir(args.output)

if not os.path.exists(args.results_path):
    os.mkdir(args.results_path)

# Load data
loader = Dataloader(args.data_path, args.processed_path, args.preprocess_model, args.preprocess, sentence_embedding_averaged=args.sentence_embedding_averaged)
X_train, y_train = loader.get_single_data(year=args.year_dataset,mode='train')
X_test, y_test = loader.get_single_data(year=args.year_dataset, mode='test')

# Compute class weights to balance out the importance of each label
# Convert to dict to fit the keras format. TODO: fix the hardcode
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, 1)), y=np.argmax(y_train, 1))
class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}

model_extra_large, model_large, model, bayes, svm = get_models()

bayes_name = '{0}/bayes_{1}.pkl'.format(args.output, args.preprocess_model.rsplit("/", 1)[-1])
svm_name = '{0}/svm_{1}.pkl'.format(args.output, args.preprocess_model.rsplit("/", 1)[-1])
model_name = '{0}/seq_model_{1}'.format(args.output, args.preprocess_model.rsplit("/", 1)[-1])
model_large_name = '{0}/seq_model_large_{1}'.format(args.output, args.preprocess_model.rsplit("/", 1)[-1])
model_extra_large_name = '{0}/seq_model_extra_large_{1}'.format(args.output, args.preprocess_model.rsplit("/", 1)[-1])

if args.op == "test":
    ## BAYES
    with open(bayes_name, 'rb') as file:
        bayes = pickle.load(file)

    ## SVM
    with open(svm_name, 'rb') as file:
        svm = pickle.load(file)

    ## MODEL
    model = load_model(model_name)
    
    ## MODEL_LARGE
    model_large = load_model(model_large_name)

    ## MODEL EXTRA LARGE
    model_extra_large = load_model(model_extra_large_name)

else:
    print("Training models...")

    ## BAYES
    print("Bayes")
    # Bayes only works with single numbers so convert labels to 0,1,2 notation
    y_train_single = np.argmax(y_train, 1)
    bayes.fit(X_train, y_train_single)

    with open(bayes_name, 'wb') as file:
        pickle.dump(bayes, file)

    ## SVM
    print("SVM")
    svm.fit(X_train, y_train_single)

    with open(svm_name, 'wb') as file:
        pickle.dump(svm, file)

    ## MODEL
    print("MODEL")
    hist = model.fit(
        X_train, 
        y_train,
        batch_size=64,
        epochs=100,
        class_weight=class_weights,
    )
    model.save(model_name)

    print("MODEL_LARGE")
    hist = model_large.fit(
        X_train, 
        y_train,
        batch_size=64,
        epochs=100,
        class_weight=class_weights,
    )
    model_large.save(model_large_name)
    
    print("MODEL_EXTRA_LARGE")
    hist = model_extra_large.fit(
        X_train, 
        y_train,
        batch_size=64,
        epochs=100,
        class_weight=class_weights,
    )
    model_extra_large.save(model_extra_large_name)

addition = "averaged-embedding" if args.sentence_embedding_averaged else "cls-embedding"
csv_sep = "f1-pos-neg " + "_{0}_{1}_{2}".format(args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition) + ": Bayes, SVM, Dense, Dense-L, Dense-XL \n"

print("Testing models...")

## BAYES
print('Bayes')
y_true, y_pred = np.argmax(y_test, 1), bayes.predict(X_test)
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report)
df['f1-pos-neg'] = (df.loc['f1-score']['0'] + df.loc['f1-score']['2']) / 2

print("Saving as: "+ args.preprocess_model.rsplit("/", 1)[-1])
csv_sep += str(df.loc['f1-score']['f1-pos-neg']) + ","
df.to_csv("{0}/results_{1}_{2}_{3}_bayes.csv".format(args.results_path, args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition))


## SVM
print('SVM')
y_true, y_pred = np.argmax(y_test, 1), svm.predict(X_test),
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report)
df['f1-pos-neg'] = (df.loc['f1-score']['0'] + df.loc['f1-score']['2']) / 2
csv_sep += str(df.loc['f1-score']['f1-pos-neg']) + ","
df.to_csv("{0}/results_{1}_{2}_{3}_svm.csv".format(args.results_path, args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition))


## MODEL
print('MODEL')
# Extract highest values from the data and results and print classification report
y_true, y_pred = np.argmax(y_test, 1), np.argmax(model.predict(X_test), 1)
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report)
df['f1-pos-neg'] = (df.loc['f1-score']['0'] + df.loc['f1-score']['2']) / 2
csv_sep += str(df.loc['f1-score']['f1-pos-neg']) + ","
df.to_csv("{0}results_{1}_{2}_{3}_dense.csv".format(args.results_path, args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition))


## MODEL L
print('MODEL_L')
# Extract highest values from the data and results and print classification report
y_true, y_pred = np.argmax(y_test, 1), np.argmax(model_large.predict(X_test), 1)
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report)
df['f1-pos-neg'] = (df.loc['f1-score']['0'] + df.loc['f1-score']['2']) / 2
csv_sep += str(df.loc['f1-score']['f1-pos-neg']) + ","
df.to_csv("{0}results_{1}_{2}_{3}_dense-L.csv".format(args.results_path, args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition))


## MODEL XL
print('MODEL_XL')
# Extract highest values from the data and results and print classification report
y_true, y_pred = np.argmax(y_test, 1), np.argmax(model_extra_large.predict(X_test), 1)
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report)
df['f1-pos-neg'] = (df.loc['f1-score']['0'] + df.loc['f1-score']['2']) / 2
csv_sep += str(df.loc['f1-score']['f1-pos-neg']) + ","
df.to_csv("{0}results_{1}_{2}_{3}_dense-XL.csv".format(args.results_path, args.year_dataset, args.preprocess_model.rsplit("/", 1)[-1], addition))

print("\nResults:")
print(csv_sep)
