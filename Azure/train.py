# import libraries
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mlflow


def main(args):
    mlflow.autolog()

    # read data
    df = get_data(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model and get vectorizer
    model, tfidf = train_model(args.reg_rate, X_train, y_train)

    # evaluate model using the fitted vectorizer
    eval_model(model, tfidf, X_test, y_test)


# function that reads the data
def get_data(path):
    print("Reading data...")
    df = pd.read_csv(path)
    return df


# function that splits the data
def split_data(df):
    print("Splitting data...")
    X = df['Cleaned_Review']  # Replace 'text' with your actual text column name
    y = df['Sentiment']  # Replace 'label' with your actual label column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# function that trains the model
def train_model(reg_rate, X_train, y_train):
    print("Training model...")

    # Fit the vectorizer on the training data
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vectorized = tfidf.fit_transform(X_train)  # Fit and transform training data

    # Train the model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(X_train_vectorized, y_train)

    return model, tfidf


# function that evaluates the model
def eval_model(model, tfidf, X_test, y_test):
    print("Evaluating model...")

    # Transform the test data using the same fitted vectorizer
    X_test_vectorized = tfidf.transform(X_test).toarray()  # Use transform here (without fit)

    # calculate accuracy
    y_hat = model.predict(X_test_vectorized)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test_vectorized)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print('AUC: ' + str(auc))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
