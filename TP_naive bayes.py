"""
dev edition for naive bayes, change for final version into test, without dev set
"""
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_training(file, task = "a"):
    corpus = pd.read_csv(file, delimiter="\t", names=["id", "tweet", "subtask_a", "subtask_b", "subtask_c"])
    tweets = corpus["tweet"]
    label_a = corpus["subtask_a"]  # OFF or NOT offensive
    label_b = corpus["subtask_b"]  # TIN/UNT/NaN Targeted Insult, Untargeted Insult ???
    label_c = corpus["subtask_c"]  # IND/GRP/OTH/NaN Individual, Group, Other, ??
    X = tweets

    if task == "a":
        y = label_a  
    elif task == "b":
        y = label_b  
    elif task == "c":
        y = label_c  

    return X, y

def get_test(file, task = "a"):
    corpus = pd.read_csv(file, delimiter="\t", names=["tweet", "subtask_a", "subtask_b", "subtask_c"])
    #ids = corpus["id"]
    tweets = corpus["tweet"]
    label_a = corpus["subtask_a"]  # OFF or NOT offensive
    label_b = corpus["subtask_b"]  # TIN/UNT/NaN Targeted Insult, Untargeted Insult ???
    label_c = corpus["subtask_c"]  # IND/GRP/OTH/NaN Individual, Group, Other, ??
    X = tweets
    if task == "a":
        y = label_a  
    elif task == "b":
        y = label_b  
    elif task == "c":
        y = label_c  

    return X, y
    
    

def vectorize(tweets):
    #print ((tweets))
    count_vect = CountVectorizer()
    vec = count_vect.fit_transform(tweets)  # transforms the tweets into vectors
    return vec, count_vect


def train(X_train, y_train):
    clf = MultinomialNB().fit(X_train, y_train)
    return clf

def evaluate(clf, X_dev, y_dev):
    #print(X_dev.shape)
    prediction = clf.predict(X_dev)
    np.mean(prediction == y_dev)
    cm = confusion_matrix(y_dev, prediction)
    cf = classification_report(y_dev, prediction)
    print(cm)
    print(cf)

def delete_null(old_X_train, old_y_train, old_X_dev, old_y_dev, task = "b"):
    """
    helper function to delete all tweets that were not labeled for the task
    all NOT offensive tweets are labeled as NULL in the other categories.
    That disturbs the classifier.
    There is probably a more elegant pandas solution, for now this will work.
    """
    helper_X_train = list()
    helper_y_train = list()
    helper_X_dev = list()
    helper_y_dev = list()
    for row in old_X_train:
        helper_X_train.append(row)
    for row in old_y_train:
        helper_y_train.append(row)
    for row in old_X_dev:
        helper_X_dev.append(row)
    for row in old_y_dev:
        helper_y_dev.append(row)
    
    X_train = list()
    y_train = list()
    X_dev = list()
    y_dev = list()
    if task == "b":        
        for i in range(len(helper_X_train)):
            if helper_y_train[i] != "UNT" and helper_y_train[i] != "TIN":
                continue
            else:
                X_train.append(helper_X_train[i])
                y_train.append(helper_y_train[i])
                
        for i in range(len(helper_X_dev)):
            if helper_y_dev[i] != "UNT" and helper_y_dev[i] != "TIN":
                continue
            else:
                X_dev.append(helper_X_dev[i])
                y_dev.append(helper_y_dev[i])

    elif task == "c":
        for i in range(len(helper_X_train)):
            if helper_y_train[i] != "IND" and helper_y_train[i] != "GRP" and helper_y_train[i] != "OTH":
                continue
            else:
                X_train.append(helper_X_train[i])
                y_train.append(helper_y_train[i])
                
        for i in range(len(helper_X_dev)):
            if helper_y_dev[i] != "IND" and helper_y_dev[i] != "GRP" and helper_y_dev[i] != "OTH":
                continue
            else:
                X_dev.append(helper_X_dev[i])
                y_dev.append(helper_y_dev[i])
        
    return X_train, y_train, X_dev, y_dev
    
data = "main"
if data == "main": # i dont understand that
    #subtask a
    training_data = "Offenseval/offenseval-training-v1.tsv"
    test_data = "Offenseval/offenseval-trial.txt"
    X_train, y_train = get_training(training_data, task = "a")
    X_test, y_test = get_test(test_data, task ="a")
    X_train_vec, count_vect = vectorize(X_train)
    X_test = count_vect.transform(X_test)
    
    clf = train(X_train_vec, y_train)
    print(y_test)
    evaluate(clf, X_test, y_test)
    print("NB")    
    
    #subtask b
    X_train, y_train = get_training(training_data, task = "b")
    X_test, y_test = get_test(test_data, task ="b")
    X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "b")
    X_train_vec, count_vect = vectorize(X_train)
    X_test = count_vect.transform(X_test)
    
    clf = train(X_train_vec, y_train)
    evaluate(clf, X_test, y_test)
    print("NB")    

    #subtask c
    X_train, y_train = get_training(training_data, task = "c")
    X_test, y_test = get_test(test_data, task ="c")
    X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "c")
    X_train_vec, count_vect = vectorize(X_train)
    X_test = count_vect.transform(X_test)
    
    clf = train(X_train_vec, y_train)
    evaluate(clf, X_test, y_test)
    print("NB")    
