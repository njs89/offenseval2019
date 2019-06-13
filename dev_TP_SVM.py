"""
dev edition for svm
"""
import pandas as pd
from sklearn import svm
from lxml import etree
import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import KFold

training_data = "Offenseval/offenseval-training-v1.tsv"

def load_data(file, cross_validation = 0, task = "a"):
    corpus = pd.read_csv(file, delimiter="\t", names=["id", "tweet", "subtask_a", "subtask_b", "subtask_c"])
    X = corpus["tweet"]
    #X = X[1:]
    label_a = corpus["subtask_a"]  # OFF or NOT offensive
    label_b = corpus["subtask_b"]  # TIN/UNT/NaN Targeted Insult, Untargeted Insult ???
    label_c = corpus["subtask_c"]  # IND/GRP/OTH/NaN Individual, Group, Other, ??
    length = len(corpus["id"])  # length of the corpus = 13241 First row = header, 13240 left

    if cross_validation == 1:    
        if task == "a":
            y = label_a
        elif task == "b":
            y = label_b
        elif task == "c":
            y = label_c
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        print(kf)
        KFold(n_splits=2, random_state=None, shuffle=False)
        counter = 0
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_dev = X[train_index], X[test_index]
            y_train, y_dev = y[train_index], y[test_index]
            
            X_vec, count_vect = vectorize(X_train)
            run_svr(X_vec, X_dev, y_train, y_dev, count_vect)
            counter+=1
            print("COUNTER", counter)
    elif cross_validation == 0:
        X_train = X[1:length - (length // 10)]  # 11916
        X_dev = X[length - (length // 10):]  # 1324
    
        if task == "a":
            y_train = label_a[1:length - (length // 10)]  # 11916
            y_dev = label_a[length - (length // 10):]  # 1324
        elif task == "b":
            y_train = label_b[1:length - (length // 10)]  # 11916
            y_dev = label_b[length - (length // 10):]  # 1324
        elif task == "c":
            y_train = label_c[1:length - (length // 10)]  # 11916
            y_dev = label_c[length - (length // 10):]  # 1324
    
        return X_train, y_train, X_dev, y_dev
    
def run_svr(X_train, X_dev, y_train, y_dev, count_vect):
    """takes the instances and goldlabels (SRL) and uses the svr magic for 
    creating an awesome prediction"""
    lin_clf = svm.LinearSVC()
    X_dev = count_vect.transform(X_dev)
    lin_clf.fit(X_train, y_train)
    prediction = lin_clf.predict(X_dev)
     
    cm = confusion_matrix(y_dev, prediction)
    cf = classification_report(y_dev, prediction)
    print(cm)
    print(cf)
 
    
def get_array_length(array):
    """ 
    Calculate the length of an array. 
    """
    counter = 0    
    for row in array:
        counter += 1        
    return counter

def vectorize(tweets):
    """vectorises the string of the tweets"""
    count_vect = CountVectorizer()
    vec = count_vect.fit_transform(tweets)  # transforms the tweets into vectors
    #AttributeError: 'list' object has no attribute 'lower'
    return vec, count_vect

def tweets_to_bow(tweets):
    """changes tweets into a string of back of words"""
    bow_list = list()
    #bow_dict = dict()
    for tweet in tweets:
        tweet = tweet.replace("@USER", "USER") #ValueError: could not convert string to float: '@USER'
        bow = tweet.split(" ")
        bow_string = ""
        for word in bow:
            bow_string += word+" "
        #bow_list.append({"tweet":bow_string})
        bow_list.append(bow_string)
    #print(bow_list)
    return bow_list        


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


cross_validation = 1    #10fold=1; single = 0 "kfold works only on subtask a"
b_and_c = 1             #only subtask a = 1; single = 0

#subtask a
print("SUBTASK A")
counter = 0
training_data = "Offenseval/offenseval-training-v1.tsv"
if cross_validation == 1:
    load_data(training_data, cross_validation, task ="a")# kfold
else:
    X_train, y_train, X_dev, y_dev = load_data(training_data, cross_validation = 0, task = "a")
    X_train_vec, count_vect = vectorize(X_train)
    X_dev = count_vect.transform(X_dev)
    
    clf = train(X_train_vec, y_train)
    evaluate(clf, X_dev, y_dev)

if b_and_c == 1:
    print("_______________________________________________________________")
    
    #subtask b
    print("SUBTASK B")
    X_train, y_train, X_dev, y_dev = load_data(training_data, cross_validation=0, task = "b")
    X_train, y_train, X_dev, y_dev = delete_null (X_train, y_train, X_dev, y_dev, task = "b")
    X_vec, count_vect = vectorize(X_train)
    run_svr(X_vec, X_dev, y_train, y_dev, count_vect)
    
    #subtask c
    print("SUBTASK C")
    X_train, y_train, X_dev, y_dev = load_data(training_data, cross_validation=0, task = "c")
    X_train, y_train, X_dev, y_dev = delete_null (X_train, y_train, X_dev, y_dev, task = "c")
    X_vec, count_vect = vectorize(X_train)
    run_svr(X_vec, X_dev, y_train, y_dev, count_vect)
