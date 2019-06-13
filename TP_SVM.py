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
import time #just for fun and measure
start = time.time()

training_data = "Offenseval/offenseval-training-v1.tsv"
test_data = "Offenseval/offenseval-trial.txt"

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


#label a OFF/NOT    
X_train, y_train = get_training(training_data, task = "a")
X_test, y_test = get_test(test_data, task ="a")
X_vec, count_vect = vectorize(X_train)
run_svr(X_vec, X_test, y_train, y_test, count_vect)

#label b UNT/TIN/NaN  (untargeted, targeted, other)
X_train, y_train = get_training(training_data, task = "b")
X_test, y_test = get_test(test_data, task ="b")
X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "b")
X_vec, count_vect = vectorize(X_train)
run_svr(X_vec, X_test, y_train, y_test, count_vect)

#label c IND/GRP/OTH/NaN Individual, Group, Other
X_train, y_train = get_training(training_data, task = "c")
X_test, y_test = get_test(test_data, task ="c")
X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "c")
X_vec, count_vect = vectorize(X_train)
run_svr(X_vec, X_test, y_train, y_test, count_vect)
