import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def get_training(file, task = "a"):
    """
    extracting the data from the training data, depending on which SemEval2019 
    subtask is choosen
    """
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

def get_test(file, label, task = "a"):
    """
    extracting the data from the training data, depending on which SemEval2019 
    subtask is choosen
    """    
    corpus = pd.read_csv(file, delimiter="\t", names=["id", "tweet"])
    label = pd.read_csv(label, delimiter=",", names=["id", "class"])
    X = corpus["tweet"][1:]
    y = label["class"]
    return X, y    

def vectorize(tweets):
    """
    takes the tweets as strings and changes them into numerical values
    """
    count_vect = CountVectorizer()
    vec = count_vect.fit_transform(tweets)  # transforms the tweets into vectors
    return vec, count_vect


def train(X_train, y_train):
    """
    trains the multinomial naive bayes classifier
    """
    clf = MultinomialNB().fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    """
    evaluates and returns the prediction of the test set.
    Returns classification report and confusion matrix
    """
    prediction = clf.predict(X_test)
    np.mean(prediction == y_test)
    cm = confusion_matrix(y_test, prediction)
    cf = classification_report(y_test, prediction)
    print(cm)
    print(cf)
    f1 = f1_score(y_test, prediction, average='macro')
    print("f1 macro: ", f1)
    return(prediction)
 

def delete_null(old_X_train, old_y_train, old_X_dev, old_y_dev, task = "b"):
    """
    only used for subtask b and c
    helper function to delete all tweets that were not labeled for the task
    all NOT offensive tweets are labeled as NULL in the other categories, which
    disturbs the classifier.
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
    
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=10):
    """
    prints the most important feature of each tweet ( a single word). 
    Taken over from here:
    https://stackoverflow.com/questions/26976362/how-to-get-most-informative-features-for-scikit-learn-classifier-for-different-c
    """
    print()
    print("MOST IMPORTANT FEATURES:")
    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print (class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print (class_labels[1], coef, feat)
    print()

def error_analysis(X_test, y_test, classification):
    """
    extract and print false predicted tweets to csv file
    """
    tweets = list()
    label = list()
    prediction = list()
    off_as_not = 0
    not_as_off = 0
    for i in range(len(classification)): 
        #print(classification[i])
        if y_test[i] != classification[i]:
            if y_test[i] == "OFF":
                off_as_not +=1
            elif y_test[i] == "NOT":
                not_as_off +=1
            tweets.append(X_test[i+1]) #this is a pandas object, counting starts with 1, not 0
            prediction.append(classification[i])
            label.append(y_test[i])
            
    df = pd.DataFrame(
            {"prediction": prediction,
             "label": label,
             "tweet": tweets
             })
    df.to_csv("created_files/NB_error_analysis", sep="\t") 