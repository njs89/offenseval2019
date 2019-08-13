"""
short script to extract all files in which both classifiers gave a wrong prediction
"""
import pandas as pd
NB = pd.DataFrame.from_csv("created_files/NB_error_analysis", sep="\t")
SVM = pd.DataFrame.from_csv("created_files/SVM_error_analysis", sep="\t")
CNN = pd.DataFrame.from_csv("created_files/CNN_Errors", sep=",")

cnn_tweets = CNN["tweet"].tolist()
cnn_pred = CNN["prediction"].tolist()
cnn_label = CNN["id"].tolist()

nb_tweets = NB["tweet"].tolist()
nb_pred = NB["prediction"].tolist()
nb_label = NB["label"].tolist()

svm_tweets = SVM["tweet"].tolist()
svm_pred = SVM["prediction"].tolist()
svm_label = SVM["label"].tolist()
same_tweet = list()
same_label = list()
same_pred = list()

counter = 0
for i in range(len(svm_tweets)):
    #for iteration the svm tweets are choosen, because there are more
    if svm_tweets[i] in nb_tweets and svm_tweets[i] in cnn_tweets:
        index = nb_tweets.index(svm_tweets[i]) #index of the tweet in nb
        same_tweet.append(svm_tweets[i])
        same_label.append(svm_label[i])
        nb_pre = nb_pred[index]
        svm_pre = svm_pred[i]
        same_pred.append((nb_pre, svm_pre))
        counter +=1

        if nb_pre != svm_pre:
            print()
            print(nb_pre)
            print(svm_pre)
            print(svm_tweets[i])
            print(svm_label[i])
            
print(counter)

df=pd.DataFrame(
        {"tweet":same_tweet,
         "label":same_label,
         "pred(NB/SVM)":same_pred})
df.to_csv("created_files/same_errors_NB_SVM", sep="\t")
      
