This package contains files for three machine learning classifiers to train and test on the SemEval 2019 Task 6 for offensive language detection.
The classifiers are a multinomial naive bayes (NB), a support vector machine (SVM) and a convolutional neural network (CNN).
The OLID Data set is already included. The classifiers can be run after the python packages have been installed.

-------------------------------

files:

created_files 		#file path for output of cnn and error analysis files
Error Analysis 		#Error Analysis files from the paper
OLIDv1.0 		#SemEval 2019 Task 6 corpus for training and test (the data is downloaded from here: https://competitions.codalab.org/competitions/20011#participate)
-------------------------------

python files:

nb_utilys.py		contains all functions for the NB
svm_utils.py		contains all functions for the SVM (some functions are the same as the NB)
nb_semeval.py		run this file for the NB
svm_semeval.py		run this file for the SVM
cnn_semeval.py		run this file for the CNN (functions in the file)
error_comparison.py	run this file AFTER the three classifiers, to create a file with the overlapping falsely classified tweets

The files nb_utils.py and svm_utils.py will be imported into the other files.
nb_semeval.py, svm_semeval.py and cnn_semeval.py can be activated in any order, as they do not interlink.
nb_semeval.py and svm_semeval.py can also train and test subtask b and c of the SemEval task 6. 
To do so, change the variable subtask_b and/or subtask_c from "OFF" to "ON". They are deactivated by default.	
The 3 classifiers give as an output a confusion matrix, the f1 score and other evaluation data. 


They also create files in the "created files" file, including error analysis files with falsely classified tweets.



error_comparison.py needs files created from all 3 classifiers, saved in the "created_files" file The results will also be stored in that file.
-------------------------------

Necessary packages for python:

The files were written with Anaconda Spyder 3.3.4 and python 3.6
- pandas: a tutorial for the installation is found here: https://pandas.pydata.org/pandas-docs/stable/install.html
- keras: Keras is a high-level neural networks API. To use keras either TensorFlow, Theano or CNTK has to be installed: https://keras.io/
- scikit learn: https://scikit-learn.org/stable/install.html
- numpy: is part of most python systems or can be installed following this guides: https://scipy.org/install.html
- nltk: https://www.nltk.org/install.html

