"""
offensive language detection with svm
"""
from svm_utils import*

#change subtask_a etc. to "ON" or "OFF" if you want to run it
subtask_a = "ON" 
subtask_b = "OFF"
subtask_c = "OFF"

#data
training_data = "OLIDv1.0/olid-training-v1.0.tsv"
test_data_a = "OLIDv1.0/testset-levela.tsv"
test_data_b = "OLIDv1.0/testset-levelb.tsv"
test_data_c = "OLIDv1.0/testset-levelc.tsv"
label_a = "OLIDv1.0/labels-levela.csv"
label_b = "OLIDv1.0/labels-levelb.csv"
label_c = "OLIDv1.0/labels-levelc.csv"

if __name__ == "__main__":
    #subtask a
    if subtask_a == "ON":
        X_train, y_train = get_training(training_data, task = "a")
        X_test, y_test = get_test(test_data_a, label_a, task ="a")
        X_vec, count_vect = vectorize(X_train)
        run_svr(X_vec, X_test, y_train, y_test, count_vect)
    elif subtask_a == "OFF":
        pass
    
    #subtask b
    if subtask_b == "ON":
        X_train, y_train = get_training(training_data, task = "b")
        X_test, y_test = get_test(test_data_b, label_b, task ="b")
        X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "b")
        X_vec, count_vect = vectorize(X_train)
        run_svr(X_vec, X_test, y_train, y_test, count_vect)
    elif subtask_b == "OFF":
        pass
    
    #subtask c
    if subtask_c == "ON":    
        X_train, y_train = get_training(training_data, task = "c")
        X_test, y_test = get_test(test_data_c, label_c, task ="c")
        X_train, y_train, X_test, y_test = delete_null (X_train, y_train, X_test, y_test, task = "c")
        X_vec, count_vect = vectorize(X_train)
        run_svr(X_vec, X_test, y_train, y_test, count_vect)
    elif subtask_c == "OFF":
        pass
