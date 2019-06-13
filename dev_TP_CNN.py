import pandas as pd
import string
import nltk
import pickle
from numpy import array
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import load_model


stop_words = nltk.corpus.stopwords.words('english')

def extract_comments(dataframe, type_of_comment, name_of_target_column):
    '''
        Extracts particular kind of comments. Returns pandas Series.
        dataframe = a pandas dataframe
        type_of_comment = keyword for specific kind of comments, it must be of str type
        name_of_target_column = neme of column that holds the target 
    '''
    
    column = dataframe[dataframe[name_of_target_column] == type_of_comment] ["tweet"]
    return column
    
def clean_tweet(a_list):
    '''
        Cleans punctuation from review, tokenizes review, turns review to lowercase. 
        Returns a list of lists(each list is a clean review).
        a_list = name of list or pandas Seriers
    '''
    
    data_list = list()
    for i in a_list:
        tokenized_row = nltk.word_tokenize(i)
        no_punct = [word for word in tokenized_row if not word in string.punctuation]
        alpha_tok = [tok for tok in no_punct if tok.isalpha]
        chars = [string for string in alpha_tok if len(string) > 1]
        clean_list = [stop for stop in chars if not stop in stop_words]
        lower_case = [wrd.lower() for wrd in clean_list]
        review = " ".join(lower_case)
        data_list.append(review)
    return data_list

def merged_list(first_list, second_list):
    '''
        Creates a merged list of offensive and not offensive comments. Returns clean merged list.
        first_list = list with data
        second_list = list with data
    '''

    list_a = clean_tweet(first_list)
    list_b = clean_tweet(second_list)
    merged_list = list_a + list_b
    return merged_list
          
def write_to_file(dataset, file_name):
    '''
        Saves a dataset to a pickle file.
        filename = name of file, type pickle binary
        dataset = name of dataset, type list
    '''
    
    with open(file_name, "wb") as infile:
        pickle.dump(dataset, infile)
        print('Saved: %s' % file_name)

def load_from_file(path_to_file):
    '''
        Loads dataset from pickle file.
        filename = name of file, type pickle binary
        dataset = name of dataset, type list
    '''
    
    with open(path_to_file, "rb") as infile:
        data = pickle.load(infile)
        return data
        
def create_tokenizer(lines):
    '''
        Creates a keras tokenizer. Returns keras tokenizer.
        lines = dataset, type list
    '''
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    '''
        Calculates the maximum document length.
        lines = dataset, type list
    '''
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    '''
        Transforms a list of num_samples sequences into a 2D Numpy.
    '''
    
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded
     
def define_model(length, vocab_size):
    '''
        Creates model. Returns model.
        length = length of the data set
        vocab_size = size of the vocabulary of the data set
    '''
	# channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 200)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 200)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 200)(inputs3)
    conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model



# Create dataframe from tsv
df = pd.DataFrame.from_csv("Offenseval/offenseval-training-v1.tsv", sep="\t")
# Extract target column(off_comments) from dataframe
list_with_off_tweets = extract_comments(df, "OFF", "subtask_a") 
# Extract target column(not_comments) from dataframe
list_with_not_tweets= extract_comments(df, "NOT", "subtask_a")
# Merge the lists with off and not tweets
training_data = merged_list(list_with_off_tweets, list_with_not_tweets)
# Create labels for off and not tweets. 0 for offens 1 for not offens
training_labels = [0 for _ in range(4400)] + [1 for _ in range(8840)]
# Write merged list and labels to a pickle file
write_to_file([training_data,training_labels], 'train.pkl')
# Make dataframe out of txt
df2 = pd.read_table("Offenseval/offenseval-trial.txt", delim_whitespace=False, names=("tweet", "subtask_a", "subtask_b", "subtask_c"))
#Extract target column(off_comments) from test dataframe
test_list_with_off_tweets= extract_comments(df2, "OFF", "subtask_a")
#Extract target column(off_comments) from test dataframe
test_list_with_not_tweets= extract_comments(df2, "NOT", "subtask_a")
#Merge the test lists with off and not tweets
test_data = merged_list(test_list_with_off_tweets, test_list_with_not_tweets)
#Create labels for off and not tweets. 0 for offens 1 for not offens
test_labels = [0 for _ in range(77)] + [1 for _ in range(243)]
# Write merged test list and test labels to a pickle file
write_to_file([test_data,test_labels], 'test.pkl')
# Load dataset from pickle file
trainLines, trainLabels = load_from_file('train.pkl')
# Create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)
# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=1, batch_size=16)
# save the model
model.save('model.h5')
# load datasets
testLines, testLabels = load_from_file('test.pkl')
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode test data
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)
# load the model
model = load_model('model.h5')
# evaluate model on training dataset
loss, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %f' % (acc*100))
# evaluate model on test dataset dataset
loss, acc = model.evaluate([testX,testX,testX],array(testLabels), verbose=0)
print(loss)
print('Test Accuracy: %f' % (acc*100))
# Get the predictions of the model 
predictions = model.predict([testX, testX, testX])
# Round up the predictions
predictions_rounded = [value.round() for value in predictions]
# Create dataframe with tweets, id's and predictions
df3 = pd.DataFrame(
    {'tweet': test_data,
     'id': test_labels,
     'prediction': predictions_rounded})
# Extract errors from df3  
df4 = df3[df3["id"] != df3["prediction"]]
df4.to_csv("Errors", sep=",")
