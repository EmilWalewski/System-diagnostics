
import pickle
import os
from klasyfikator_filmowy.vectorizer import vect
import numpy as np

clf = pickle.load(open(os.path.join('klasyfikator_filmowy', 'pkl_objects', 'classifier2.pkl'), 'rb'))

label = {0: 'Negative', 1: 'Positive'}

def print_result(text, processedText):
    print('Text: %s\nClass: %s\nProbability: %.2f%%\n' % (text, label[clf.predict(processedText)[0]], np.max(clf.predict_proba(processedText)) * 100))


example0 = ['I like this movie :)']
example1 = ['I hate this movie']
example2 = ['I do not like it']
example3 = ['This is my new favourite movie now']


X1 = vect.transform(example0)
X2 = vect.transform(example1)
X3 = vect.transform(example2)
X4 = vect.transform(example3)

print_result(example0, X1)
print_result(example1, X2)
print_result(example2, X3)
print_result(example3, X4)
