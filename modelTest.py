import gensim
#from nltk import RegexpTokenizer
import nltk
import gensim
import numpy as np
from textPreProc import normalize, basicReplacement, remSW
#from scipy import spatial
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


###########
file1 = 'yelp_labelled.txt'
file2 = 'amazon_cells_labelled.txt'
file3 = 'imdb_labelled.txt'
file4 = 'uci_testdata.txt'

sent1=np.genfromtxt(file1, dtype='string', delimiter='\t', usecols=[0])
labels1=np.genfromtxt(file1, delimiter='\t', usecols=[1])
sent2=np.genfromtxt(file2, dtype='string', delimiter='\t', usecols=[0])
labels2=np.genfromtxt(file2, delimiter='\t', usecols=[1])
sent3=np.genfromtxt(file3, dtype='string', delimiter='\t', usecols=[0])
labels3=np.genfromtxt(file3, delimiter='\t', usecols=[1])
sent4=np.genfromtxt(file4, dtype='string', delimiter='\t', usecols=[1])
labels4=np.genfromtxt(file4, delimiter='\t', usecols=[0])

sent = np.concatenate((sent1,sent2,sent3,sent4))
labels = np.concatenate((labels1,labels2,labels3,labels4))
print "Size of labelled data: " ,np.size(sent)
###########
model = gensim.models.Doc2Vec.load('d2v.model')


file5 = 'test_data.txt'

test_sent=np.genfromtxt(file5, dtype='string', delimiter='\t', usecols=[0])
test_labels=np.genfromtxt(file5, delimiter='\t', usecols=[1])


size_features = 30

test_arrays=np.zeros((len(test_sent),size_features))


train_arrays=np.zeros((len(sent),size_features))
train_labels=np.zeros(len(sent))
	

corpus1 = sent
for i in range(len(sent)):
	train_arrays[i][0:30] = model.infer_vector(nltk.word_tokenize(basicReplacement(corpus1[i].decode('utf-8'))))

	train_labels[i] = labels[i]
	

corpus2 = test_sent
for i in range(len(test_sent)):
	test_arrays[i][0:30] = model.infer_vector(nltk.word_tokenize(basicReplacement(corpus2[i].decode('utf-8'))))
	
	
#Classifier Models
#Logistic Regression
lr_clf = LogisticRegression()
lr_clf.fit(train_arrays,train_labels)
LogisticRegression()



#SVM
lin_clf = LinearSVC()
lin_clf.fit(train_arrays, train_labels) 
LinearSVC()

#RFC not so good, ignore
rfc_clf = RandomForestClassifier()
rfc_clf.fit(train_arrays, train_labels)
RandomForestClassifier(criterion='entropy')

#Naive Bayes Gaussian
nb_clf = GaussianNB()
nb_clf.fit(train_arrays,train_labels)
GaussianNB(priors=None)

#Naive Bayes MN

#Dec Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_arrays,train_labels)
DecisionTreeClassifier(max_depth=3)

#train_arrays=preprocessing.MinMaxScaler(train_arrays)





print "Log Reg: ",lr_clf.score(test_arrays, test_labels)
print "Linear SVM: " ,lin_clf.score(test_arrays, test_labels)
print "RF Classifier: " ,rfc_clf.score(test_arrays, test_labels)
print "Naive Bayes: ",nb_clf.score(test_arrays,test_labels)
print "Decision Tree: ",dt_clf.score(test_arrays,test_labels)

import math
while 1:
	text = raw_input("Enter a sentence: ")
	text = basicReplacement(text)

	print "1: Positive, 0: Negative"
	vec = model.infer_vector(nltk.word_tokenize(text.decode('utf-8')))
	vec = vec.reshape(-1, size_features)
	print "Sentiment: ",(1 if (lr_clf.predict(vec)+dt_clf.predict(vec)+lin_clf.predict(vec)+rfc_clf.predict(vec)+nb_clf.predict(vec))>2.5 else 0)
	
	'''
	print "%d Log Reg: "%lr_clf.predict((vec))
	print "%d Linear SVM: " %lin_clf.predict((vec))
	print "%d RF Classifier: " %rfc_clf.predict((vec))
	print "%d Naive Bayes: "%nb_clf.predict((vec))
	print "%d Decision Tree: "%dt_clf.predict((vec))
	'''