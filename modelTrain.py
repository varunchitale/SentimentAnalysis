import gensim
#from nltk import RegexpTokenizer
import nltk
import gensim
from textPreProc import normalize, basicReplacement, remSW
#from remSF import remSF
import numpy as np



class LabeledLineSentence(object):

    def __init__(self, doc_list, labels_list):

        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):

        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])


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

print sent
print labels


corpus = sent
lines = [nltk.word_tokenize(basicReplacement(sent.decode('utf-8'))) for sent in corpus]


it = LabeledLineSentence(lines, labels)


model = gensim.models.Doc2Vec(size=30, dm=0, min_count=1, window = 5, workers=4)
model.build_vocab(it)

#training of model
for epoch in range(100):
		print "Iteration: "+str(epoch)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
		model.train(it, epochs = model.iter, total_examples = len(lines))
model.save('d2v.model')
