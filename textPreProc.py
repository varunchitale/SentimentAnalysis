#Pre Processing Text Files, Varun Chitale 08/17/2017

import nltk
import re
from nltk.stem import PorterStemmer

length = 0
sentences1=''

def basicReplacement(text):
	text = text.lower()
	
	#text = text.replace('\"',' ')
	text = text.replace(';',' ')
	text = text.replace('-',' ')
	text = text.replace('.',' ')
	

#removes the bullets: 1. 2. etc
	text = re.sub(r'(\ )*[0-9]\.','',text)
	
	#PreProcessing Tweak: Replace any amount(in rupees) by a constant word
	text = re.sub(r'(rs|rs\.|rupees)(\ )*[0-9\,\-\.\/]+','monetary_value',text)
	
	#Replace all dates by a keyword
	text = re.sub(r'(\d{2}(\-|\.|\/|\ )*([12]\d{3}))','some_date',text)
	text = text.replace(',',' ')
	
#removes junk characters	
	#text = re.sub(r'^[a-zA-Z0-9 \.]','',text)
	return text

def remSW(text):
	with open ('stopWords.txt') as f:
		stopWords = f.read().split()

	text1 = " ".join(word for word in text.split() if word not in stopWords)

	return text1


def normalize(text):
	ps = PorterStemmer()
	text1=""
	for word in text.split():
		text1 += ps.stem(word)
		text1 += " " 

	return text1


#text = open('text1.txt').read()
#print  normalize(remSW(basicReplacement(text)))
