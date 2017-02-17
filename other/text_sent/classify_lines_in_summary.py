# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

################################################################################
# Split text into sentences
################################################################################
# Method 1 - gets tripped up on abbreviations, e.g. J.D.
# import nltk.data
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# fp = open('pines.txt', 'rb')
# data = fp.read()
# print '\n-----\n'.join(tokenizer.tokenize(data))


# Method 2 - 
# http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
import re
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

sents = split_into_sentences(open('pines.txt', 'rb').read())

################################################################################
# Classify each sentence
################################################################################

from senti_classifier import senti_classifier
positives = []
negatives = []
for sent in sents:
# sentences = ['The movie was the worst movie']
    pos_score, neg_score = senti_classifier.polarity_scores([sent])
    print sent
    print pos_score, neg_score
    print '-' * 5
    positives.append(pos_score)
    negatives.append(neg_score)

import matplotlib.pylab as plt
n = len(sents)
plt.plot(range(n), positives, range(n), negatives)
plt.show()

