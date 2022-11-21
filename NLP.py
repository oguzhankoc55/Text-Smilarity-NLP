import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
# from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

dosya_konumu=os.getcwd()
print(dosya_konumu)

def make_data(path):
  veriler = []
  pdfler = os.listdir(path)
  for pdf in pdfler:
    try:
        veri= {
            "isim":"",
        }
        veri.update({"isim" : pdf})
        dosya = open(path+pdf,"r")
        for metin in dosya:
            veri.update({"metin":metin})
        
        dosya.close()
        veriler.append(veri)
    except:
        print( "hata")
  return veriler

def refactor(path):
    results = []
    dosya = open(path,"r")
    for metin in dosya:
        veri= {
                "isim":"",
            }
        result = metin.replace("\n","").split("\t")
        _List = []
        for i in range(len(result)):
            if(i == 0 ):
                veri.update({"isim" : result[i]})
            else:
                _List.append({str(i):result[i]})
            veri.update({"revievers":_List})  
            results.append(veri)

    return results


#adding datasets
path_manu= dosya_konumu+"\\Dataset\\manuscripts\\"+"0902.1601.txt"
path_revi= dosya_konumu+"\\Dataset\\reviewers\\"+"aalto, s..txt"
path_ground = dosya_konumu+"\\Dataset\\groundturth.txt"

manuscripts_veriler = make_data(path_manu)
revievers_veriler = make_data(path_revi)
groundturth_veriler = refactor(path_ground)

def To_lower(veriler):
  for i in range(len(veriler)):
    veriler[i]["metin"] = veriler[i]["metin"].lower()
    veriler[i]["metin"] = veriler[i]["metin"].replace("\n"," ").replace("\t"," ")
    veriler[i]["metin"] = re.sub(r'[^a-zA-Z\s]', ' ', veriler[i]["metin"])
  return veriler

manuscripts_veriler=To_lower(manuscripts_veriler)
revievers_veriler=To_lower(revievers_veriler)

nltk.download('stopwords') 
stop_words =  set(stopwords.words('english'))

def del_stopWords(veriler):
  for i in range(len(veriler)):
    veriler[i]["metin"]  = " ".join([c for c in veriler[i]["metin"].split() 
    if c not in stop_words if len(c)>1])
  return veriler

manuscripts_veriler = del_stopWords(manuscripts_veriler)
revievers_veriler= del_stopWords(revievers_veriler)

stemmer = nltk.porter.PorterStemmer()

metin = ' '.join([stemmer.stem(word) for word in manuscripts_veriler[1]["metin"].split()])
print(metin)

print(manuscripts_veriler[1]["metin"])