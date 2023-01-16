#veri seti = https://github.com/aitsc/WSIM/tree/main/datasets/Second%20dataset

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence ,TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os


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

path_manu= "/content/drive/MyDrive/kodlamalar/okul_projesi/nlp/Dataset/manuscripts/"
path_revi= "/content/drive/MyDrive/kodlamalar/okul_projesi/nlp/Dataset/reviewers/"
path_ground = "/content/drive/MyDrive/kodlamalar/okul_projesi/nlp/Dataset/groundturth.txt"

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

def stemming(veriler):
  for i in range(len(veriler)):
    veriler[i]["metin"]  = " ".join([stemmer.stem(word) for word in veriler[i]["metin"].split() ])
  return veriler

manuscripts_veriler=stemming(manuscripts_veriler)
revievers_veriler=stemming(revievers_veriler)


def jaccard(word_tokens1, word_tokens2):
# kelimeri birlestir
	butun_kelimeler = word_tokens1.split() + word_tokens2.split()
	birlesim_kumesi = set(butun_kelimeler)

	# Kesisim hesapla.
	kesisim = set()
	for w in word_tokens1:
		if w in word_tokens2:
			kesisim.add(w)

	jaccard_degeri = len(kesisim)/len(birlesim_kumesi)
	return jaccard_degeri


def jaccard_hesapla(man_ve, rev_ve):
  veriler = []
  
  for val_ma in man_ve:
    veri={
      "manuscript_isim":val_ma["isim"],
      "sonuclar":[]
    }
    list_ = []
    for val_re in rev_ve:
      veri1={
          "reviever_name":val_re["isim"],
          "jaccard_skor":jaccard(val_ma["metin"],val_re["metin"])
      }
      list_.append(veri1)
    
    veri.update({"sonuclar":list_})
    veriler.append(veri)
  return veriler

jaccard_skorları = jaccard_hesapla(manuscripts_veriler,revievers_veriler)

def get_db(veriler):
  list_ =[]
  for val in veriler:
    veri={
        "manuscript_isim":val["manuscript_isim"],
        "db":pd.DataFrame(val["sonuclar"])
    }
    veri["db"]=veri["db"].sort_values('jaccard_skor',ascending=False ,ignore_index = True )
    veri["db"]=veri["db"][:20]
    list_.append(veri)
  return list_

db_jaccard=get_db(jaccard_skorları) 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_tfidf_similarity(word_tokens1,word_tokens2):

  corpus = [word_tokens1,word_tokens2]
  # Use sklearn to generate document term matrix
  vectorizer = TfidfVectorizer()
  document_term_matrix = vectorizer.fit_transform(corpus)

  pairwise_similarity = document_term_matrix * document_term_matrix.transpose()

  # Show the document similarity matrix
  arr=pairwise_similarity[0].toarray()
  ls = arr.tolist()[0][1]

  return ls

def tf_idf_hesapla(man_ve, rev_ve):
  veriler = []
  i=0;
  for val_ma in man_ve:
      veri={
        "manuscript_isim":val_ma["isim"],
        "sonuclar":[]
      }
      list_ = []
      for val_re in rev_ve:
        veri1={
            "reviever_name":val_re["isim"],
            "tf_idf_skor":process_tfidf_similarity(val_ma["metin"],val_re["metin"])
        }
        list_.append(veri1)
    
        veri.update({"sonuclar":list_})
        veriler.append(veri)
        return veriler

tf_idf_skorları = tf_idf_hesapla(manuscripts_veriler,revievers_veriler)

def get_db_tf(veriler):
  list_ =[]
  for val in veriler:
    veri={
        "manuscript_isim":val["manuscript_isim"],
        "db":pd.DataFrame(val["sonuclar"])
    }
    veri["db"]=veri["db"].sort_values('tf_idf_skor',ascending=False ,ignore_index = True )
    veri["db"]=veri["db"][:20]
    list_.append(veri)
  return list_

db_tf_idf=get_db_tf(tf_idf_skorları)



db_tf___ = pd.DataFrame(db_tf_idf)
db_jaccard___ = pd.DataFrame(db_jaccard)

db_tf___.to_csv("/content/drive/MyDrive/kodlamalar/okul_projesi/nlp/Dataset/db_tf.csv")
db_jaccard___.to_csv("/content/drive/MyDrive/kodlamalar/okul_projesi/nlp/Dataset/db_jaccard.csv")
