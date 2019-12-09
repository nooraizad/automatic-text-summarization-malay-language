import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
import re

df = pd.read_csv("data2.csv",encoding = "ISO-8859-1")
df.head()
df['article_text'][0]

# Split text into sentences

from nltk.tokenize import sent_tokenize
sentences = []
for s in df['article_text']:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list

sentences[:12]

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

clean_sentences[:12]

%%time
import malaya

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('malay.txt')

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

clean_sentences[:12]

# Extract word vectors
word_embeddings = {}
f = open('ms.tsv')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    word_embeddings[word] = coefs
f.close()

len(word_embeddings)

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((300,))
  sentence_vectors.append(v)

  # Similarity Matrix preparation

# similarity matrix
sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]

  import networkx as nx
import pylab as plt

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Extract top 10 sentences as the summary
for i in range(2):
  print(ranked_sentences[i][1])























