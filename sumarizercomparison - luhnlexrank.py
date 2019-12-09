import re
from collections import defaultdict

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
 
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#name of the plain-text file ~ bbc news dataset
file = '010.txt'
parser = PlaintextParser.from_file(file, Tokenizer('english'))

# LexRank summarizer

from sumy.summarizers.lex_rank import LexRankSummarizer 
summarizer = LexRankSummarizer()
#Summarize the document with 2 sentences
summary = summarizer(parser.document, 2) 
for sentence in summary:
 print(sentence)


# Luhn Summarizer

 from sumy.summarizers.luhn import LuhnSummarizer
summarizer_1 = LuhnSummarizer()
summary_1 =summarizer_1(parser.document,2)
for sentence in summary_1:
 print(sentence)

# Edmundson Summarizer

print ("--EdmundsonSummarizer--")     
summarizer = EdmundsonSummarizer() 
words = ("deep", "learning", "neural" )
summarizer.bonus_words = words
     
words = ("another", "and", "some", "next",)
summarizer.stigma_words = words
    
     
words = ("another", "and", "some", "next",)
summarizer.null_words = words
for sentence in summarizer(parser.document, 2):
        print(sentence)  
