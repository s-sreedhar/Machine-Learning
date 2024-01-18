import nltk as nltk
#nltk.download('all')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

text = "NLTK provides a suite of text processing tools, including tokenization, stemming, tagging, parsing, and semantic reasoning. It's a powerful library for exploring and analyzing text data. Let's experiment with its capabilities!"
text=text.strip(",./\' ")
sentences=sent_tokenize(text)
words=word_tokenize(text)
