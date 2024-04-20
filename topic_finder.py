import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English
import IPython
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora
import pickle
import pyLDAvis.gensim
import random
nltk.download('stopwords')
nltk.download('wordnet')
parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

text_data = []
df = pd.read_csv('data/out.csv')

for index, row in df.iterrows():
    # sentiment = int(row['LINK_SENTIMENT'])
    item = row['SOURCE_SUBREDDIT']
    if isinstance(row['TEXT'], str) and item == 'subredditdramas':
        tokens = prepare_text_for_lda(row['TEXT'])
        text_data.append(tokens)




dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('data/dictionary.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('data/model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)



dictionary = gensim.corpora.Dictionary.load('data/dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('data/model5.gensim')

visualisation = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(visualisation, 'data/LDA_Visualization.html')
