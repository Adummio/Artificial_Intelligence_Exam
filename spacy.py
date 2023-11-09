import string
import spacy
from stop_words import get_stop_words
from spacy.lang.it import Italian
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


def spacy_tokenizer(sentence):
    
    mytokens = parser(sentence)

    # Lemmatizzazione
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Rimozione stopwords
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    #ritorna una lista di token senza stopwords
    return mytokens


#transformer con spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
    # Rimuove gli spazi e converte il testo in lowercase
    return text.strip().lower()


punctuations = string.punctuation

#lista di stopwords
nlp = spacy.load('it')
stop_words = get_stop_words('it')

df_castello = pd.read_csv("reviews.csv", sep="\t")


# Italian tokenizer
parser = Italian()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)


X = df_castello['reviews'] # cosa vogliamo analizzare
ylabels = df_castello['feedback'] # la label da testare

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)


classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)

from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Accuratezza del modello
print(metrics.accuracy_score(y_test, predicted))
print(metrics.precision_score(y_test, predicted))
print(metrics.recall_score(y_test, predicted))
