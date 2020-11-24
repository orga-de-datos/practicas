# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Introducción a NLP
#
# En este notebook vamos a dar los primeros pasos en el procesamiento de lenguaje natural. Para eso, vamos a utilizar un [dataset](https://github.com/jbesomi/texthero/tree/master/dataset/Superheroes%20NLP%20Dataset) de superhéroes para intentar predecir si un superhéroe es bueno o malo en base a la descripción de su historia, utilizando Bag of Words y TF-IDF. 

# +
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import (
    RegexpTokenizer,
    TreebankWordTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

pd.options.display.max_columns = None
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# -

pd.set_option('display.max_colwidth', 100)
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbesomi/texthero/master/dataset/superheroes_nlp_dataset.csv"
)
df.head()

pd.options.display.max_colwidth = None
df_txt = (
    df.loc[df.alignment.isin(['Good', 'Bad']), ['name', 'history_text', 'alignment']]
    .dropna()
    .reset_index(drop=True)
)
df_txt.head()

# Preprocesamos la columna alignment
le = LabelEncoder()
df_txt['alignment'] = le.fit_transform(df_txt['alignment'])


# ### Funciones Auxiliares

# +
def helper(train, train_label, model):
    cv = []
    pred_based_on_cv = pd.DataFrame(data=np.zeros(shape=(train.shape[0], 2)))
    kfold = KFold(n_splits=5, shuffle=True, random_state=2019)
    for t_index, v_index in kfold.split(train_label.ravel()):
        xtrain, ytrain = train[t_index, :], train_label[t_index]
        xtest, ytest = train[v_index, :], train_label[v_index]

        model.fit(xtrain, ytrain)
        pred_based_on_cv.loc[v_index, :] = model.predict_proba(xtest)
        cv.append(roc_auc_score(ytest, pred_based_on_cv.loc[v_index, 1]))
    return (np.mean(cv), pred_based_on_cv)


def plotting_helper(cv, pred, label):
    print("AUC CV score : %s" % cv)
    plt.figure(figsize=(9, 5))
    sns.heatmap(
        confusion_matrix(label, np.argmax(pred.values, axis=1)).round(2),
        annot=True,
        fmt='g',
    )
    plt.title("Accuracy : %s" % accuracy_score(np.argmax(pred.values, axis=1), label))
    return (accuracy_score(np.argmax(pred.values, axis=1), label), cv)


# -

# ## Tokenización 

# WordPunctTokenizer
wordpunct_tokenizer = WordPunctTokenizer()
pow_tokens = wordpunct_tokenizer.tokenize(' '.join(df_txt.history_text)[:200])
print(pow_tokens)

# TreebankWordTokenizer
treebank_tokenizer = TreebankWordTokenizer()
pow_tokens = treebank_tokenizer.tokenize(' '.join(df_txt.history_text)[:200])
print(pow_tokens)

# RegexpTokenizer
capword_tokenizer = RegexpTokenizer('[A-Z]\w+')
pow_tokens = capword_tokenizer.tokenize(' '.join(df_txt.history_text)[:200])
print(pow_tokens)

# ## Normalización

# ### Stemming

# SnowballStemmer
s_stemmer = SnowballStemmer(language='spanish')
words = ['run','runner','running','ran','runs','feet', 'cats', 'cacti']
for word in words:
    print(f'{word} --> {s_stemmer.stem(word)}')


# ### Lemmatization

# +
# WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = ['run','runner','running','ran','runs','feet', 'cats', 'cacti']
for word in words:
    print(f'{word} --> {lemmatizer.lemmatize(word)}')
# -

# # Bag of Words

# 1. Remover palabras no deseadas (stop words)
# 2. Crear tokens
# 3. Aplicar tokenización al texto
# 4. Crear vocabulario y generar vectores

# Armamos un set con las stop words
stop_words = set(stopwords.words('english'))
#stopwords.words('english')

# +
count_vec = CountVectorizer(
    stop_words=stop_words
)
count_vec.fit(df_txt['history_text'].values.tolist())
df_count_vec = count_vec.transform(df_txt['history_text'].values.tolist())
df_count_vec.shape

# Podemos observar los features que seleccionó el modelo
print(count_vec.get_feature_names()[500:510])
print()

print("Cantidad de features: " + str(len(count_vec.get_feature_names())))
print()
# -

# Después de ejecutar el modelo obtenemos la siguiente matriz
# Las dimensiones son las filas del dataset x la cantidad de features
print(df_count_vec.toarray().shape)
print()
print(df_count_vec.toarray())

print("Primera fila de la matriz:", df_count_vec.toarray()[0][:10], "\n")
print("Total features primer fila:", df_count_vec.toarray()[0].sum(), "\n")
print("Primeras cinco features:\n")
display(
    pd.Series(count_vec.get_feature_names())[df_count_vec.toarray()[0] == 1].head(5)
)

# Palabras por frecuencia de aparicion
top_words = pd.Series(
    df_count_vec.toarray().sum(axis=0), index=count_vec.get_feature_names()
).sort_values(ascending=False)
top_words.index.name = 'word'
top_words = top_words.to_frame('count').reset_index()
top_words.head(10)

plt.figure(dpi=150)
sns.barplot(x="word", y="count", data=top_words[:20])
plt.ylabel("Frecuencia")
plt.xlabel("Palabras")
plt.title("Frecuencia de palabras")
plt.xticks(rotation=90)
plt.show()

# +
# nltk también ofrece una forma de graficar la frecuencia de las palabras
tokenizer = RegexpTokenizer(r'\w+')
pow_tokens = tokenizer.tokenize(' '.join(df_txt.history_text.str.lower()))

pow_tokens = [token for token in pow_tokens if token not in stop_words]

pow_freq_dist = nltk.FreqDist(pow_tokens)
plt.figure(figsize=(15, 5))
plt.xticks(fontsize=20)
pow_freq_dist.plot(25)
# -

cv, pred = helper(df_count_vec, df_txt.alignment, MultinomialNB())
print("AUC CV score:", cv)
count_acc_mnb, count_auc_cv_mnb = plotting_helper(cv, pred, df_txt.alignment.values)


# ### Otra opción 

# +
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.rt = RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.rt.tokenize(doc)]
    
count_vec = CountVectorizer(
    stop_words=stop_words_new,
    tokenizer=LemmaTokenizer(),
    ngram_range=(2,2)
)
count_vec.fit(df_txt['history_text'].values.tolist())
df_count_vec = count_vec.transform(df_txt['history_text'].values.tolist())
df_count_vec.shape
# -

# Palabras por frecuencia de aparicion
top_words = pd.Series(
    df_count_vec.toarray().sum(axis=0), index=count_vec.get_feature_names()
).sort_values(ascending=False)
top_words.index.name = 'word'
top_words = top_words.to_frame('count').reset_index()
top_words.head(10)

plt.figure(dpi=150)
sns.barplot(x="word", y="count", data=top_words[:20])
plt.ylabel("Frecuencia")
plt.xlabel("Palabras")
plt.title("Frecuencia de palabras")
plt.xticks(rotation=90)
plt.show()

cv, pred = helper(df_count_vec, df_txt.alignment, MultinomialNB())
print("AUC CV score:", cv)
count_acc_mnb, count_auc_cv_mnb = plotting_helper(cv, pred, df_txt.alignment.values)

# ### Cómo aplicamos TF-IDF
#
# Como vimos antes CountVectorizer nos devuelve la cantidad de apariciones de cada palabra en un documento. De esta forma, nos devuelve la _frecuencia_ de cada palabra.

# +
# Al resultado de CountVectorizer le aplicamos TfidfTransformer para obtener el valor de TF-IDF
tfidf_transformer = TfidfTransformer() 
tfidf_transformer.fit(df_count_vec)
tf_idf_vector = tfidf_transformer.transform(df_count_vec)

print(tf_idf_vector.toarray().shape)
print()
print(tf_idf_vector.toarray())

# +
first_document_vector=tf_idf_vector[0] 

# Imprimimos el resultado para el primer documento
df = pd.DataFrame(first_document_vector.T.todense(), index=count_vec.get_feature_names() , columns=["tfidf"]) 
df.sort_values(by=["tfidf"],ascending=False)
# -

df_txt.loc[0]['history_text']

# Vemos que las palabras que tienen valor 0 son las que no se encuentran en el documento al que hace referencia el vector. 
# Mientras menos común sea la palabra en nuestro corpus, mayor puntaje va a tener.  

# +
# Otra opción es usar directamente TfidfVectorizer que es el equivalente a usar CountVectorizer + TfidfTransformer

tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
tfidf_vec.fit(df_txt['history_text'].values.tolist())
df_tfidf_vec = tfidf_vec.transform(df_txt['history_text'].values.tolist())

print(df_tfidf_vec.toarray().shape)
print()
print(df_tfidf_vec.toarray())

# +
df_tfidf_first_document_vector = df_tfidf_vec[0] 

# Imprimimos el resultado para el primer documento
df = pd.DataFrame(df_tfidf_first_document_vector.T.todense(), index=tfidf_vec.get_feature_names() , columns=["tfidf"]) 
df.sort_values(by=["tfidf"],ascending=False)
# -

cv, pred = helper(df_tfidf_vec, df_txt.alignment, MultinomialNB())
print("AUC CV score:", cv)
count_acc_mnb, count_auc_cv_mnb = plotting_helper(cv, pred, df_txt.alignment.values)
