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

import matplotlib.pyplot as plt
import nltk

# +
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xgboost as xgb
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import (
    RegexpTokenizer,
    TreebankWordTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

pd.options.display.max_columns = None
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# -

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

le = LabelEncoder()
df_txt['alignment'] = le.fit_transform(df_txt['alignment'])


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


# +
param_xgb = {}
param_xgb['objective'] = 'binary:logistic'
param_xgb['learning_rate'] = 0.1
param_xgb['seed'] = 666
param_xgb['eval_metric'] = 'auc'


def xgb_helper(train, train_label):
    cv = []
    pred_based_on_cv = pd.Series(data=np.zeros(train.shape[0]))
    kfold = KFold(n_splits=5, shuffle=True, random_state=2019)
    for t_index, v_index in kfold.split(train_label.ravel()):
        xtrain, ytrain = train.loc[t_index, :], train_label[t_index]
        xtest, ytest = train.loc[v_index, :], train_label[v_index]
        trainset = xgb.DMatrix(xtrain, label=ytrain)
        testset = xgb.DMatrix(xtest, label=ytest)
        model = xgb.train(
            list(param_xgb.items()),
            trainset,
            evals=[(trainset, 'train'), (testset, 'test')],
            num_boost_round=5000,
            early_stopping_rounds=200,
            verbose_eval=200,
        )
        pred_based_on_cv.loc[v_index] = model.predict(
            testset, ntree_limit=model.best_ntree_limit
        )

    cv.append(roc_auc_score(ytest, pred_based_on_cv.loc[v_index]))
    return (np.mean(cv), pred_based_on_cv)


# -

# # Bag of Words

# +
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


stop_words = set(stopwords.words('english'))
count_vec = CountVectorizer(
    stop_words=stop_words
    # tokenizer=LemmaTokenizer()  #TODO
    # ngram_range=(1,3),
)
count_vec.fit(df_txt['history_text'].values.tolist())
df_count_vec = count_vec.transform(df_txt['history_text'].values.tolist())
df_count_vec.shape
# -

display(pd.Series(count_vec.get_feature_names()).head(10))
display(pd.Series(count_vec.get_feature_names()).sample(5))

print("Vector primer fila:", df_count_vec.toarray()[0][:10])
print("Total palabras primer fila:", df_count_vec.toarray()[0].sum())
print("Primeras cinco:")
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

fig = px.bar(
    top_words[:20],
    x='word',
    y='count',
    title='Palabras con mas apariciones',
    template='plotly_white',
    labels={'ngram': 'Bigram', 'count': 'Count'},
)
fig.update_layout(autosize=False, width=1000)
fig.show()

# +
# cv, pred = xgb_helper(pd.DataFrame.sparse.from_spmatrix(df_count_vec), df_txt.alignment.values)
# count_acc_xgb, count_auc_cv_xgb = plotting_helper(cv, pred, df_txt.alignment.values)

# +
# from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer, WordPunctTokenizer, TreebankWordTokenizer

tokenizer = RegexpTokenizer(r'\w+')
# tokenizer = WhitespaceTokenizer()
pow_tokens = tokenizer.tokenize(' '.join(df_txt.history_text.str.lower()))

pow_tokens = [token for token in pow_tokens if token not in stop_words]

pow_freq_dist = nltk.FreqDist(pow_tokens)
plt.figure(figsize=(15, 5))
plt.xticks(fontsize=20)
pow_freq_dist.plot(25)

# +
# TODO - Nube de palabras
# -

cv, pred = helper(df_count_vec, df_txt.alignment, MultinomialNB())
print("AUC CV score:", cv)
# count_acc_mnb, count_auc_cv_mnb = plotting_helper(cv, pred, df_txt.alignment.values)

# +
# TDOD - Reduccion dimensiones
# TODO - n-grams
# -

# # TfidfVectorizer

tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
tfidf_vec.fit(df_txt['history_text'].values.tolist())
df_tfidf_vec = tfidf_vec.transform(df_txt['history_text'].values.tolist())
df_tfidf_vec.shape

cv, pred = helper(df_tfidf_vec, df_txt.alignment, MultinomialNB())
print("AUC CV score:", cv)
# count_acc_mnb, count_auc_cv_mnb = plotting_helper(cv, pred, df_txt.alignment.values)
