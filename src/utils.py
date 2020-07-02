import re
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymystem3 import Mystem
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans

def clean_numbers(text:str) -> str:
    text = re.sub('\d{5+}', '#####', text )
    text = re.sub('\d{4}', '####', text )
    text = re.sub('\d{3}', '###', text )
    text = re.sub('\d{2}', '##', text)
    text = re.sub('\d', '#', text)
    return text

def clean_text(text:str) -> str:
    text = text.lower()
    text = re.sub('[^а-я0-9]', ' ', text)
    text = clean_numbers(text)
    return text

def lemmatize_rus_text(text:str) -> str:
    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords and token != " "]
    text = " ".join(tokens)
    return text

def tokenize_ru(text:str) -> list:
    tokens = text.split()
    return tokens

def create_map_dict(topic:list) -> dict:
    numbers = np.arange(len(topic))
    topic_dict = dict(zip(topic, numbers))
    return topic_dict

def mean_vector(tokens:list) -> np.array:
    vec = 0
    skip_words = 0
    for word in tokens:
        try:
            vec+=model.wv.get_vector(word)
        except:
            skip_words+=1
    return vec/(len(tokens)-skip_words)

def df_top_keywords(data, clusters:list, words, clusters_names:dict):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    data = []
    for i, row in df.iterrows():
        data.append([words[t] for t in np.argsort(row)[-10:]])
    df = pd.DataFrame(data).rename(clusters_names).T
    df.to_csv('dataset/top10words.csv', index=False)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'

    plt.figure()
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def fit_kmeans(matrix, n_cluster:int):
    km = MiniBatchKMeans(n_clusters = n_cluster, random_state=999)
    labels = km.fit_predict(matrix)
    return labels

def get_lda_labels(lda_matrix) -> list:
    return [np.argmax(i) for i in lda_matrix]

def delete_pos_tag(model_vocab:dict) -> dict:
    new_vocab = {}
    for keyword, vec in model_vocab.items():
        keyword = re.sub('[^а-я]', '',keyword)
        new_vocab[keyword] = vec
    model_vocab = new_vocab
    return model_vocab

def save_model(model, name:str = 'lda'):
    joblib.dump(model, f'models/{name}')
