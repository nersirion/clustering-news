import re
import numpy as np
import pandas as pd
from pymystem3 import Mystem
from nltk.corpus import stopwords


def clean_numbers(test:str) -> str:
    text = re.sub('\d{5+}', '#####', text )
    text = re.sub('\d{4}', '####', text )
    text = re.sub('\d{3}', '###', text )
    text = re.sub('\d{2}', '##', text)
    text = re.sub('\d', '#', text)
    return text

def clean_text(test:str) -> str:
    text = str(text).lower()
    text = re.sub('[^а-я0-9]', ' ', text)
    text = clean_numbers(text)
    return text

def lemmatize_rus_text(test:str) -> str:
    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords              and token != " "] 

    text = " ".join(tokens)
    
    return text

def tokenize_ru(test:str) -> list:
    tokens = text.split()
    return tokens

def create_map_dict(topic:list) -> dict:
    numbers = np.arange(len(topic))
    topic_dict = dict(zip(topic, numbers))
    return topic_dict

def mean_vector(tokens):
    vec = 0
    skip_words = 0
    for word in tokens:
        try:
            vec+=model.wv.get_vector(word)
        except:
            skip_words+=1
    return vec/(len(tokens)-skip_words)

def get_top_keywords(data, clusters, words, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i+1))
        print(','.join([words[t] for t in np.argsort(r)[-n_terms:]]))
        
def show_chart_clusters(kmeans, clust_labels, data_matrix):
    clust_centers = kmeans.cluster_centers_

    embeddings_to_tsne = np.concatenate((data_matrix ,clust_centers), axis=0)

    tSNE =  TSNE(n_components=2, perplexity=15)
    tsne_embeddings = tSNE.fit_transform(embeddings_to_tsne)
    tsne_embeddings, centroids_embeddings = np.split(tsne_embeddings, [len(clust_labels)], axis=0)
    
    clust_indices = np.unique(clust_labels)

    clusters = {clust_ind : [] for clust_ind in clust_indices}
    for emb, label in zip(tsne_embeddings, clust_labels):
        clusters[label].append(emb)

    for key in clusters.keys():
        clusters[key] = np.array(clusters[key])
    colors = cm.rainbow(np.linspace(0, 1, len(clust_indices)))
    
    plt.figure(figsize=(10,10))
    for ind, color in zip(clust_indices, colors):
        x = clusters[ind][:,0]
        y = clusters[ind][:,1]
        plt.scatter(x, y, color=color)

        centroid = centroids_embeddings[ind]
        plt.scatter(centroid[0],centroid[1], color='b', marker='x', s=100)

    plt.show()


def fit_kmeans(matrix, n_cluster:int):
    km = MiniBatchKmeans(n_cluster = n_cluster, random_state=999)
    kmeans_matrix = km.fit_predict(matrix)
    return kmeans_matrix



def cut_often_and_rare_words(tf_idf, lower_thresh, upper_thresh):
    not_often = tf_idf.idf_ > lower_thresh
    not_rare = tf_idf.idf_ < upper_thresh
    mask = not_often * not_rare
    good_words = np.array(td_idf.get_feature_names())[mask]
    return good_words

def good_words_to_vocab(good_words:np.array) -> dict:
    vocab = dict(zip(good_words, np.arange(len(good_words))))
    return vocab


