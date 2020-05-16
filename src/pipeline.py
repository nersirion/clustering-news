import os
import joblib
import numpy as np
import pandas as pd
from utils import clean_text, lemmatize_rus_text, tokenize_ru, get_lda_labels


def pipeline(max_text:int=100) -> np.array:
    df = pd.read_csv("../dataset/to_lda_analyze.csv")
    if len(df)<=max_text:
        df["text"] = df["text"].apply(clean_text)
        df["text"] = df["text"].apply(lemmatize_rus_text)
        models_names = os.listdir("../models")
        models_dict = {}
        for model_name in models_names:
            path = f"../models/{model_name}"
            models_dict[model_name]=joblib.load(path)
        predicts = []
        for i in range(4):
            model_name = f"count_vec{i}.joblib"
            count_vec = models_dict[model_name].transform(df["text"])
            model_name = f"lda{i}.joblib"
            lda_matrix = models_dict[model_name].transform(count_vec)
            lda_labels = get_lda_labels(lda_matrix)
            predicts.append(lda_labels)
        pred_df = pd.DataFrame(predicts).T
        final_predict = models_dict["Knn.joblib"].predict(pred_df)
        clusters_names = {3: "Россия",
                         1: "Экономика",
                         6: 'Интернет и СМИ',
                         7: 'Спорт',
                         0: 'Наука и техника',
                         5: "Культура",
                         2: 'Мир',
                         4: 'Бывший СССР',
                         9: 'Из жизни',
                         8: 'Дом'}
        return final_predict, clusters_names
    else:
        print("Too many rows in df")





