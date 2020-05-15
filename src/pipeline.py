import joblib
import numpy as np
import pandas as pd
from src.preprocessing_text import preprocessing
from src.utils import get_lda_labels


def pipeline(max_text:int=100) -> np.array:
    df = pd.read_csv("dataset/to_lda_analyze.csv")
    if len(df)<=max_text:
        df = preprocessing(df)
        models_names = os.listdir(path)
        for model_name in models_names:
            models_dict = {model: joblib.load(model)}

        predicts = []
        for i in range(4):
            model_name = f"count_vec{i+1}"
            count_vec = models_dict[model_name].trasnsorm(df["lemma"])
            model_name = f"lda{i+1}"
            lda_matrix = models_dict[model_name].transform(count_vec)
            lda_labels = get_lda_labels(lda_matrix)
            predicts.append(lda_labels)
        pred_df = pd.DataFrame(predicts).T
        final_predict = models_dict["Knn"].predict(pred_df)
        return final_predict
    else:
        print("Too many rows in df")





