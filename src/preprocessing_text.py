import os
from multiprocessing import Pool
import pandas as pd
from src.utils import clean_text, lemmatize_rus_text, tokenize_ru



def preprocessing(df):
    df["text"] = df["text"].apply(clean_text)
    text = df["text"].values
    with Pool(processes=os.cpu_count()-2) as pool:
        lemma = pool.map(lemmatize_rus_text, text)
    df["lemma"] = lemma
    df["tokenize_text"] = df["lemma"].apply(tokenize_ru)
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"D:/NLP/Dataset/lenta-news-25k.csv")
    df = preprocessing(df)
    df.to_csv("lenta-news-25k.csv", index=False)

