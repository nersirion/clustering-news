import pandas as pd
from utils import clean_text, lemmatize_rus_text, tokenize_ru



def preprocessing(df):
    df["text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(lemmatize_rus_text)
    df["tokenize_text"] = df["text"].apply(tokenize_ru)
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"D:/NLP/Dataset/lenta-news-50k.csv")
    df = preprocessing(df)
    df.to_csv("lenta-news-50k.csv", index=False)

