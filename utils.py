import pandas as pd
import spacy
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import torch
from allennlp.predictors.predictor import Predictor
import tensorflow_hub as hub
from tqdm import tqdm

if torch.cuda.is_available():
    cuda_device = 0 #TODO: is there a non hard-code way?
else:
    cuda_device = -1

sp = spacy.load('en_core_web_sm')
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
open_info_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
                                                       cuda_device=cuda_device)
coref_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
                                                    cuda_device=cuda_device)

def remove_wordy(s, wordy_list):
    for i in wordy_list:
        s=s.replace(i, "")
    return s

def add_stemmed_col_to_df(df, speeches_col, stemmed_col):
    """
    :param df: dataframe containing at least a column with speeches
    :return: new dataframe, with added column for stemmed speeches
    """

    tqdm.pandas()
    ps = PorterStemmer()

    stemmed = df[speeches_col].progress_apply(lambda x: " ".join([ps.stem(w.lower()) for w in word_tokenize(x)]))
    df[stemmed_col] = stemmed

    return df


if __name__ == "__main__":
    speeches = pd.read_pickle("all_speech_sentence_filtered.pkl")
    new_speeches = add_stemmed_col_to_df(speeches, "Questions", "Stemmed")
    new_speeches.to_pickle("all_speech_filtered_stemmed.pkl")

    # check the new dataset
    new_speeches = pd.read_pickle("all_speech_filtered_stemmed.pkl")
    print(len(new_speeches))
    print(new_speeches.iloc[0].Questions)
    print(new_speeches.iloc[0].Stemmed)


