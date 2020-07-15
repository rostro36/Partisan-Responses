import pandas as pd
import spacy
import re
import torch
from allennlp.predictors.predictor import Predictor
import tensorflow_hub as hub

if torch.cuda.is_available():
    cuda_device = 0 #TODO: is there a non hard-code way?
else:
    cuda_device = -1

sp = spacy.load('en_core_web_sm')
# model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# open_info_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
#                                                        cuda_device=cuda_device)
# coref_extractor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
#                                                     cuda_device=cuda_device)

def read_full_speech(filepath):
    """
    Load into a dataframe the data from a text file in hein-bound
    :param filepath: filepath
    :return: speeches dataframe
    """
    with open(filepath, errors="ignore") as f:
        speech = f.readlines()
        speech = [s.strip() for s in speech]
        speech = [[s[:s.find('|')], s[s.find('|') + 1:]] for s in speech]
        speech_df = pd.DataFrame(speech[1:], columns=speech[0])
    return speech_df

def read_speakermap(filepath):
    """
    Load into a dataframe the speaker map from a text file in hein-bound
    :param filepath: filepath
    :return: speakermap dataframe
    """
    with open(filepath, errors="ignore") as f:
        speakermap_df = pd.read_table(f, delimiter = "|")
    return speakermap_df

def merge_speech_speaker(speech_df, speaker_df):
    """
    Merge a dataframe containing speeches with one containing the speakermap
    :param speech_df: speeches dataframe
    :param speaker_df: speakermap dataframe
    :return: merged dataframe
    """
    speech_df = speech_df.astype({"speech_id": type(speaker_df.loc[:,'speech_id'][0])})
    return speaker_df.merge(speech_df, on="speech_id", how="left")

def get_speeches_filename(idx):
    """
    Returns the name of the speeches file given an index
    :param idx: index
    :return: filename
    """
    return "speeches_{}.txt".format(idx)

def get_speakermap_filename(idx):
    """
    Returns the name of the speakermap file given an index
    :param idx: index
    :return: filename
    """
    return "{}_SpeakerMap.txt".format(idx)

def lemmatize(phrase):
    """
    Given some text, it returns the lemmatized text
    :param phrase: text
    :return: lemmatized text
    """
    return " ".join([word.lemma_ for word in sp(phrase)])

def change_comma(speech):
    """
    Fixes comma issues due to OCR errors
    :param speech: text of the speech
    :return: corrected text
    """
    return re.sub("\.(?=\s[a-z0-9]|\sI[\W\s])", ",", speech)


if __name__ == "__main__":
    filepath = "../hein-bound/{}".format(get_speeches_filename(111))
    speech_df = read_full_speech(filepath)

    filepath = "../hein-bound/{}".format(get_speakermap_filename(111))
    speaker_df = read_speakermap(filepath)

    final_df = merge_speech_speaker(speech_df, speaker_df)
    # print(final_df.head())

    small_df = final_df[:10000]
    lemmatized_speeches = small_df.copy().apply(lambda x: lemmatize(x.loc["speech"]), axis=1)
    small_df.insert(1, "lemmatized_speech", lemmatized_speeches)

    small_df.to_pickle("speech.pkl")

    unpickled_df = pd.read_pickle("speech.pkl")
    print(unpickled_df)
