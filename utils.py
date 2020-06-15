import pandas as pd 

# Read Relevant Files
def read_full_speech(filepath):
    with open(filepath, errors="ignore") as f:
        speech = f.readlines()
        speech = [s.replace("\n", "").replace(" | ", " ").split("|") for s in speech]#Full speech
        speech_df = pd.DataFrame(speech[1:], columns=speech[0])
    return speech_df

def read_speakermap(filepath):
    with open(filepath, errors="ignore") as f:
        speakermap_df = pd.read_table(f, delimiter = "|")
    return speakermap_df

def merge_speech_speaker(speech_df, speaker_df):
    speech_df = speech_df.astype({"speech_id": type(speaker_df.loc[:,'speech_id'][0])})
    return speaker_df.merge(speech_df, on="speech_id", how="left")