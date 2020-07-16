import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from utils import change_comma, get_speeches_filename, read_speakermap, merge_speech_speaker, get_speakermap_filename, lemmatize


def read_topic_words(file_name):
    """
    Return a set of phrases from given file

    :param file_name: the name of the file
    :return: set of phrases
    """
    topic_phrases = pd.read_table(file_name, sep="|")

    return set(topic_phrases['phrase'])

def make_dataset(topic_words):
    """
    Crates a dataframe with columns (question, answer, party_q, party_a)
    from the hein-bound corpus by selecting the questions that are related
    to a predefined set of topics

    :param topic_words: set of phrases related to the given topics
    :return: pandas dataframe with columns (question, answer, party_q, party_a)
    """

    questions = []
    answers = []
    party_q = []
    party_a = []
    year = []

    ps = PorterStemmer()

    for i in range(43, 112):
        file = i
        if i < 100:
            file = '0{}'.format(i)

        no = 0

        # read speeches
        with open("../hein-bound/{}".format(get_speeches_filename(file)), errors="ignore") as f:
            speech = f.readlines()
            speech = [s.strip() for s in speech]
            speech = [[s[:s.find('|')], s[s.find('|') + 1:]] for s in speech]
            speech_df = pd.DataFrame(speech[1:], columns=speech[0])

        # read speakermap
        speaker_df = read_speakermap("../hein-bound/{}".format(get_speakermap_filename(file)))

        # merge speeches
        final_df = merge_speech_speaker(speech_df, speaker_df)

        df_merged = pd.concat([final_df, final_df.shift(-1).add_prefix('next_')], axis=1)
        speeches_with_questions = df_merged.loc[df_merged['speech'].str.contains('\?')]

        for j in tqdm(range(len(speeches_with_questions))):
            # party_q.append(speeches_with_questions.iloc[j].party)
            # party_a.append(speeches_with_questions.iloc[j].next_party)
            # change comma for both question speech and answer speech
            question = change_comma(str(speeches_with_questions.iloc[j].speech))
            answer = change_comma(str(speeches_with_questions.iloc[j].next_speech))
            # tokenize question speech
            quest_sents = nltk.sent_tokenize(question)
            # select question sentences from speech
            quests = [q for q in quest_sents if q.find("?") + 1]
            for q in quests:
                words = word_tokenize(q)
                words = [ps.stem(w.lower()) for w in words]
                stemmed_q = ' '.join(words)
                if any(phrase in stemmed_q for phrase in topic_words) and len(quest_sents) == 1:
                    questions.append(q)
                    answers.append(answer)
                    party_q.append(speeches_with_questions.iloc[j].party)
                    party_a.append(speeches_with_questions.iloc[j].next_party)
                    year.append(file)
                    no += 1

        print("finished file {}. {} pairs found".format(file, no))

    df = pd.DataFrame(list(zip(questions, answers, party_q, party_a, year)),
                      columns=['question', 'answer', 'party_q', 'party_a', 'year'])

    df = filter_answers(df)
    df = filter_by_party(df)
    df = lemmatize_answers(df)

    df.to_pickle("dataset.pkl")

    return df


def filter_answers(df):
    """
    Given a pandas dataframe that contains a column called answer,
    returns a new dataframe by filtering out those with answers longer
    than 50 sentences or with a single sentence

    :param df: pandas dataframe
    :return: filtered pandas dataframe
    """

    # filter out answers with a single sentence and very long answers
    new_df = df[df.apply(lambda x: len(nltk.sent_tokenize(x['answer'])) > 1 and len(nltk.sent_tokenize(x['answer'])) < 50, axis=1)]

    return new_df

def filter_by_party(df):
    """
    Given a pandas dataframe that contains a column called party_a
    (the party of the speaker who answered the question),
    returns a new dataframe by filtering out the answers that are not
    given by republicans or democrats

    :param df: a pandas dataframe
    :return: a new pandas dataframe
    """

    # keep only republican and democrat answers
    new_df = df[df['party_a'].isin(['R', 'D'])]

    return new_df



def lemmatize_answers(df):
    """
    Given a pandas dataframe that contains a column called answer,
    returns a new dataframe with the lemmatized text from this column

    :param df: a pandas dataframe
    """

    # lemmas = [lemmatize(speech) for speech in tqdm(df['answer'])]
    tqdm.pandas()
    lemmas = df['answer'].progress_apply(lambda x: lemmatize(x))
    df['lemmatized_answer'] = lemmas

    return df


if __name__ == "__main__":

    # df = pd.read_pickle("data/final_data.pkl")
    # lemmas = lemmatize_answers(df)
    # lemmas.to_pickle("final_lemmas.pkl")

    # df = pd.read_pickle("final_lemmas.pkl")
    # print(df.iloc[0])
    topic_words = read_topic_words("../phrase_clusters/topic_phrases.txt")
    df = make_dataset(topic_words)
    print(df.head())
    print(df.iloc[0])




