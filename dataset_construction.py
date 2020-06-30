import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from utils import change_comma, get_speeches_filename, read_speakermap, merge_speech_speaker, get_speakermap_filename


def read_topic_words(file_name):
    topic_phrases = pd.read_table(file_name, sep="|")

    return set(topic_phrases['phrase'])

def make_dataset(topic_words):
    questions = []
    answers = []
    party_q = []
    party_a = []

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
            party_q.append(speeches_with_questions.iloc[j].party)
            party_a.append(speeches_with_questions.iloc[j].next_party)
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
                    no += 1

        print("finished file {}. {} pairs found".format(file, no))

    df = pd.DataFrame(list(zip(questions, answers, party_q, party_a)),
                      columns=['question', 'answer', 'party_q', 'party_q'])
    df.to_pickle("dataset.pkl")


def filter_answers(df):
    # filter out answers with a single sentence and very long answers
    new_df = df[df.apply(lambda x: len(nltk.sent_tokenize(x['answer'])) > 1 and len(nltk.sent_tokenize(x['answer'])) < 50, axis=1)]

    return new_df


if __name__ == "__main__":
    topic_words = read_topic_words("../phrase_clusters/topic_phrases.txt")
    make_dataset(topic_words)

    df = pd.read_pickle("dataset.pkl")

    df = filter_answers(df)
    df.to_pickle("filtered_dataset.pkl")
    print(df.head())
