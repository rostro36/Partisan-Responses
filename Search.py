import numpy as np
import pandas as pd
import spacy
import pickle
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class Search:
    def __init__(self, speeches=None,
                 rep_speeches=None,
                 dem_speeches=None,
                 rep_vectorizer=None,
                 dem_vectorizer=None,
                 rep_tfidf= None,
                 dem_tfidf=None):
        if speeches is not None:
            self.speeches = speeches
            self.split_by_party()
        else:
            assert rep_speeches is not None, "must provide republican speeches"
            assert dem_speeches is not None, "must provide democrat speeches"
            self.rep = rep_speeches
            self.dem = dem_speeches

        self.sp = spacy.load('en_core_web_sm')
        if rep_vectorizer is not None:
            # load existing vectorizer and tfidf
            self.rep_vectorizer = rep_vectorizer
            assert rep_tfidf != None, "must provide tfidf for republicans"
            self.rep_tfidf = rep_tfidf
        else:
            # fit vectorizer
            self.rep_vectorizer = TfidfVectorizer(stop_words='english',
                                          min_df=5, max_df=.5, ngram_range=(1,2))
            self.rep_tfidf = self.rep_vectorizer.fit_transform(self.rep['lemmatized_answer'])

        if dem_vectorizer is not None:
            # load existing vectorizer and tfidf
            self.dem_vectorizer = dem_vectorizer
            assert dem_tfidf != None, "must provide tfidf for democrats"
            self.dem_tfidf = dem_tfidf
        else:
            # fit vectorizer
            self.dem_vectorizer = TfidfVectorizer(stop_words='english',
                                          min_df=5, max_df=.5, ngram_range=(1,2))
            self.dem_tfidf = self.dem_vectorizer.fit_transform(self.dem['lemmatized_answer'])


    def split_by_party(self):
        """
        Given a dataframe that contains both republican and democrat speeches,
        it splits it into 2 dataframes depending on party
        :return:
        """
        self.rep = self.speeches[self.speeches['party_a'] == 'R']
        self.dem = self.speeches[self.speeches['party_a'] == 'D']

        print("republican speeches: {}".format(len(self.rep)))
        print("democrat speeches: {}".format(len(self.dem)))


    def lemmatize(self, phrase):
        """
        Given some text, returns the lemmatized text

        :param phrase: text to lemmatize
        :return: lemmatized text
        """
        return " ".join([word.lemma_ for word in self.sp(phrase)])


    def search(self, question, party, topk=10):
        """
        Given a question and a party, returns the most relevant
        speeches in the dataset

        :param question: question to find answer to
        :param party: R(republican) or D(democrat)
        :param topk: how many speeches to return
        :return:
        """
        if party not in ['R', 'D']:
            raise Exception("The party can only be R or D")
        if party == 'R':
            # transform query
            query = self.rep_vectorizer.transform([self.lemmatize(question)])
            # sort based on cosine similarity
            scores = (self.rep_tfidf * query.T).toarray()
        else:
            # transform query
            query = self.dem_vectorizer.transform([self.lemmatize(question)])
            # sort based on cosine similarity
            scores = (self.dem_tfidf * query.T).toarray()

        results = (np.flip(np.argsort(scores, axis=0)))

        if party == 'R':
            return self.rep.iloc[results[:topk, 0]]
        else:
            return self.dem.iloc[results[:topk, 0]]

    def save_data(self):
        """
        Saves speeches, vectorizers and tfidf matrices for later use

        :return:
        """
        data = {}
        data['rep_speeches'] = self.rep
        data['dem_speeches'] = self.dem
        data['rep_vectorizer'] = self.rep_vectorizer
        data['dem_vectorizer'] = self.dem_vectorizer
        data['rep_tfidf'] = self.rep_tfidf
        data['dem_tfidf'] = self.dem_tfidf

        with open("data/tfidf_data.pkl", 'wb') as file:
            pickle.dump(data, file)


if __name__ == "__main__":
    # speeches = pd.read_pickle("data/final_lemmas.pkl")
    #
    # search = Search(speeches=speeches)
    # print("----- Saving data -----")
    # search.save_data()

    with open("data/tfidf_data.pkl", "rb") as file:
        data = pickle.load(file)

    # print(data['rep_speeches'])
    search = Search(rep_speeches=data['rep_speeches'],
                    dem_speeches=data['dem_speeches'],
                    rep_vectorizer=data['rep_vectorizer'],
                    dem_vectorizer=data['dem_vectorizer'],
                    rep_tfidf=data['rep_tfidf'],
                    dem_tfidf=data['dem_tfidf'])

    question = "What reforms were adopted by the 110th Congress?"
    party = 'D'
    results = search.search(question, party, topk=5)
    print("----- Search Done -----")
    for result in results['answer']:
        print(result)

    qa_pipeline = pipeline("question-answering")

    question_df = pd.DataFrame.from_records([{
        'question': question,
        'context': res
    } for res in results["answer"]])

    preds = qa_pipeline(question_df.to_dict('records'))
    answer_df = pd.DataFrame.from_records(preds).sort_values(by="score", ascending=False)
    print("----- Short Answers -----")
    print(answer_df)





