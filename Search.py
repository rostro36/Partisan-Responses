import numpy as np
import pandas as pd
import spacy
import pickle
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class Search:
    def __init__(self, speeches):
        self.speeches = speeches
        self.sp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          min_df=5, max_df=.5, ngram_range=(1,2))
        self.tfidf = None
        self.fit_vectorizer()


    def lemmatize(self, phrase):
        return " ".join([word.lemma_ for word in self.sp(phrase)])

    def fit_vectorizer(self):
        self.tfidf = self.vectorizer.fit_transform(self.speeches['lemmatized_speech'])

    def search(self, question, topk=10):
        # transform query
        query = self.vectorizer.transform([self.lemmatize(question)])
        # sort based on cosine similarity
        scores = (self.tfidf * query.T).toarray()
        results = (np.flip(np.argsort(scores, axis=0)))

        return self.speeches.iloc[results[:topk, 0]]


if __name__ == "__main__":
    df = pd.read_pickle("speech.pkl")

    search = Search(df)
    question = "What reforms were adopted by the 110th Congress?"
    results = search.search(question)
    print("----- Search Done -----")
    print(results[['party', 'speech']])

    qa_pipeline = pipeline("question-answering")

    question_df = pd.DataFrame.from_records([{
        'question': question,
        'context': res
    } for res in results["speech"]])

    preds = qa_pipeline(question_df.to_dict('records'))
    answer_df = pd.DataFrame.from_records(preds).sort_values(by="score", ascending=False)
    print("----- Short Answers -----")
    print(answer_df)





