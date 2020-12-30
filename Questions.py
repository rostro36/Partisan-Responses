from bs4 import BeautifulSoup
import requests
import pickle

class Questions:
    def __init__(self):
        self.questions = []
    def __len__(self):
        # number of questions
        return len(self.questions)
    def __getitem__(self, idx):
        return self.questions[idx]

class GallupQuestions(Questions):
    def __init__(self):
        super(GallupQuestions, self).__init__()
        self.gallup_topics = "https://news.gallup.com/poll/trends.aspx#P"
        
    def collect(self):    
        gallup_topics_html = requests.get(self.gallup_topics)
        soup = BeautifulSoup(gallup_topics_html.text, "html.parser")

        questions_by_topic = {}
        # Iterate over A-Z
        for letter in soup.find_all('h4'):
            for topic in letter.parent.ul.find_all('a'):
                topic_link = topic.get('href')
                topic_questions = self.collect_topic_questions(topic_link)
                if topic_questions[0] is not None:
                    questions_by_topic[topic_questions[0]] = topic_questions[1]
                    self.questions += topic_questions[1]
        return questions_by_topic

    def collect_topic_questions(self, topic_link):
        # Exclude Rating Topics e.g Presidential Rating
        if 'rating' not in str.lower(topic_link):
            topic_html = requests.get(topic_link)
            topicsoup = BeautifulSoup(topic_html.text, "html.parser")
            topic = topicsoup.h1.text
            print("Collecting: {} \n".format(topic))
            questionslist = []
            for figcaption in topicsoup.find_all('figcaption'):
                q = figcaption.div.string
                if q.startswith("Next, "):
                    q = self.rephrase_question_with_next(q)
                q = q.replace("-- ", "")
                q = q.replace("... ", "")
                if "[ROTATED:" in q:
                    q = self.correct_randrotate(q)
                if "[RANDOM ORDER]" in q:
                    questionslist += self.correct_randorder(figcaption, q)
                    continue
                questionslist.append(q)
            return topic, questionslist
        else:
            return None, []

    def rephrase_question_with_next(self, question):
        prefix1 = "Next, I'm going to read you a list of issues. Regardless of whether or not you think it should be legal, for each one, please tell me whether"
        prefix2 = "Next, we'd like to know how you feel about the state of the nation in each of the following areas."
        prefix3 = "Next, I'd like your overall opinion of some"
        prefix4 = "Next, do you favor or oppose each of the following proposals?"
        if question.startswith(prefix1):
            prefix = prefix1
            question = question.replace(prefix, "Do")
            tmp = question.split(". ")
            question, issue = tmp[0], tmp[1].replace("How about ", "")[:-1]
            question = question.replace("in general it is", "in general "+issue+" is")
        elif question.startswith(prefix2):
            prefix = prefix2
            question = question.split(". ")[-1]
            question = question.replace("How", "Are you satisfied")
        elif question.startswith(prefix3):
            splitresult = question.split(". ")
            target = splitresult[-1].replace("Next, how about ... ", "")[:-1]
            question = splitresult[1].replace("[RANDOM ORDER]", target)
            question = question.split("?")[0]
        elif question.startswith(prefix4):
            clause = question.replace(prefix4, "").lower()+"?"
            question = "Do you favor or oppose that" + clause
        return question

    def correct_randrotate(self, question):
        q = question.replace(" [ROTATED: ", ": ").replace("]", "")
        q = q.replace("(", "").replace(")", "")
        return q 
        
        
    def correct_randorder(self, captiontag, question):
        questions = []
        for i in captiontag.parent.find_all("th", {"scope":"rowgroup"}):
            target = i.text.lower()
            question = question.replace("[RANDOM ORDER]", target)
            if question.startswith("Next,"):
                question = self.rephrase_question_with_next(question)
        return questions
if __name__ == "__main__":
    gallup = GallupQuestions()
    gallup_questions_by_topic = gallup.collect()
    pickle.dump(gallup_questions_by_topic, open("./gallup_questions_by_topic.pickle", "wb"))
#'Do you think abortions should be legal under any circumstances, legal only under certain circumstances or illegal in all circumstances? (Asked of those who say abortion should be legal under certain circumstances)
# \\nDo you think abortion should be legal in most circumstances or only in a few circumstances?'