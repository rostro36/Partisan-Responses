import pandas as pd 
import os 
from tqdm import tqdm 
import re
from bs4 import BeautifulSoup
import requests
import pickle
import csv 
import nltk 
from nltk.stem import PorterStemmer
from utils import *

class Heinbound:
    def __init__(self, path):
        self.path = path
        self.stemmer = PorterStemmer()

    def get_topic_phrase(self):
        df = pd.read_table(os.path.join(path, "topic_phrases.txt"), sep="|")
        self.topic_phrase = set(df['phrase'])
        return self.topic_phrase
    
    def read_full_speech(self, filepath):
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

    def read_speakermap(self, filepath):
        """
        Load into a dataframe the speaker map from a text file in hein-bound
        :param filepath: filepath
        :return: speakermap dataframe
        """
        with open(filepath, errors="ignore") as f:
            speakermap_df = pd.read_table(f, delimiter = "|")
        return speakermap_df

    def merge_speech_speaker(self, speech_df, speaker_df):
        """
        Merge a dataframe containing speeches with one containing the speakermap
        :param speech_df: speeches dataframe
        :param speaker_df: speakermap dataframe
        :return: merged dataframe
        """
        speech_df = speech_df.astype({"speech_id": type(speaker_df.loc[:,'speech_id'][0])})
        return speaker_df.merge(speech_df, on="speech_id", how="left")

    def get_speeches_filename(self, idx):
        """
        Returns the name of the speeches file given an index
        :param idx: index
        :return: filename
        """
        return "speeches_{}.txt".format(idx)

    def get_speakermap_filename(self, idx):
        """
        Returns the name of the speakermap file given an index
        :param idx: index
        :return: filename
        """
        return "{}_SpeakerMap.txt".format(idx)

    def lemmatize(self, phrase):
        """
        Given some text, it returns the lemmatized text
        :param phrase: text
        :return: lemmatized text
        """
        return " ".join([word.lemma_ for word in sp(phrase)])

    def change_comma(self, speech):
        """
        Fixes comma issues due to OCR errors
        :param speech: text of the speech
        :return: corrected text
        """
        return re.sub("\.(?=\s[a-z0-9]|\sI[\W\s])", ",", speech)
            
    def filter_speech(self, speech_speaker_df, row_id):
        speech = self.change_comma(str(speech_speaker_df.iloc[row_id].speech))
        sents = nltk.sent_tokenize(speech)
        party = speech_speaker_df.iloc[row_id].party
        for s in sents:
            tokens = [self.stemmer(t.lower()) for t in nltk.tokenize.word_tokenize(s)]
            stemmed_sent = " ".join(tokens)
            if any(phrase in stemmed_sent for phrase in self.topic_phrase):
                return speech, party 
        return None

    def filter_edition(self, edition):
        """
        edition: int
        """
        speeches = []
        parties = []
        # Read speeches
        edition = str(edition).zfill(3)
        speech_df = self.read_full_speech(self.get_speeches_filename(edition))
        speaker_df = self.read_speakermap(self.get_speakermap_filename(edition))
        final_df = self.merge_speech_speaker(speech_df, speaker_df)
        # Iterate over all speeches in this edition
        for j in tqdm(range(len(final_df))):
            result = self.filter_speech(final_df, j)
            if result:
                speeches.append(result[0])
                parties.append(result[1])
        return speeches, parties

    def filter_multiple_editions(self, edition_list): #106-112
        speeches = []
        parties = []
        for e in edition_list:
            result = self.filter_edition(e)
            speeches += result[0]
            parties += result[1]
        filtered = pd.DataFrame(list(zip(speeches, parties)), columns =['speech', 'party'])
        return filtered

    def get_filtered_editions(self, filename):
        if os.path.exists(filepath):
            result = pd.read_pickle(filepath)
        else:
            result = self.filter_multiple_editions()
        return result
    # TODO
    def remove_redundancy(self, speech_df):
        # Remove short sentences, e.g Mr.Speaker
        repeat_short_sent = df['Speech'].apply(lambda s: s.split(",")).explode().value_counts()
        repeat_short_sent = repeat_short_sent[repeat_short_sent>=10]
        repeat_short_sent_df = pd.DataFrame(data = {"sent":repeat_short_sent.index, "counts":repeat_short_sent})
        repeat_short_sent_df = repeat_short_sent_df[repeat_short_sent_df['sent'].apply(lambda x: len(x.split())>1)]
        thres = np.quantile(repeat_short_sent_df['counts'], 0.99)
        to_remove = repeat_short_sent_df['sent'][repeat_short_sent_df['counts']>=thres].tolist()
        to_remove = [i+"," for i in to_remove]
        # Remove Long sentence
        df['Speech']  = df['Speech'].apply(lambda s: remove_wordy(s,to_remove))
        repeat_long_sent = df['Speech'].apply(lambda s: [" ".join(i.split()) for i in s.split(".") if len(i.split())>1]).explode().value_counts()
        repeat_long_sent = repeat_long_sent[repeat_long_sent>=10]
        thres = np.quantile(repeat_long_sent, 0.99)
        to_remove_long = repeat_long_sent[repeat_long_sent>=thres].index.tolist()
        to_remove_long = [i+"." for i in to_remove_long]

        df['Speech']  = df['Speech'].apply(lambda s: remove_wordy(s,to_remove_long))
        df['Speech'] = df['Speech'].apply(lambda s: " ".join(s.split()))
        # Remove 1-sent or 2-sent Speeches
        speechlen = df["Speech"].apply(lambda x: len(nltk.sent_tokenize(x)))
        final_df = df[speechlen > 2]
        pickle.dump(final_df, "all_speech_sentence_filtered.pkl")

class PresidencyProject:
    def __init__(self):
        self.presidency_proj_direc = "data/presidency_project"
        if not os.path.exists(self.presidency_proj_direc):
            os.makedirs(self.presidency_proj_direc)
        self.mainpage = "https://www.presidency.ucsb.edu/"
        self.partyAffiliation = {}

    def get_party(self, name, web_suffix):
        """
        web_suffix: str, suffix to direct to person's page
        """
        if name in self.partyAffiliation.keys():
            return self.partyAffiliation[name]
        url = self.mainpage+web_suffix
        if "president" in url:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            party = soup.find_all("div", {"class":"f-item"})[3].text
            self.partyAffiliation[name] = party
            return party
        else:
            self.partyAffiliation[name] = None 
            #TODO: external source?
            return None

class NewsConference(PresidencyProject):
    """
    News Conference
    """
    def __init__(self):
        super()
    def get_newsconf_weblink_list(self):
        """
        return:
        conferences: list, [(url, date)...]
        """
        indexpage_suffix = "documents/app-categories/presidential/news-conferences?items_per_page=60&page={}" #36pages
        indexpage = self.mainpage + indexpage_suffix
        conferences = []
        for i in range(36):
            soup = BeautifulSoup(requests.get(indexpage.format(i)).text, 'html.parser')
            rows = soup.find("div", {"class": "view-content"}).find_all("div", {"class":"col-sm-8"})
            conferences += [(prefix+i.find("a")["href"], i.span['content']) for i in rows]
        assert len(conferences) == 2157
        '''
        with open() as f:
            f.writelines(conferences)
        '''
        return conferences 
    def get_newsconf_QA(self, conferences, save_file):
        """
        conferences: list of (str) news conference links
        """ 
        for id, (url, date) in enumerate(conferences):
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")
            last, cur = None, None
            QA = []

            briefing = soup.find("div", {"class": "field-docs-content"})
            answerer = soup.find("h3", {"class": "diet-title"})
            answerer_name = answerer.a.text
            party = self.get_party(answerer_name, answerer.a['href'])
            date = None
            for k, speech in enumerate(briefing.contents):
                try:
                    if speech.i.text.startswith("Q."): #TODO: no i element
                        cur = k
                        if last is not None:
                            question = briefing.contents[last].contents[1]
                            answer = " ".join([a.text for a in briefing.contents[last+1:cur]])
                            QA.append([id, question, answer, answerer_name, party, date])
                            last = cur 
                        else:
                            last = k
                            continue
                except AttributeError:
                    continue
        if save_file:
            csvwriter = csv.writer(save_file, delimiter=",")
            csvwriter.writerow(["id", "question", "answer", "answerer_name", "party", "date"])
            csvwriter.writerow(QA)
            save_file.close()
        return QA
    def collect_and_save(self, save_file=None): #TODO: MaxRequest
        """
        save_file: str, path to save news conference QA table, if None, only collect
        """
        conferences = self.get_newsconf_weblink_list()
        f = open(os.path.join(presidency_proj_direc,"president_newsconference.csv"), 'w', newline='')
        self.get_newsconf_QA(conferences, save_file)

class Debate(PresidencyProject):
    """
    Presidential Campaign Debate
    """
    def __init__(self):
        self.url = "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0"
    def collect_debate_guidebook(url, save_direc):
        statistics = {"presidential":0, "democrat":0, "republican":0}
        debate_id, year, category = 0, None, None
        # Init csvfile
        f = open(os.path.join(save_direc,"presidency_project_debates_guidebook.csv"), 'w', newline='')
        debates_csvwriter = csv.writer(f, delimiter=",")
        debates_csvwriter.writerow(['debate_id', 'year', 'category', 'debate_name', 'url'])
        # Find debate guidebook table
        debates_table = find_debate_table(url)
        for row in debates_table.find_all("tr")[:-1]:
            category_row = row.find_all("strong")
            if category_row == []: 
                name, url = find_debate_name_and_url(row)
                if name is None:
                    continue
                debates_csvwriter.writerow([debate_id, year, category, name, url])
                statistics[category] += 1
                debate_id += 1
            else:
                year, category = find_year_and_category(category_row) 
        f.close()
        print(statistics)
        print(debate_id)

    def find_debate_table(url):
        guidebook = requests.get(url)
        guidebook_soup = BeautifulSoup(guidebook.text, "html.parser")
        debates_table = guidebook_soup.find("tbody")
        return debates_table

    def find_debate_name_and_url(row):
        cells = row.find_all("td")
        if len(cells) >1:
            name = cells[1].text
        else:
            return None, None
        try:
            url = cells[1].a['href']
        except TypeError:
            print(cells[1].text)
            return None, None
        return name, url

    def find_year_and_category(row):
        year = row[0].text
        cat_text = row[1].text.lower()
        if "general" in cat_text:
            category = "presidential"
        elif "democratic" in cat_text:
            category = "democrat"
        elif "republican" in cat_text:
            category = "republican"
        return year, category


#collect_debate_guidebook(url, presidency_proj_direc)
guidebook = pd.read_csv(os.path.join(presidency_proj_direc,"presidency_project_debates_guidebook.csv"))
dem = guidebook[guidebook['category'] == 'republican']
# Extract debate script
# Democrat

for i in range(dem.shape[0]):
    url = dem.iloc[i]['url']
    debate_id = dem.iloc[i]['debate_id']
    debate_html = requests.get(url)
    soup = BeautifulSoup(debate_html.text, "html.parser")
    script = soup.findAll('div', {'class': 'field-docs-content'})

    if len(script) == 1:
        speeches = script[0].find_all('p')
        current_speaker = None
        current_party = None #None if not candidate
        date = soup.find("div", {"class":"field-docs-start-date-time"}).span['content']
        p_id = 0
        try:
            start = speeches[p_id].contents[0].text
        except AttributeError:
            start = speeches[p_id].contents[0]
        if start.upper() == "PARTICIPANTS:": #skip to next <p>
            p_id += 1
        start = speeches[p_id]
        try:
            if start.contents[0].text.lower().startswith("moderator"):
                p = re.compile("([Mm][Oo][Dd][Ee][Rr][Aa][Tt][Oo][Rr][Ss]?:)|(\([\.\w\s-]+\))") # remove string moderator(s): and organization
                moderators = [i.strip() for i in p.sub('', speeches[1].text.replace("and", "")).split(";")]
                transtable = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
                moderators_lastname = [name.split(",")[0].split(" ")[-1].translate(transtable).lower() for name in moderators]
                p_id += 1
            speech_id = 0
            for speech in speeches[p_id:]:
                if len(speech.contents) == 2:
                    current_speaker = speech.contents[0].text.replace(":", "").lower()
                    if current_speaker not in moderators_lastname and current_speaker != "q":
                        current_party = "R" #todo
                    else:
                        current_party = "M"
                    speech = speech.contents[1].strip()
                else:
                    speech = speech.text
                record = [debate_id, speech_id, current_speaker, speech, current_party]
                csvwriter.writerow(record)
                speech_id += 1
        except AttributeError:
            continue
    else:
        print("More than 1 script found on the url:")
        print(url)
        print("="*20)
# write to file
f = open(os.path.join(presidency_proj_direc,"republican_candidates_debates_speeches.csv"), 'w', newline='')
csvwriter = csv.writer(f, delimiter=",")
csvwriter.writerow(['debate_id', 'speech_id', 'speaker', 'speech', 'party'])
f.close()
#unique_start = {'BRIT HUME, FOX NEWS:', 'PARTICIPANTS:', 'WOLF BLITZER:', 'Participants:', 'Moderators:', 'TOM BROKAW:'}
#unique_start = {'PARTICIPANTS:', 'ANNOUNCER:', 'Moderators:', 'COKIE ROBERTS:', 'Participants:'}
