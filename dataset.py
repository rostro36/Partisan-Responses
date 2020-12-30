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
#from utils import *
from time import sleep
import spacy 
import neuralcoref
import random 
import json 
from sklearn.model_selection import train_test_split
from Answer import Answer
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
        self.presidency_proj_direc = "data/presidency_project/"
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
    
    def collect_and_save(self, save_file=None):
        pass

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            self.collect_and_save(filepath)
        data = pd.read_csv(filepath)
        return data 
    
    def filter(self, filepath):
        df = pd.read_csv(filepath)
        df = df[-pd.isnull(df.answer)]
        df = df[df.question.apply(lambda x: "?" in x)]
        print("Remove entries with either invalid questions or null answers\n")
        print(df.party.value_counts())
        print()
        df.to_csv(filename, index_label=False)
        return df 

    def split_data(self, df, party):
        df = df[df.party==party]
        train, test = train_test_split(df, test_size=0.2, random_state=0)
        train, val = train_test_split(train, test_size=0.125, random_state=0)
        direc = "data/presidency_project/newsconference"
        prefix = party.lower()[:3]
        print("{}:\nTrain Size: {}\nValidation Size: {}\nTest Size: {}\n".format(party, train.shape[0], val.shape[0], test.shape[0]))
        
        train.to_csv(os.path.join(direc, "{}_train.csv".format(prefix)), index_label=False)
        val.to_csv(os.path.join(direc, "{}_val.csv".format(prefix)), index_label=False)
        test.to_csv(os.path.join(direc, "{}_test.csv".format(prefix)), index_label=False)
        return train, val, test

    def split_to_annotate(self, filepath, ann_size):
        random.seed(0)
        data = self.load_data(filepath)
        ann_idx = random.sample(range(data.shape[0]), ann_size)
        for i in ann_idx:
            with open("./brat/data/newsconf_coref/{}_{}.txt".format(data.id.iloc[i], i), 'w') as f:
                for line in nltk.sent_tokenize(str(data.answer.iloc[i])): # TODO
                    f.write(line.strip())
                    f.write("\n")
                with open("./brat/data/newsconf_coref/{}_{}.ann".format(data.id.iloc[i], i), 'w') as f:
                    f.write("")
    '''
    def write_json(self, df, save_path):
        result = []
        for i in range(df.shape[0]):
            sample = df.iloc[i]
            unprocessed_sample = {"question": sample.question, 
                        "entities":[], 
                        "types": None, 
                        "relations": [], 
                        "answer":None, 
                        "answer_og": sample.answer}
            result.append(sample)
        json.dump(result, save_path)
    '''

    def write_gw_input(self, input_data, output_direc, output_file):
        if type(input_data) is str:
            df = pd.read_csv(input_data)
        else:
            df = input_data

        verb_dict_path = os.path.join(output_direc, "verb_dict.pickle")
        verb_list_path = os.path.join(output_direc, "verb_list.pickle")
        if os.path.exists(verb_dict_path):
            verb_dict = pickle.load(open(verb_dict_path, 'rb'))
        else:
            verb_dict = {}
        if os.path.exists(verb_list_path):
            verb_list = pickle.load(open(verb_list_path, 'rb'))
        else:
            verb_dict = {}
            verb_list = []
            
        with open(os.path.join(output_direc, output_file), "w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            print("Output to: {}".format(os.path.join(output_direc, output_file)))
            for i in range(df.shape[0]):
                entry = df.iloc[i]
                preprocessed_row, verb_dict, verb_list = Answer(entry.answer).create_training(verb_dict,verb_list)
                phrase_corpus, phrase_type, triplet_id, parsed_text, parsed = preprocessed_row
                phrase_corpus = ' ; '.join(phrase_corpus)
                triplet_id = ' ; '.join([re.sub('\,','',str(x))[1:-1] for x in triplet_id])
                parsed = ' '.join([str(x) for x in parsed])
                tsvwriter.writerow([entry.question, phrase_corpus, phrase_type, triplet_id, parsed_text, parsed])
        pickle.dump(verb_dict, open(verb_dict_path, 'wb'))
        pickle.dump(verb_list, open(verb_list_path, 'wb'))
        return verb_list

class NewsConference(PresidencyProject):
    """
    News Conference
    """
    def __init__(self):
        super().__init__()
        self.direc = self.presidency_proj_direc+"newsconference/"
        if not os.path.exists(self.direc):
            os.makedirs(self.direc)
        self._path_to_conference_list = os.path.join(self.direc, "newsconference_list.txt")
        self._ckpt = os.path.join(self.direc, "ckpt.txt")
        self.pipe = spacy.load('en')
        neuralcoref.add_to_pipe(self.pipe)

    def _load_conference_list(self):
        with open(self._path_to_conference_list, "r") as f:
            return [i.split() for i in f.readlines()]

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
            conferences += [(self.mainpage+i.find("a")["href"], i.span['content']) for i in rows]
        assert len(conferences) == 2157
        with open(self._path_to_conference_list, "w") as f:
            f.writelines([i[0]+"\t"+i[1]+"\n" for i in conferences])
        return conferences 
    
    def load_ckpt(self, conferences):
        if os.path.exists(self._ckpt):
            ckpt = int(open(self._ckpt, "r").read())
            if ckpt+1 == len(conferences):
                return 
            conferences = conferences[ckpt:]
            print("Last conference id scraped: {}".format(ckpt)) 
        else:
            ckpt = 0
        return ckpt, conferences

    def is_q_start(self, speech, answer, row):
        is_q = False
        if hasattr(speech, "i"): 
            if hasattr(speech.i, "text") and speech.i.text.startswith("Q."):
                is_q = True 
        if hasattr(speech, "em"):
            if hasattr(speech.em, "text") and speech.em.text.startswith("Q."):
                is_q = True
        try:
            if speech.text.startswith("Q. "):
                is_q = True
        except AttributeError:
            if speech.startswith("Q. "):
                is_q = True
        
        return is_q
    
    def is_a_start(self, speech, answerer_name, question, row):
        is_a = False
        lastname = answerer_name.lower().split()[-1]
        if question != '':
            if hasattr(speech, "i"):
                if hasattr(speech.i, "text"):
                    if speech.i.text.lower().startswith("the president.") or speech.i.text.lower().startswith('president '+lastname):
                        is_a = True 
            if hasattr(speech, "em"):
                if hasattr(speech.em, "text"):
                    if speech.em.text.lower().startswith('the president.') or speech.em.text.lower().startswith('president '+lastname):
                        is_a = True 
            if hasattr(speech, "text"):
                prefix = speech.text.split(".")[0][:30].lower()
            else:
                prefix = speech.split(".")[0][:30].lower()
            if prefix.startswith('the president') or prefix.startswith('president '+lastname):
                is_a = True
            if is_a:
                row=[question]
                question = ""
        return is_a, question, row
            
    def is_a_following(self, speech, answer):
        is_a_following=True
        if answer == "":
            is_a_following = False
        if hasattr(speech, "i") and speech.i is not None:
            is_a_following = False
        if hasattr(speech, "em") and speech.em is not None:
            is_a_following=False
        if hasattr(speech, "text"):
            t=speech.text.lower()
        else:
            t = speech.lower()
        if t.startswith('president ') or t.startswith('prime minister ') or t.startswith('chancellor '):
            is_a_following=False
        return is_a_following
    
    def get_metadata(self, url):
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        briefing = soup.find("div", {"class": "field-docs-content"})
        answerer = soup.find("h3", {"class": "diet-title"})
        answerer_name = answerer.a.text
        party = self.get_party(answerer_name, answerer.a['href'])
        return briefing, answerer_name, party

    def scrape_newsconf_QA(self, conferences, save_file):
        """
        conferences: list of (str) news conference links
        """ 
        print("Start Scraping...")
        ckpt, conferences = self.load_ckpt(conferences)
        
        for id, (url, date) in enumerate(conferences):
            conf_id = ckpt + id +1
            QA = []
            try:
                briefing, answerer_name, party = self.get_metadata(url)
                question = ""
                answer = ""
                row = []
                q_start = -1
                for k, speech in enumerate(briefing.contents):
                    is_q =self.is_q_start(speech, answer, row)
                    is_a, question, row =self.is_a_start(speech, answerer_name, question, row)
                    _is_a_following = self.is_a_following(speech, answer)
                    if is_q:
                        if row != []:
                            resolved = self.pipe(row[0]+" ## "+answer)._.coref_resolved.split(" ## ")
                            q = resolved[0]
                            a = "".join(resolved[1:])
                            row = [conf_id, q, a, answerer_name, party]
                            QA.append(row)
                            row = []
                        answer = ""
                        try:
                            question += speech.text[3:]
                        except AttributeError:
                            question += speech[3:]
                        q_start = k
                    elif question != "":
                        try:
                            question += speech.text
                        except AttributeError:
                            question += speech
                    elif is_a or _is_a_following:
                        if hasattr(speech, "text"):
                            answer += speech.text
                        else:
                            answer += speech 
                    else:
                        if row != [] and answer != "":
                            resolved = self.pipe(row[0]+" ## "+answer)._.coref_resolved.split(" ## ")
                            q = resolved[0]
                            a = "".join(resolved[1:])
                            row = [conf_id, q, a, answerer_name, party]
                            QA.append(row)
                            row = []
                        continue
                if save_file:
                    if conf_id > 1:
                        with  open(save_file, "a") as f:
                            csvwriter = csv.writer(f, delimiter=",")
                            csvwriter.writerows(QA)
                    else:
                        with open(save_file, "w") as f:
                            csvwriter = csv.writer(f, delimiter=",")
                            csvwriter.writerow(["id", "question", "answer", "answerer_name", "party", "date"])
                            csvwriter.writerows(QA)
                print("Conf id {} Finished".format(conf_id))
                open(self._ckpt, 'w').write(str(conf_id)) 
            except requests.exceptions.SSLError:
                print("Wait for 5 minutes to continute...")
                sleep(300)
                self.scrape_newsconf_QA(conferences, save_file)

    def collect_and_save(self, save_file=None): #TODO: MaxRequest
        """
        save_file: str, path to save news conference QA table, if None, only collect
        """
        if os.path.exists(save_file):
            print("Already saved!")
            return 
        conferences = self._load_conference_list() if os.path.exists(self._path_to_conference_list) else self.get_newsconf_weblink_list()
        self.scrape_newsconf_QA(conferences, save_file)
        self.filter(save_file)
    
class Debate(PresidencyProject):
    """
    Presidential Campaign Debate
    """
    def __init__(self):
        self.url = "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0"
    def collect_debate_guidebook(self, url, save_direc):# TODO: check local directory
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

    def find_debate_table(self, url):
        guidebook = requests.get(url)
        guidebook_soup = BeautifulSoup(guidebook.text, "html.parser")
        debates_table = guidebook_soup.find("tbody")
        return debates_table

    def find_debate_name_and_url(self, row):
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

    def find_year_and_category(self, row):
        year = row[0].text
        cat_text = row[1].text.lower()
        if "general" in cat_text:
            category = "presidential"
        elif "democratic" in cat_text:
            category = "democrat"
        elif "republican" in cat_text:
            category = "republican"
        return year, category

# #collect_debate_guidebook(url, presidency_proj_direc)
# guidebook = pd.read_csv(os.path.join(presidency_proj_direc,"presidency_project_debates_guidebook.csv"))
# dem = guidebook[guidebook['category'] == 'republican']
# Extract debate script
# Democrat
# for i in range(dem.shape[0]):
#     url = dem.iloc[i]['url']
#     debate_id = dem.iloc[i]['debate_id']
#     debate_html = requests.get(url)
#     soup = BeautifulSoup(debate_html.text, "html.parser")
#     script = soup.findAll('div', {'class': 'field-docs-content'})

#     if len(script) == 1:
#         speeches = script[0].find_all('p')
#         current_speaker = None
#         current_party = None #None if not candidate
#         date = soup.find("div", {"class":"field-docs-start-date-time"}).span['content']
#         p_id = 0
#         try:
#             start = speeches[p_id].contents[0].text
#         except AttributeError:
#             start = speeches[p_id].contents[0]
#         if start.upper() == "PARTICIPANTS:": #skip to next <p>
#             p_id += 1
#         start = speeches[p_id]
#         try:
#             if start.contents[0].text.lower().startswith("moderator"):
#                 p = re.compile("([Mm][Oo][Dd][Ee][Rr][Aa][Tt][Oo][Rr][Ss]?:)|(\([\.\w\s-]+\))") # remove string moderator(s): and organization
#                 moderators = [i.strip() for i in p.sub('', speeches[1].text.replace("and", "")).split(";")]
#                 transtable = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
#                 moderators_lastname = [name.split(",")[0].split(" ")[-1].translate(transtable).lower() for name in moderators]
#                 p_id += 1
#             speech_id = 0
#             for speech in speeches[p_id:]:
#                 if len(speech.contents) == 2:
#                     current_speaker = speech.contents[0].text.replace(":", "").lower()
#                     if current_speaker not in moderators_lastname and current_speaker != "q":
#                         current_party = "R" #todo
#                     else:
#                         current_party = "M"
#                     speech = speech.contents[1].strip()
#                 else:
#                     speech = speech.text
#                 record = [debate_id, speech_id, current_speaker, speech, current_party]
#                 csvwriter.writerow(record)
#                 speech_id += 1
#         except AttributeError:
#             continue
#     else:
#         print("More than 1 script found on the url:")
#         print(url)
#         print("="*20)
# write to file
# f = open(os.path.join(presidency_proj_direc,"republican_candidates_debates_speeches.csv"), 'w', newline='')
# csvwriter = csv.writer(f, delimiter=",")
# csvwriter.writerow(['debate_id', 'speech_id', 'speaker', 'speech', 'party'])
# f.close()
#unique_start = {'BRIT HUME, FOX NEWS:', 'PARTICIPANTS:', 'WOLF BLITZER:', 'Participants:', 'Moderators:', 'TOM BROKAW:'}
#unique_start = {'PARTICIPANTS:', 'ANNOUNCER:', 'Moderators:', 'COKIE ROBERTS:', 'Participants:'}

if __name__ == "__main__":
    conf = NewsConference()
    filename = "./data/presidency_project/newsconference/newsconference_2157_201229.csv"
    df = conf.load_data(filename)
    for party in set(df.party):
        train, val, test = conf.split_data(df, party)
        # GW naive
        gwnaive_direc = "./data/presidency_project/newsconference/gwnaive/{}".format(party)
        if not os.path.exists(gwnaive_direc):
            os.makedirs(gwnaive_direc)
        # write processed data
        print("Process {} data to GW input...".format(party))
        conf.write_gw_input(train, output_direc=gwnaive_direc, output_file="preprocessed.train.tsv")
        conf.write_gw_input(val, output_direc=gwnaive_direc, output_file="preprocessed.val.tsv")
        verb_list = conf.write_gw_input(test, output_direc=gwnaive_direc, output_file="preprocessed.test.tsv")
        # write relation vocab
        print("Write vocabulary\n")
        with open(os.path.join(gwnaive_direc,'relations.vocab'), 'w') as vocabfile:
            vocabfile.writelines("%s\n" % verb.upper() for verb in verb_list)