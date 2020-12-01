from bs4 import BeautifulSoup
import requests
import pickle
import csv 
import os 
import pandas as pd 

presidency_proj_direc = "data/presidency_project"
if not os.path.exists(presidency_proj_direc):
    os.makedirs(presidency_proj_direc)
"""
News Conference
"""
def get_newsconf_index():
    prefix = "https://www.presidency.ucsb.edu"
    indexpage = "https://www.presidency.ucsb.edu/documents/app-categories/presidential/news-conferences?items_per_page=60&page={}" #36pages
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

def get_party(person):
    prefix = "https://www.presidency.ucsb.edu"
    url = prefix+person
    if "president" in url:
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        party = soup.find_all("div", {"class":"f-item"})[3].text
        return party
    else:
        #TODO: external source?
        return None

def get_newsconf_QA(conferences, save_file):
    """
    conferences: list of (str) news conference links
    """
    csvwriter = csv.writer(save_file, delimiter=",")
    csvwriter.writerow(["id", "question", "answer", "answerer_name", "party", "date"])
    for id, (url, date) in enumerate(conferences):
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        briefing = soup.find("div", {"class": "field-docs-content"})
        last, cur = None, None
        QA = []
        answerer = soup.find("h3", {"class": "diet-title"})
        answerer_name = answerer.a.text
        party = get_party(answerer.a['href'])
        date = None
        for k, speech in enumerate(briefing.contents):
            try:
                if speech.i.text.startswith("Q."):
                    cur = k
                    if last is not None:
                        question = briefing.contents[last].contents[1]
                        answer = " ".join([a.text for a in briefing.contents[last+1:cur]])
                        QA.append([id, question, answer, answerer_name, party, date])
                        csvwriter.writerow([id, question, answer, answerer_name, party, date])
                        last = cur 
                    else:
                        last = k
                        continue
            except AttributeError:
                continue
    save_file.close()
    return QA
conferences = get_newsconf_index()
f = open(os.path.join(presidency_proj_direc,"president_newsconference.csv"), 'w', newline='')
get_newsconf_QA(conferences, f)
"""
Presidential Campaign Debate
"""
url = "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0"
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
import re
# Init csvfile
f = open(os.path.join(presidency_proj_direc,"republican_candidates_debates_speeches.csv"), 'w', newline='')
csvwriter = csv.writer(f, delimiter=",")
csvwriter.writerow(['debate_id', 'speech_id', 'speaker', 'speech', 'party'])
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
f.close()
#unique_start = {'BRIT HUME, FOX NEWS:', 'PARTICIPANTS:', 'WOLF BLITZER:', 'Participants:', 'Moderators:', 'TOM BROKAW:'}
#unique_start = {'PARTICIPANTS:', 'ANNOUNCER:', 'Moderators:', 'COKIE ROBERTS:', 'Participants:'}
