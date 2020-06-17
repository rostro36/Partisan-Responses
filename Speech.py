import re
import nltk
class Speech:
    def __init__(self, speech):
        #self.speaker = speech['lastname'] + " " + speech['firstname']
        #self.party = speech['party']
        self.content = speech #['speech'] 
    
    def change_comma(self):
        self.content = re.sub("\.(?=\s[a-z0-9]|\sI[\W\s])", ",", self.content)

    def _find_triplets(self, openinfo_result):
        # delete modal verbs
        arg0 = "ARG0: "
        arg1 = "ARG1: "
        speech_triplet = []
        for sentence in openinfo_result:
            sent_triplet = []
            if sentence is not []:
                for d in sentence:
                    subjidx = d['description'].rfind(arg0) 
                    predidx = d['description'].rfind(arg1)
                    if subjidx != -1 and predidx != -1:
                        subj = re.search("(?<=ARG0: )[\w\s\'\",\.\:]*(?=])", d['description']).group(0)
                        verb = d['verb']
                        predicate = re.search("(?<=ARG1: )[\w\s\'\",\.\:]*(?=])", d['description']).group(0)
                        sent_triplet.append((subj, verb, predicate))
            speech_triplet.append(sent_triplet)
        return speech_triplet

    def create_triplet(self, coref_extractor, oie_extractor):
        coref_content = coref_extractor.coref_resolved(self.content)
        sents = nltk.tokenize.sent_tokenize(coref_content)
        oie_result = [oie_extractor.predict(i)['verbs'] for i in sents]
        triplets = self._find_triplets(oie_result)
        #triplets.append(self.party)
        return triplets
        