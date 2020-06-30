import pandas as pd
# import allennlp_models.structured_prediction
# import allennlp_models.coref
import nltk
import re
import utils

auxillary_verbs=['can','could','may','might','must','shall','should','will','would'] #https://englishstudyonline.org/auxiliary-verbs/
distance_threshold=0.5

class Speech:
    def __init__(self, speech):
        self.speaker = speech['lastname'] + " " + speech['firstname']
        self.party = speech['party']
        self.content = speech['speech']
        
    def change_comma(self):
        """
        Replace improper period to comma
        """
        self.content = re.sub("\.(?=\s[a-z0-9]|\sI[\W\s])", ",", self.content)

    def _find_triplets(self, openinfo_result):
        """
        Find one or more triplets of each sentence from allennlp OIE results
        Param:
        ========

        Return:
        ========
        speech_triplets: list, a list of lists of triplet tuples (of a speech)
        """
        arg0 = "ARG0: "
        arg1 = "ARG1: "
        modalverbs = ["can", "could", "may", "might", "must", "shall", "should", "will", "would"]
        speech_triplet = []
        for sentence in openinfo_result:
            sent_triplet = []
            if sentence is not []:
                for d in sentence: # Extract from 'description' result of OIE
                    verb = d['verb']
                    if verb not in modalverbs:
                        subjidx = d['description'].rfind(arg0) 
                        predidx = d['description'].rfind(arg1)
                        if subjidx != -1 and predidx != -1:
                            subj = re.search("(?<=ARG0: )[\w\s\'\",\.\:]*(?=])", d['description']).group(0)
                            predicate = re.search("(?<=ARG1: )[\w\s\'\",\.\:]*(?=])", d['description']).group(0)
                            sent_triplet.append((subj, verb, predicate))
            speech_triplet.append(sent_triplet)
        return speech_triplet
                
    def create_triplet(self):
        """
        Generate (subject, verb, object) triplets of a speech text
        Param:
        ========
        coref_extractor: allennlp coreferece resolution predictor
        oi_extractor: allennlp open information extractor

        Return:
        ========
        triplets: list, a list of triplet tuples except the last item being party string
        """
        oie_result=self.create_oieresult()
        triplets = self._find_triplets(oie_result)
        triplets.append(self.party)
        return triplets
    
    def create_oieresult(self):
        coref_content = utils.coref_extractor.coref_resolved(self.content)
        sents = nltk.tokenize.sent_tokenize(coref_content)
        sents = [{"sentence":s} for s in sents] #Format for oie batch predictor
        oie_result = utils.open_info_extractor.predict_batch_json(sents)
        oie_result = [i['verbs'] for i in oie_result]
        return oie_result
        
if __name__ == "__main__":
    df = pd.read_pickle("speech.pkl")
    print(df.loc[0])

    

    sample_speech = Speech(df.loc[0])
    sample_speech.change_comma()
    print(sample_speech.content)

    sample_triplets = sample_speech.create_triplet(coref_extractor, open_info_extractor)
    print(sample_triplets)
        
