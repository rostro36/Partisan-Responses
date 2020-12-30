import pandas as pd
import nltk
import re
import utils
import hnswlib

auxillary_verbs=['can','could','may','might','must','shall','should','will','would'] #https://englishstudyonline.org/auxiliary-verbs/
distance_threshold=0.5

class Answer:
    def __init__(self, answer):
        self.content = answer
        self.phrase_corpus=[]
        self.phrase_type= "" #e.g 
        self.phrase_index=None
        self.triplet=[]
        self.triplet_id=[]
        self.parsed=[]
        
    def change_comma(self):
        """
        Replace improper period to comma
        """
        self.content = re.sub("\.(?=\s[a-z0-9]|\sI[\W\s])", ",", self.content)
        
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
    
    def create_coref(self):
        return utils.coref_extractor.coref_resolved(self.content)
        
    def create_oieresult(self):
        coref_content=self.content
        sents = nltk.tokenize.sent_tokenize(coref_content)
        sents = [{"sentence":s} for s in sents] #Format for oie batch predictor
        oie_result = utils.open_info_extractor.predict_batch_json(sents)
        oie_result = [i['verbs'] for i in oie_result]
        return oie_result
    
    def add_phrase(self, phrase, phrase_type):
        Id=len(self.phrase_corpus)
        self.phrase_corpus.append(phrase)
        self.phrase_type += " <{}> ".format(phrase_type)
        self.phrase_index = hnswlib.Index('cosine', 512)
        self.phrase_index.init_index(len(self.phrase_corpus), ef_construction=200, M=48, random_seed=36)
        if len(self.phrase_corpus) > 1:
            self.phrase_index.load_index("phrase_index", max_elements=len(self.phrase_corpus))
        self.phrase_index.add_items(utils.model([phrase]))
        self.phrase_index.save_index("phrase_index")
        return Id, phrase

    def deduplicate(self,phrase, phrase_type):
        if len(self.phrase_corpus)==0:
            return self.add_phrase(phrase, phrase_type)
        nearest_neighbor=self.phrase_index.knn_query(utils.model([phrase]))
        if nearest_neighbor != []:
            closest_neighbor, closest_distance = nearest_neighbor
        if closest_neighbor[0] == []:
            return self.add_phrase(phrase, phrase_type)
        if closest_distance[0][0] > distance_threshold:
            return self.add_phrase(phrase, phrase_type)
        return_phrase=self.phrase_corpus[closest_neighbor[0][0]]
        return self.phrase_corpus.index(return_phrase),return_phrase

    def create_training(self,verb_dict, verb_list):
        #self.change_comma()
        triplets = self.create_oieresult()
        return_text=""
        for sentence in triplets:
            if len(sentence)==0:
                self.parsed.append('str(len(self.phrase_corpus))+" -1"')
                continue
            text=re.sub('\[[^\s]*','',sentence[0]['description'])
            text=re.sub('\]','',text).split()
            tags=[False]*len(sentence[0]['tags'])
            phrasetype_prefix = [False]*len(sentence[0]['tags'])
            for triplet in sentence:
                arg_points=[x in ['I-ARG0','B-ARG0','I-ARG1','B-ARG1'] for x in triplet['tags']]
                abort=False
                for others in sentence:
                    for place in range(len(others['tags'])):
                        if others['tags'][place][-2:]=='-V' and arg_points[place]:
                            abort=True
                            break
                        if abort:
                            break
                if abort:
                    continue
                subject=' '.join([text[x] for x in range(len(text)) if triplet['tags'][x] in ['I-ARG0','B-ARG0']])
                objekt=' '.join([text[x] for x in range(len(text)) if triplet['tags'][x] in ['I-ARG1','B-ARG1']])
                verb=triplet['verb']
                verb=utils.lemmatize(triplet['verb'])
                if verb in auxillary_verbs:
                    continue
                if len(subject)==0:
                    continue
                if len(objekt)==0:
                    continue
                if verb in verb_dict.keys():
                    verb_id=verb_dict[verb]
                else:
                    verb_id=len(verb_list)
                    verb_list.append(verb)
                    verb_dict[verb]=verb_id

                max_id=len(self.phrase_corpus)
                subject_id,subject=self.deduplicate(subject, phrase_type='subject')
                if subject_id==max_id:
                    self.parsed.append(str(subject_id))
                tags=[str(subject_id) if triplet['tags'][x] in ['I-ARG0','B-ARG0'] else tags[x] for x in range(len(text))]
                phrasetype_prefix=['subject' if triplet['tags'][x] in ['I-ARG0','B-ARG0'] else phrasetype_prefix[x] for x in range(len(text))]
                
                max_id=len(self.phrase_corpus)
                objekt_id,objekt=self.deduplicate(objekt, phrase_type='object')
                if objekt_id==max_id:
                    self.parsed.append(str(objekt_id))
                tags=[str(objekt_id) if triplet['tags'][x] in ['I-ARG1','B-ARG1'] else tags[x] for x in range(len(text))]
                phrasetype_prefix=['object' if triplet['tags'][x] in ['I-ARG1','B-ARG1'] else phrasetype_prefix[x] for x in range(len(text))]

                if (subject,objekt,verb) not in self.triplet:
                    self.triplet.append((subject,objekt,verb))
                    self.triplet_id.append((subject_id,verb_id,objekt_id))
                    self.parsed.append("str(len(self.phrase_corpus)+"+str(len(self.triplet))+")")
            self.parsed.append('str(len(self.phrase_corpus))+" -1"')
            text=['<{}_'.format(phrasetype_prefix[x])+str(tags[x])+'>' if tags[x] else text[x] for x in range(len(text))]
            text.append(None)
            text=[text[x] for x in range(len(text)-1) if (text[x]!=text[x+1] or text[x][0]!='<')]
            return_text=return_text+' '+' '.join(text)

        masked_text = return_text[1:]
        parsed = [eval(x, {"self": self}) for x in self.parsed]
        preprocessed_row = [self.phrase_corpus, self.phrase_type.strip(), self.triplet_id, masked_text, parsed]
        return preprocessed_row, verb_dict, verb_list

    def create_test(self,verb_dict, verb_list):
        self.change_comma()
        triplets = self.create_oieresult()
        return_text=""
        for sentence in triplets:
            if len(sentence)==0:
                self.parsed.append('str(len(self.phrase_corpus))+" -1"')
                continue
            text=re.sub('\[[^\s]*','',sentence[0]['description'])
            text=re.sub('\]','',text).split()
            tags=[False]*len(sentence[0]['tags'])
            for triplet in sentence:
                arg_points=[x in ['I-ARG0','B-ARG0','I-ARG1','B-ARG1'] for x in triplet['tags']]
                abort=False
                for others in sentence:
                    for place in range(len(others['tags'])):
                        if others['tags'][place][-2:]=='-V' and arg_points[place]:
                            abort=True
                            break
                        if abort:
                            break
                if abort:
                    continue
                subject=' '.join([text[x] for x in range(len(text)) if triplet['tags'][x] in ['I-ARG0','B-ARG0']])
                objekt=' '.join([text[x] for x in range(len(text)) if triplet['tags'][x] in ['I-ARG1','B-ARG1']])
                verb=triplet['verb']
                verb=utils.lemmatize(triplet['verb'])
                if verb in auxillary_verbs:
                    continue
                if len(subject)==0:
                    continue
                if len(objekt)==0:
                    continue
                verb = verb.upper()
                if verb not in verb_dict.keys():
                    print(verb)
                    continue #if verb does not exist in verb_dict it can not be used to create
                verb_id=verb_dict[verb]
                max_id=len(self.phrase_corpus)
                subject_id,subject=self.deduplicate(subject, 'subject')
                if subject_id==max_id:
                    self.parsed.append(str(subject_id))
                tags=[str(subject_id) if triplet['tags'][x] in ['I-ARG0','B-ARG0'] else tags[x] for x in range(len(text))]

                max_id=len(self.phrase_corpus)
                objekt_id,objekt=self.deduplicate(objekt, 'object')
                if objekt_id==max_id:
                    self.parsed.append(str(objekt_id))
                tags=[str(objekt_id) if triplet['tags'][x] in ['I-ARG1','B-ARG1'] else tags[x] for x in range(len(text))]
                if (subject,objekt,verb) not in self.triplet:
                    self.triplet.append((subject,objekt,verb))
                    self.triplet_id.append((subject_id,verb_id,objekt_id))
                    self.parsed.append("str(len(self.phrase_corpus)+"+str(len(self.triplet))+")")
            self.parsed.append('str(len(self.phrase_corpus))+" -1"')
            text=['<phrase_'+str(tags[x])+'>' if tags[x] else text[x] for x in range(len(text))]
            text.append(None)
            text=[text[x] for x in range(len(text)-1) if (text[x]!=text[x+1] or text[x][0]!='<')]
            return_text=return_text+' '+' '.join(text)
        return self.phrase_corpus, self.triplet_id, return_text[1:], [eval(x, {"self": self}) for x in self.parsed]

if __name__ == "__main__":
    df = pd.read_csv("./data/presidency_project/newsconference/dem_train.csv")
    answer = df.answer.iloc[2]
    a = Answer(answer)
    phrase_corpus, phrase_type, triplet_id, masked_text, parsed = a.create_training({}, [])