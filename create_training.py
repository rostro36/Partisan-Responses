import pandas as pd
from Answer import Answer
import utils
import re

def parse_entry(entry,verb_dict,verb_list):
    result=dict()
    result['question']=' '.join([token.text for token in utils.sp(entry['question'])])
    phrase_corpus, triplet_id, parsed_text,parsed=Answer(entry['answer']).create_training(verb_dict,verb_list)
    result['corpus']=' ; '.join(phrase_corpus)
    result['tags']=' '.join(['<phrase>']*len(phrase_corpus))
    result['triplet_id']=' ; '.join([re.sub('\,','',str(x))[1:-1] for x in triplet_id])
    result['parsed_text']=parsed_text
    result['parsed']=' '.join([str(x) for x in parsed])
    return result

if __name__ == "__main__":
    df = pd.read_pickle("final_data.pkl")

    verb_dict = dict()
    verb_list = []
    result = []
    df[:5].apply(lambda x: result.append(parse_entry(x, verb_dict, verb_list)), axis=1)
    m = pd.DataFrame(result)
    m.to_csv('test123.tsv', sep='\t', index=False, header=False)
    with open('test123.vocab', 'w') as filehandle:
        filehandle.writelines("%s\n" % verb.upper() for verb in verb_list)
