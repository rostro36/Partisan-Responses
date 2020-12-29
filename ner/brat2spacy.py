path = '/Users/Mithomas 1/Desktop/ETH/fs20/legal_nlp/Partisan-Responses/brat/data/newsconf_demo/3000_338.ann'
raw = '/Users/Mithomas 1/Desktop/ETH/fs20/legal_nlp/Partisan-Responses/brat/data/newsconf_demo/3000_338.txt'
with open(path, 'r') as f:
    annotation = f.readlines()
    ent_type, start, end = annotation[0].split('\t')[1].split()
    start, end = int(start), int(end)
with open(raw, 'r') as f:
    text = f.read()
    print(text[start:end])

import spacy
from spacy.gold import biluo_tags_from_offsets
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
entities = [(start, end, ent_type)]
tags = biluo_tags_from_offsets(doc, entities)
print(tags)