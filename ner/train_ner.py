import spacy 

def load_or_create_model(model=None):
    if model is not None:
        pipe = spacy.load(model)
    else:
        pipe = spacy.blank("en")
    return pipe 

def get_ner_component(pipe):
    if "ner" not in pipe.pipe_names:
        ner = pipe.create_pipe("ner")
        pipe.add_pipe(ner, last=True)
    else:
        ner = pipe.get_pipe("ner")
    return ner, pipe

def add_labels(data, ner):
    for _, ann in data:
        for ent in ann.get("entities"):
            ner.add_label(ent[2])
    return ner 

def train_batch(batch, pipe):
    texts, anns = zip(*batch)
    pipe.update(texts, 
                anns,
                drop= , #TODO
                losses = losses,  
                sgd=optimizer)
    return pipe

def eval(data):
    for text, _ in data:
        doc = pipe(text)

def main(epochs = 50):
    # init model
    pipe = load_or_create_model(model)
    ner, pipe = get_ner_component(pipe)
    # preprocessing
    ner = add_labels(training_data)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [p for p in pipe.pipe_names if p not in pipe_exceptions]
    with pipe.disable_pipes(*other_pipes) :
        #
        # Training new model
        if model is None:
            pipe.begin_training()
        for i in range(epochs):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=)
            for b in batches:
                pipe = train_batch(b, pipe)
        # print loss?
    # Eval
    # Save 
    