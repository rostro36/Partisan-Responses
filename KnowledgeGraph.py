import hnswlib
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow_hub as hub

class KnowledgeGraph:
    def __init__(self):
        self.phrase_corpus = []
        self.graph = nx.MultiDiGraph()
        self.sp = spacy.load('en_core_web_sm')
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.phrase_corpus_length = 1
        self.node_index = None
        self.distance_threshold = 0.5

    def lemmatize(self, phrase):
        return " ".join([word.lemma_ for word in self.sp(phrase)])

    def add_node(self, phrase):
        self.phrase_corpus_length +=  1
        self.phrase_corpus.append(phrase)
        if self.phrase_corpus_length > 2:
            self.node_index = None
        self.node_index = hnswlib.Index('cosine', 512)
        self.node_index.init_index(self.phrase_corpus_length, ef_construction=200, M=48, random_seed=36)
        if self.phrase_corpus_length > 2:
            self.node_index.load_index("node_index", max_elements=self.phrase_corpus_length)
        self.node_index.add_items(self.model([phrase]))
        self.node_index.save_index("node_index")
        # return

    def return_node(self, phrase):
        non_stop_phrase = ' '.join([token.text for token in self.sp(phrase)])
        if len(non_stop_phrase) > 1:
            phrase = non_stop_phrase
        if self.node_index is None:
            self.add_node(phrase)
        nearest_neighbor = self.node_index.knn_query(self.model([phrase]))
        if nearest_neighbor != []:
            closest_neighbor, closest_distance = nearest_neighbor
        if closest_neighbor[0] == []:
            self.add_node(phrase)
            return phrase
        if closest_distance[0][0] > self.distance_threshold:
            self.add_node(phrase)
            return phrase
        return self.phrase_corpus[closest_neighbor[0][0]]

    def other(self, partisanship):
        if partisanship == 'R':
            return 'D'
        if partisanship == 'D':
            return 'R'
        print(partisanship)
        return None

    def add_edges(self, preprocess_output):
        sentences = preprocess_output[0]
        partisanship = preprocess_output[1]
        for sentence in sentences:
            for phrase in sentence:
                subject = self.return_node(phrase[0])
                objekt = self.return_node(phrase[2])
                predicate = self.lemmatize(phrase[1])
                attributes = self.graph.get_edge_data(subject, objekt, predicate)
                if attributes:
                    self.graph.remove_edge(subject, objekt, predicate)
                    weight = attributes['weight'] + 1
                    if partisanship in ('D', 'R'):
                        attributes[partisanship] = attributes[partisanship] + 1
                    dem = attributes['D']
                    rep = attributes['R']
                else:
                    weight = 1
                    dem = 0
                    rep = 0
                    if partisanship == 'D':
                        dem = 1
                    elif partisanship == 'R':
                        rep = 1
                self.graph.add_edge(subject, objekt, key=predicate, weight=weight, D=dem, R=rep)


if __name__ == "__main__":
    G = KnowledgeGraph()
    # sample = [[[], [('I', 'like', 'apples')], [('I', 'introducing', 'the College Opportunity Tax Credit Act of 2009')],
    #            [('This legislation', 'creates',
    #              'a new tax credit that will put the cost of higher education in reach for American families'),
    #             ('a new tax credit', 'put', 'the cost of higher education')]], 'D']
    sample = [[('I', 'join', 'with the senior Senator from Arizona')],
             [],
             [],
             [],
             [],
             [],
             [('I', 'work', 'in fighting earmark reform over the last two decades')],
             [],
             [],
             [],
             [('Congress', 'enacted', 'the most significant earmark reform'),
              ('by Members', 'growing', 'recognition'),
              ('earmarks', 'avoid', 'the scrutiny of the authorizing process')],
             [],
             [],
             [('I', 'commend', 'our Majority Leader')],
             [],
             [('Presidentelect Obama',
               'ensuring',
               'the passage of the Lobbying and Ethics Reform measure')],
             [],
             [],
             [('Earmark Reform , and Accountability Act of 2009',
               'will',
               'build on the significant achievement'),
              ('Earmark Reform , and Accountability Act of 2009',
               'build',
               'on the significant achievement of the 110th Congress'),
              ('one', 'makes', 'it much more difficult to enact earmarks')],
             [],
             [('To overcome a point of order',
               'need',
               'to obtain a supermajority of the Senate'),
              ('supporters of the unauthorized earmark',
               'obtain',
               'a supermajority of the Senate')],
             [('the Lobbying and Ethics Reform measure',
               'provides',
               'that any earmarked funding which is successfully stricken from the appropriations bill will be unavailable for other spending in the appropriations bill'),
              ('from the appropriations bill', 'stricken', 'any earmarked funding')],
             [('the Lobbying and Ethics Reform measure', 'closes', 'a loophole'),
              ('all',
               'authorizing',
               'conference reports to be electronically searchable 48 hours before the Senate considers the conference report'),
              ('the Senate', 'considers', 'the conference report')],
             [('the Lobbying and Ethics Reform measure',
               'requires',
               'federal funds to disclose any money spent on registered lobbyists'),
              ('all recipients', 'disclose', 'any money spent on registered lobbyists'),
              ('Presidentelect Obama',
               'announced',
               'that the expected economic recovery package which may be proposed in the next few days should be kept free of earmarks , I could nt agree more'),
              ('I', 'agree', 'more'),
              ('I',
               'expect',
               'to join with Senators Mr. McCAIN , the junior Senator from Missouri'),
              ('I',
               'join',
               'with Senators Mr. McCAIN , the junior Senator from Missouri')],
             [],
             [('Mr. COBURN',
               'see',
               'that the expected economic recovery package which may be proposed in the next few days is free of unauthorized earmarks')],
             [('just', 'attracted', 'unauthorized earmarks')],
             [('We',
               'keeping',
               'the expected economic recovery package which may be proposed in the next few days and other appropriations bills free of'),
              ('We',
               'use',
               'the tools proposed in the Lobbying and Ethics Reform measure')],
             [('I',
               'ask',
               'unanimous consent that the text of the Lobbying and Ethics Reform measure be printed In the RECORD')],
             'D']

    G.add_edges(sample)

    options = {
        'node_color': 'green',
        'node_size': 200,
        'width': 1
    }
    pos=nx.circular_layout(G.graph)
    edge_labels=dict([((start,finish,), predicate+', '+str(weights['weight'])+', '+str(weights['R'])+', '+str(weights['D'])) for start,finish,predicate,weights in G.graph.edges(data=True,keys=True)])
    nx.draw(G.graph, pos, with_labels=True, font_weight='bold', **options)
    nx.draw_networkx_edge_labels(G.graph, pos,edge_labels=edge_labels)
    
    # plt.savefig("first_graph.png")
    plt.show()

    # print("Output for an edge that exists: e.g. ('I','ice cream','like')")
    # print(G.graph.get_edge_data('I', 'ice cream', 'like'))
    # print("Output for an edge that does not exist: e.g. ('I','ice')")
    # print(G.graph.get_edge_data('I', 'ice'))
    # print('All edges')
    # print(G.graph.edges)
    #
    # G.add_edges([[[('you', 'ate', 'a lot of oranges and pizza')]], 'R'])
    #
    # pos=nx.circular_layout(G.graph)
    # edge_labels=dict([((start,finish,), predicate+', '+str(weights['weight'])+', '+str(weights['R'])+', '+str(weights['D'])) for start,finish,predicate,weights in G.graph.edges(data=True,keys=True)])
    #
    # nx.draw(G.graph, pos, with_labels=True, font_weight='bold', **options)
    # nx.draw_networkx_edge_labels(G.graph, pos,edge_labels=edge_labels)
    #
    # plt.savefig("updated_graph.png")
