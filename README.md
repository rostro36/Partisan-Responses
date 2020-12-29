# Partisan-Responses
Project for the class "Introduction of Natural Language Processing (Fall 2020)", an extension to a previous course project. 
## Dataset (in-progress)
- [Presidency Project](https://www.presidency.ucsb.edu/)
  - [News Conference](https://www.presidency.ucsb.edu/documents/app-categories/presidential/news-conferences)
)
  - [Presidential Campaings](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0")
)
- [Gallup Topic Questions](https://news.gallup.com/poll/trends.aspx#P)

## Build KG
### Pipeline
raw text -> coreference resolution -> annotation (entities and relations) -> Train NER (with spacy) -> Train semantic relation extraction(SRE)(not sure...) -> KG 

## Run the code
Follow through the numbered notebooks. Starting at 06 only the dataset from all speeches since 2000 gets executed.
## Important papers
Building Knowledge Graph
- [Opinion-aware Knowledge Graph for Political Ideology Detection](https://www.ijcai.org/Proceedings/2017/0510.pdf)
  - only half applies, since we do not have a background graph like ConceptNet
- [An Automatic Knowledge Graph Creation Framework fromNatural Language Text](https://pdfs.semanticscholar.org/eb1d/438e7aca8600cfd87d7b0ecfaf36f36f5c37.pdf)
- [Knowledge Graph Construction](https://hal.archives-ouvertes.fr/hal-02277063/document)
  - [Coreference Resolution](https://github.com/huggingface/neuralcoref)
  - [Open Information Extraction](https://demo.allennlp.org/open-information-extraction)
  - "To merge nodes, the TF-IDF overlap of the new node’s name is calculated with the existing graph node names, and the new node is merged into an existing node if theTF-IDF  is  higher  than  some  threshold."
  - [Graph Engine](https://networkx.github.io/)
  
Graph to text 
- [Enhancing Topic-to-Essay Generation with External Commonsense
Knowledge](https://www.aclweb.org/anthology/P19-1193.pdf)
- [Graph Writer](https://arxiv.org/pdf/1904.02342.pdf)
- [Graph Attention Networks](https://github.com/PetarV-/GAT)
