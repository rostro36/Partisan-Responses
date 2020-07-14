# Partisan-Responses
Project for the class "Legal DNA"
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

## Pipeline
Data Acquisition (starting from the United States Congressional Record)

- Extract question-answer pairs from the whole corpus of speeches
  - Find speeches comprising of a single interrogative sentence and take the question together with the next speech as an answer (as otherwise it is unclear to which question the answer responds)
  - Extract questions that contain at least one of the phrases found in the phrase clusters provided by the authors (as these are certainly not organisational questions)
  - Remove question-answer pairs in which the answer is either too long or too short
  - Remove remaining organisational questions (e.g. Madam Speaker, may I inquire as to the time left on each side?)
- Map each speech to the corresponding party 
- ??? Perform error correction as speeches from earlier years contain many spelling mistakes or words not segmented properly
  - Spelling correction
  - Segment words
  
  
Training 

- Feed into the text generation model ([Graph Writer](https://arxiv.org/pdf/1904.02342.pdf)) each question, together with the real answer (from the corpus) and a precomputed knowledge graph made from the answer
  - Knowledge Graph:
    - Use [Open Information Extraction](https://demo.allennlp.org/open-information-extraction) and [Coreference Resoltuion](https://demo.allennlp.org/coreference-resolution) to extract triplets of the form (subject, verb, object) from the answer
    - Add triplets to the graph making sure that duplicates are not added (using Universal sentence Encoder to check similarity)
    
    
Text Generation 
- Given a question and a party (R/D), extract the top k most relevant speeches from the corpus using TF-IDF
- Construct a knowledge graph (using the same procedure as before) from the extracted speeches
- Feed the question and the knowledge graph to the trained model to produce an answer

## Current documents
- [Project Requirements](https://docs.google.com/document/d/1oli_He_bl7CpDNeu28eJwPZsJZV_k54V2JeaPlcVBsA/edit)
- [Overleaf](https://www.overleaf.com/project/5f0da15855ac0b00018d532f)
