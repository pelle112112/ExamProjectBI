import spacy
from spacy import displacy
import langdetect
from langdetect import DetectorFactory, detect, detect_langs
from spacy.lang.en import English
import pandas as pd
from sentence_transformers import SentenceTransformer, util
data = pd.read_csv('Data\megaGymDataset.csv', index_col='index')

allDocs = []
for index in data.index:
    allDocs.append(f"{data['Title'][index]} - {data['Desc'][index]} - {data['BodyPart'][index]} - {data['Equipment'][index]} - {data['Level'][index]} - {data['Rating'][index]}")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpusEmbeddings = embedder.encode(allDocs)

def getResponse(input):
    

    #input = ['Is there an exercise which includes dumbbells for adominal movement?']
    embeddedInput = embedder.encode(input)

    similarities = util.cos_sim(embeddedInput, corpusEmbeddings)[0]
    
    index = similarities.argmax().item()
    print(index)
    bestSimilarity = similarities[index]
    response = data.iloc[index]
    '''
    for index, embedding in enumerate(corpusEmbeddings):
        similarity = util.cos_sim(embeddedInput, embedding)
        if similarity > bestSimilarity:
            bestSimilarity = similarity
            response = data.iloc[index]

    #print(bestSimilarity)
    #print(response)
    '''
    return response, bestSimilarity
'''
# function to detect the language
def langDetect(text):    
    mylang = ''        # language
    mylangprob = 0.0   # probability 
    try:
        langs = langdetect.detect_langs(text)
        mylang, mylangprob = langs[0].lang, langs[0].prob 
        print('Detected language:', mylang, mylangprob)
        
        if mylang=='en':     
            # default = English()  # Include English language data    
            model = 'en_core_web_md'
        
    # another language?
    except langdetect.lang_detect_exception.LangDetectException:
        print('Language not recognised')
        pass
    return model

model = langDetect(allDocs[0])

# function to split the text into tokens (words) and discover their role in it
def tokenize(text, model):
        
    mytokens = []
    nlp = spacy.load(model)
    text = text.lower()
    doc = nlp(text)
    stopw = nlp.Defaults.stop_words
    print(stopw)  # uncomment to see them
    
    print("\nPipeline:", nlp.pipe_names, "\n")    
    for token in doc:
        if not (token.is_stop or token.is_punct or token.is_space):
            data = {'token': token.text,
                    'lemma': token.lemma_, 
                    'POS': token.pos_, 
                    'tag': token.tag_, 
                    'ent_type': token.ent_type_,
                    'vector': token.has_vector,  # vectors come with the model
                    'oov': token.is_oov # out of vocabulary of the selected model
                   }
            print(data)
            # print('vector: ', token.vector)  # uncomment to see the vectors
            mytokens.append(token.text)
    clean_text = " ".join(mytokens)
    options = {"compact": True, "color": "blue"}
    spacy.displacy.render(doc, style="ent")
    spacy.displacy.render(doc, style="dep", options=options)
    return clean_text

token1 = tokenize(allDocs[0], model)
token2 = tokenize(allDocs[56], model)
nlp = spacy.load(model)
doc1 = nlp(token1)
doc2 = nlp(token2)
print(doc1.similarity(doc2))

input = 'Is there an exercise which includes dumbbells for adominal movement?'
inputToken = tokenize(input, model)
inputDoc = nlp(inputToken)

bestSimilarity = 0.0
output = None

for index, doc in enumerate(allDocs):
    token = tokenize(allDocs[index], model)
    doc = nlp(token)
    similarityScore = doc.similarity(inputDoc)

    if similarityScore > bestSimilarity:
        bestSimilarity = similarityScore
        output = doc

print(output)
'''