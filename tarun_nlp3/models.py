
import pandas as pd
import numpy as np
from keras.models import model_from_json
import pickle
from mtranslate import translate

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder 
import os
#os.chdir('/home/tarun.bhavnani/Desktop/Projects/rasa_custom/final_bot71')
import numpy as np


def tarun_nlp(txt):
 # load json and create model
 json_file = open('tarun_nlp3/model.json', 'r')
 loaded_model_json = json_file.read()
 json_file.close()
 loaded_model = model_from_json(loaded_model_json)
 # load weights into new model
 loaded_model.load_weights("tarun_nlp3/model.h5")
 #print("Loaded model from disk")
 loaded_model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")
 ###########################################################
 #load tokenizer
 with open('tarun_nlp3/tokenizer.pickle', 'rb') as handle:
     tok = pickle.load(handle)
 
 le = LabelEncoder()
 le.classes_ = np.load('tarun_nlp3/classes.npy',allow_pickle=True)

 #txt="what is ur name?"

 #import numpy as np
 
 #txt=translate(txt).lower()
 #it is taking time, see if optimized
 
 #txt=[tok.word_index[i] for i in txt.lower()]
 txt=[tok.word_index[i] if i in tok.word_index else 1 for i in txt.lower().split()]

 txt=np.asarray(txt)
 #txt.shape
 txt= txt.reshape(1,txt.shape[0])
 #txt=np.zeros((1,100))
 #txt=tok.texts_to_sequences(txt)
 txt=pad_sequences(txt, maxlen=200)

 y_p=loaded_model.predict(txt)

 #predicted_intent = encoder.inverse_transform(y_p.argmax(axis=-1))[0]
 ranking_dat=[{"name": le.inverse_transform([i]), "confidence": j}  for i,j in enumerate(y_p[0])]
 ranking_dat= pd.DataFrame(ranking_dat)
 ranking_dat=ranking_dat.sort_values("confidence", ascending=False).reset_index().drop(["index"], axis=1)

 #ranking_dat=ranking_dat.reset_index().drop(["index"], axis=1)
 intents=[i[0] for i in ranking_dat.name]
 probabilities=[i for i in ranking_dat.confidence]
 ranking= {i[0]:j for i,j in zip(ranking_dat['name'], ranking_dat['confidence'])}


 return intents, probabilities, ranking


#intents, probabilities, ranking=tarun_nlp("hi")
#from mtranslate import translate

#texts=translate('hi').lower()
#intent = {"name": intents[0], "confidence": probabilities[0], "translate":texts}
#intent_ranking = [{"name": i, "confidence": ranking[i]} for i in ranking]

