#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:56:04 2019

@author: tarun.bhavnani
"""

import unicodedata
import re
import pickle
from mtranslate import translate
import re

from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, LSTM
import pandas as pd
import numpy as np
import os
os.chdir('/home/tarun.bhavnani/Desktop/Projects/rasa_custom/final_bot71/tarun_nlp3')
os.listdir()

"""
#create clean right data from lnu_data.json, or directly read nlu_train_data.csv

data=pd.read_json("nlu_data.json")
hj=[i for i in data["rasa_nlu_data"].iloc[0]]
#hj[0]['intent'], hj[0]['text']


intents=[]
texts=[]
for i in range(0,len(hj)):
    intents.append(hj[i]['intent'])
    texts.append(hj[i]['text'])


from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()
#texts1=[translate(i) if ad.only_alphabet_chars(i, "LATIN") ==False else i for i in texts ]    
#texts1  is all the hindi scripts translated
dat=pd.DataFrame({"intent":intents, "text": texts})

dat['new']=dat['text']
for i in range(77,len(dat)):
    print(i)
    #print(dat['intent'].iloc[i])
    if ad.only_alphabet_chars(dat['text'].iloc[i], "LATIN") ==False and ad.only_alphabet_chars(dat['new'].iloc[i], "LATIN")==False:
        dat['new'].iloc[i]= translate(dat['text'].iloc[i])
        print(dat['new'].iloc[i])
    else:
        dat['new'].iloc[i]=dat['text'].iloc[i]

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w=str(w)
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^0-9a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()
    return w


#create df
#dat=pd.DataFrame()
dat["text_clean"]= [preprocess_sentence(i) for i in dat['new']]
#dat['intent']=[i for i in intents]

#dat.to_csv("nlu_train_data.csv")

"""

dat= pd.read_csv("nlu_train_data.csv")
dat["text_clean"]= [preprocess_sentence(i) for i in dat['new']]


#dat1= dat[dat["intent"]!="inform"]

#tokenizer
#put all the alphabets / plphanumeric and numers in tokenizers for char level, see how oov works for char level

tok= Tokenizer(num_words=2000, oov_token="-OOV-", char_level=False)
#tok.fit_on_texts(dat["text_clean"])

tok.fit_on_texts(dat['text_clean'].values)

X=tok.texts_to_sequences(dat["text_clean"])
#[len(i) for i in X]



#create X and y
MAX_SEQUENCE_LENGTH=200
X= pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

le=LabelEncoder()
dat["labels"]=le.fit_transform(dat["intent"])
Y= pd.get_dummies(dat["labels"])



#create glove embedding matrix 

embeddings_index = {}
#glove_dir="C:\\Users\\tarun.bhavnani\\Desktop\\embed_kera\\glove.6B"
glove_dir='/home/tarun.bhavnani/Desktop/destop/glove_data'
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'), encoding="Latin1")
for line in f:
    #print(line)
    values = line.split()
    word = values[0]
    try:
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
    except:
        print(word)
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=100
embedding_matrix = np.zeros((len(tok.word_index) + 1, EMBEDDING_DIM))
for word, i in tok.word_index.items():
    print(i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.layers import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping


embedding_layer = Embedding(len(tok.word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(128)(embedded_sequences)
x=  Dense(32, activation="relu")(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(set(dat["intent"])), activation="softmax")(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history= model.fit(X,Y, epochs=300, batch_size=32,callbacks=[early_stop])



"""
#Model
model= Sequential()
#model.add(Input(shape= input_shape))
model.add(Embedding(2000, 128, input_length=X.shape[1]))
model.add(Dropout(rate=.2))

model.add(LSTM(128))
model.add(Dense(32, activation="relu"))
#model.add(Dropout(rate=.2))#with this 57 epochs acc:91.38

model.add(Dense(len(set(dat["intent"])), activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer="adam")

#history=model.fit(X,Y, validation_split=.3, batch_size=12, epochs=30)
#history=model.fit(X,Y, batch_size=12, epochs=50)#acc:.9203, without 92.78 epochs 70
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history= model.fit(X,Y, epochs=300, batch_size=32,callbacks=[early_stop])



  check why acc low in this

from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, MaxPooling1D, GlobalAveragePooling1D, SeparableConv1D, Flatten

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim= 128,input_length=X.shape[1]))
model.add(Dropout(rate=.2))
model.add(SeparableConv1D(filters=32, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
    
model.add(SeparableConv1D(filters=32, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
model.add(MaxPooling1D())
    #model.add() 
model.add(SeparableConv1D(filters=64, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))
model.add(SeparableConv1D(filters=64, kernel_size=3, padding="same", dilation_rate=1,
                              activation="relu", bias_initializer="random_uniform",
                              depthwise_initializer="random_uniform"))

model.add(GlobalAveragePooling1D())
model.add(Dropout(rate=.2))
model.add(Dense(len(set(dat["intent"])), activation="softmax"))
model.summary()
    

model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer= "adam")
"""



#put early stoopage and more epochs
#Save Model

#save encoder model
np.save('classes.npy', le.classes_)

#save tokenizer

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)


# serialize model to JSON

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")






#predict


txt="30 Percent or less"
txt='moving pussy'
txt="haven't done"#haven't not there
txt="yes yes i did it"
txt='good'


txt=[tok.word_index[i] if i in tok.word_index else 1 for i in txt.lower().split()]
txt=np.asarray(txt)
txt= txt.reshape(1,txt.shape[0])
from keras.preprocessing.sequence import pad_sequences
txt=pad_sequences(txt, maxlen=MAX_SEQUENCE_LENGTH)

y_p=model.predict(txt)
predicted_intent = le.inverse_transform(y_p.argmax(axis=-1))[0]
print(predicted_intent)


ranking_dat=[{"name": le.inverse_transform(i), "confidence": j} for i,j in enumerate(y_p[0])]
ranking_dat= pd.DataFrame(ranking_dat)
ranking_dat=ranking_dat.sort_values("confidence", ascending=False).reset_index().drop(["index"], axis=1)

#ranking_dat=ranking_dat.reset_index().drop(["index"], axis=1)
intents=[i for i in ranking_dat.name]
probabilities=[i for i in ranking_dat.confidence]
ranking= [(i,j) for i,j in zip(ranking_dat['name'], ranking_dat['confidence'])]


##################end#############33
#intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
#intent = {"name": intents[0], "confidence": probabilities[0]}
"""
