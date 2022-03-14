#!/usr/bin/env python
# coding: utf-8
from fastapi import FastAPI
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel

app= FastAPI()

@app.get("/Creator/")
def home():
    return {"Creator":"Mahmoud Osama Mohamed Mohamed "}

    

filename = 'MachineLearningModelSVC.sav'
loaded_model = pickle.load(open(filename, 'rb'))

filename2 = 'TFIDF2.pkl'
TFIDF = pickle.load(open(filename2, 'rb'))

##-------------------------------------------------MACHINE LEARNING MODEL------------------------------
class inputtext(BaseModel):
    sentence : str
        
@app.post("/MachineLearningModel/")
def model(input : inputtext):
    x = input.sentence
    return {"output" : loaded_model.predict(TFIDF.transform([x]))[0]}

## -------------------------------------------------DEEP LEARNING MODEL--------------------------------
from transformers import AutoTokenizer, AutoModelForMaskedLM , BertForSequenceClassification 
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
model = BertForSequenceClassification.from_pretrained("aubmindlab/bert-base-arabertv02-twitter", num_labels = 18)
model.train()
from transformers import TextClassificationPipeline
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

label_dict = {'LABEL_0' : 'AE', 'LABEL_1' : 'BH', 'LABEL_2' : 'DZ', 'LABEL_3' :'EG', 'LABEL_4' : 'IQ', 'LABEL_5' : 'JO', 'LABEL_6' :'KW', 'LABEL_7' :'LB', 'LABEL_8' : 'LY',
              'LABEL_9': 'MA', 'LABEL_10' : 'OM', 'LABEL_11' : 'PL', 'LABEL_12' :'QA', 'LABEL_13' :'SA', 'LABEL_14' :'SD', 'LABEL_15' :'SY', 'LABEL_16' : 'TN', 'LABEL_17' : 'YE'}

    
class inputtext2(BaseModel):
    sentence2 : str
        
@app.post("/DeepLearningModel/")
def deeplearning(input : inputtext2):
    y = input.sentence2
    for pred in pipe(y):
        return {"output": label_dict[pred["label"]]}
    #return {"output":[ label_dict[pred["label"]] for pred in pipe(y)] [0]}