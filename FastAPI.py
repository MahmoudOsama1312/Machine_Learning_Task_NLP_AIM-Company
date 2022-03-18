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
def MLmodel(input : inputtext):
    x = input.sentence
    return {"output" : loaded_model.predict(TFIDF.transform([x]))[0]}

## -------------------------------------------------DEEP LEARNING MODEL--------------------------------

from transformers import BertModel
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
        self.classifier = torch.nn.Linear(768, 18)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        return output




from transformers import AutoTokenizer, AutoModelForMaskedLM , BertForSequenceClassification 
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")
model = BERTClass()

# Load
# Specify a path
PATH = "state_dict_model.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

label_dict = {0 : 'AE', 1 : 'BH', 2 : 'DZ', 3 : 'EG', 4 : 'IQ', 5 : 'JO', 6 : 'KW', 7 : 'LB', 8 : 'LY',
              9 : 'MA', 10 : 'OM', 11 : 'PL', 12 : 'QA', 13 : 'SA', 14 : 'SD', 15 : 'SY', 16 : 'TN', 17 : 'YE'}

    
class inputtext2(BaseModel):
    sentence2 : str
        
@app.post("/DeepLearningModel/")
def deeplearning(input : inputtext2):
    y = input.sentence2
    inputs = tokenizer([y])
    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
    model(input_ids,attention_mask)
    return {"output": label_dict[np.argmax(model(input_ids,attention_mask).detach().numpy())][0]}
    #return {"output":[ label_dict[pred["label"]] for pred in pipe(y)] [0]}