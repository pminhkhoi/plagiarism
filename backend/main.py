from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataset import Dataset
from model import Model


origins = [
    'http://localhost:5173/'
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

class Query(BaseModel):
    content: str
    k: int

class ListQuery(BaseModel):
    content: list
    k: int

@app.post("/predict_string/", response_model=Query)
async def predict_string(query: Query):
    corpus = Dataset(punct_file='./punctuation.txt', stopword_file='./stopwords.txt', data_path='./applications.json')
    model = Model(corpus=corpus, q = [query.content])
    result = model.get_top_k(query.k)
    return result

@app.post("/predict_list_string/", response_model=ListQuery)
async def predict_list_string(query: ListQuery):
    corpus = Dataset(punct_file='./punctuation.txt', stopword_file='./stopwords.txt', data_path='./applications.json')
    model = Model(corpus=corpus, q = ListQuery.content)
    result = model.get_top_k(query.k)
    return result