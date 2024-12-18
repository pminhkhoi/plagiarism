import json
import os
from bs4 import BeautifulSoup
from processing import Processing
import unicodedata

class Dataset:
    def __init__(self, punct_file: str, stopword_file: str, data_path = None, **kargs) -> None:
        if data_path:
            if os.path.isfile(data_path):
                self._data_path = data_path
                with open(self._data_path, 'r') as f:
                    self._data = json.load(f)

                self._data = [' '.join(BeautifulSoup(x['content'][0]['answer'], 'html.parser').stripped_strings).replace('\xa0', ' ').strip() for x in self._data]
            else:
                raise FileNotFoundError("Input file does not exist!")
        else:
            for _, v in kargs.items():
                self._data = [' '.join(BeautifulSoup(x, 'html.parser').stripped_strings).replace('\xa0', ' ').strip() for x in v]
                break

        self._corpus = [' '.join(Processing(data).processing_pipeline(punct_file=punct_file, stopword_file=stopword_file)) for data in self._data]
      
        self._dictionary = set()
        for sent in self._corpus:
            for token in sent.split():
                self._dictionary.add(token)

        self._punct_file = punct_file
        self._stopword_file = stopword_file


    def get_raw_data(self):
        return self._data
    
    def get_tokenized_data(self):
        return [row.split() for row in self._corpus]
    
    def get_normalized_data(self):
        return self._corpus
    
    def __len__(self):
        return len(self._data)
    
    def get_num_tokens(self):
        return len(self._dictionary)
    
    def get_dictionary(self):
        return self._dictionary
    
    def get_punct_file(self):
        return self._punct_file
    
    def get_stopword_file(self):
        return self._stopword_file

if __name__ == "__main__":
    obj = Dataset(punct_file="./punctuation.txt", stopword_file="./stopwords.txt", data_path="./applications.json")
    # obj = Dataset(punct_file="./punctuation.txt", stopword_file="./stopwords.txt", data = ["Hôm nay trời đẹp thế", "Bạn cảm thấy thế nào?"])
    print(obj.get_normalized_data())