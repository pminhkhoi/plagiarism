from pyvi.ViTokenizer import ViTokenizer
import os

class Processing:
    def __init__(self, data: str) -> None:
        if not isinstance(data, str):
            raise TypeError("Input must be in type string!")
        self._data = data
        self._tokenizer = ViTokenizer
        self._tokens = None

    def tokenize(self) -> list:
        self._tokens = self._tokenizer.tokenize(self._data.lower()).split()
        return self._tokens

    def remove_punctuation(self, punct_file: str) -> list:
        if not os.path.isfile(punct_file):
            raise FileNotFoundError("Input file does not exist!")

        with open (punct_file) as f:
            puncts = set(f.read().split())
        
        self._tokens = [token for token in self._tokens if token not in puncts]
        return self._tokens

    def remove_stop_words(self, stopword_file: str) -> list:
        if not os.path.isfile(stopword_file):
            raise FileNotFoundError("Input file does not exist!")
        
        with open(stopword_file) as f:
            stop_words = set(f.read().split())
        
        self._tokens = [token for token in self._tokens if token not in stop_words]
        return self._tokens

    def processing_pipeline(self, punct_file: str, stopword_file: str):
        self.tokenize()
        self.remove_punctuation(punct_file)
        self.remove_stop_words(stopword_file)
        return self._tokens


if __name__ == "__main__":
    obj = Processing('Hôm nay thời tiết đẹp quá nhỉ?')
    print(obj.processing_pipeline('./punctuation.txt', './stopwords.txt'))