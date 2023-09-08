1. Stemming, Tokenization and Bag of Words
```
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass

words = ['organize', 'organs', 'organizing', 'organism']
a = [stem(w) for w in words]
print(a)
```

2. Create Training Data
3. Pytorch model and training
4. Save/Load model and implement the chat