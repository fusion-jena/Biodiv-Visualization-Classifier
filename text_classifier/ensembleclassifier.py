import pickle
import string
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.utils import shuffle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler 

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import path


black_list = stopwords.words('english') + list(string.punctuation)
stemmer = SnowballStemmer("english")

def stemming_tokenizer(text):
    words =  [stemmer.stem(word) for word in word_tokenize(text) if word.isalpha() and len(word) > 2 and word not in black_list]
    return words

class Ensemble:
    # loading previously saved binary classifiers
    def __init__(self, labels):
        self.labels = labels
        self.classifier = [pickle.load(open(label+".pickle", "rb")) for label in labels if path.exists(label+".pickle") ]
    
    # classifying a vector of texts; returns vector of labels (may be empty)
    def classify_batch(self, texts, threshold=0.9):
        results = []
        for (classifer, label) in zip(self.classifier, self.labels):
            results.append([i[1] for i in classifer.predict_proba(texts)])
        
        # using only the max confident label
        #return [max(zip(self.labels, x), key=lambda x: x[1]) for x in zip(*results)]
        
        # using all labels over a threshold
        return [[label for (label, value) in zip(self.labels, x) if value > threshold] for x in zip(*results)] 

    # classifying a single text; returns vector of labels (may be empty)
    def classify_single(self, text, threshold=0.9):
        return  self.classify_batch([text], threshold)

    # train the classifiers with the provided data; if save option is True, the classifiers are saved after training
    def train(self, training_data, training_labels, save=False):
        self.classifier = []
        for label in self.labels:
            right = [caption for (caption, caption_label) in zip(training_data, training_labels) if caption_label == label]
            print(label, len(right))
            if(len(right) > 0):
                false = resample([caption for (caption, caption_label) in zip(training_data, training_labels) if caption_label != label], n_samples=len(right), random_state=0)

                data = right + false
                labels = [True for item in right] + [False for item in false]

                shuffle(data, labels, random_state=0)

                classifier = Pipeline([
                    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer, ngram_range=(1,2), max_features=750, min_df=3, use_idf=True)),
                    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42)),   
                ])
                print(label + "   %s" % (sum(cross_val_score(classifier, data, labels, cv=5, scoring="f1")) / 5.0))
                
                classifier.fit(data, labels)
                self.classifier.append(classifier)

                if save: 
                    pickle.dump(classifier, open(label+".pickle", "wb"))

    # evaluates the classifiers using the supplied data; returns a vector of evaluation results [accuracy(F1-score), fraction of tagged data, confidence threshold, average number of labels, counts of multi-labels]  
    def evaluate(self, evaluation_data, evaluation_labels, threshold=0.9):
        y_pred = self.classify_batch(evaluation_data, threshold)
        error_list = [("-".join(pred), gold) for (pred, gold) in zip(y_pred, evaluation_labels) if len(pred) > 1]
        mapping = sorted([x for x in zip(Counter(error_list).keys(), Counter(error_list).values())], key=lambda x: x[0][0])
        return (sum([1 for (comp, gold) in zip(y_pred, evaluation_labels) if gold in comp]) / len([1 for comp in y_pred if len(comp) > 0]), len([1 for comp in y_pred if len(comp) > 0])/ len(evaluation_labels), threshold, sum([len(comp) for comp in y_pred])/len([1 for comp in y_pred if len(comp) > 0]), mapping) 
