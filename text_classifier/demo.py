import nltk
nltk.download('punkt')
import xml.etree.ElementTree as ET
import pickle
import queue
import json
import numpy as np
import nltk
from os import listdir
from os.path import isfile
from os.path import join

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.stem.snowball import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import Text

black_list = stopwords.words('english')
stemmer = SnowballStemmer("english")


def stemming_tokenizer(text):
    words = [stemmer.stem(word) for word in word_tokenize(text) if
             word.isalpha() and len(word) > 2 and word not in black_list]
    return words


# load precalculated binary classifiers
loc= "C://Users//Pawandeep//Desktop//Classification//repo//Scripts//"
labeling = ["Map", "Heatmap", "Area_Chart", "Column_Chart", "Ordination_Scatterplot", "Line_Chart", "NoViz",
            "Dendrogram", "Scatterplot", "Boxplot", "Timeseries", "Histogram", "Node-link_Diagram", "Stacked_Area_Chart", "Pie_Chart"]
old_labeling = ["Map", "Heatmap", "Area_Chart", "Column_Chart", "Ordination_Scatterplot", "Line_Chart", "NoViz",
            "Dendrogram", "Scatterplot", "Boxplot", "Timeseries", "Histogram", "Node-link_Diagram",
            "Stacked_Area_Chart", "Pie_Chart", "Proportion"]
classifiers = [pickle.load(open(loc+label + ".pickle", "rb")) for label in labeling]
vectorizer = pickle.load(open(loc+"vectorizer.pickle", "rb"))


file=open("C://Users//Pawandeep//Desktop//Classification//repo//example.txt")
text= file.read()
print(type(text))
# # load metadata
# meta_path = "befchina372/"
# meta_files = [f for f in listdir(meta_path) if isfile(join(meta_path, f))]
#
# ns = {'eml': 'eml://ecoinformatics.org/eml-2.1.0'}
#
# i = 0
# for f in meta_files:  # for each metadata (lets take the first 5)
#     tree = ET.parse(join(meta_path, f))
#     root = tree.getroot()
#
#     # for bfchina for now only the abstract
#     text = " ".join(root.itertext())
#
print("Predictions")
for (label, classifier) in zip(labeling, classifiers):
    print(label, classifier.predict_proba([text])[0,1])
    print("\n\n")
