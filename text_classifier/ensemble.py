import csv
import json
import ensembleclassifier
from ensembleclassifier import stemming_tokenizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


labeled_items = []

# all possible labels of the dataset (if there are no examples of one class, it is simply ignored) 
labeling = ["Map", "Heatmap", "Area_Chart", "Column_Chart", "Proportion", "Ordination_Scatterplot", "Line_Chart", "NoViz", "Dendrogram", "Scatterplot", "Boxplot", "Timeseries", "Histogram", "Node-link_Diagram", "Stacked_Area_Chart", "Pie_Chart"]
  
data = {}

# read in the annotated dataset (csv variant)
csvfile =  open('data_incremental.csv', newline='', encoding="utf-8")
reader = csv.DictReader(csvfile)
counter = 0
for row in reader:
    data[row["id"].strip()] = {"label": row["label"].strip(), "caption": row["caption"].strip()}
    counter += 1
csvfile.close()


# read in the annotated dataset (json variant)
#with open('labeled_data_iterative.json', newline='', encoding='utf-8') as jsonfile:
#    jsondata = json.loads(jsonfile.read())
#    for item in jsondata:
#        if len(item["label"]) == 1:
#            data[item["id"].strip()] = {"label": item["label"][0].strip(), "caption": item["caption"].strip()}

# remove some common phrases, without any information concerning visualization types
labeled_items = []
for ident in data:
    item = data[ident]
    item["id"] = ident
    item["raw"] = item["caption"].replace("For interpretation of the references to color in this figure legend", "")\
                                 .replace("For interpretation of the references to colour in this figure legend", "")\
                                 .replace("for interpretation of the references to colour in this figure legend", "")\
                                 .replace("for interpretation of the references to color in this figure legend", "")\
                                 .replace("For interpretation of the references to color in figure legend", "")\
                                 .replace("For interpretation of the color information in this figure legend", "")\
                                 .replace("For interpretation of text references to color in this figure legend", "")\
                                 .replace("the reader is referred to the web version of this article", "")\
                                 .replace("is referred to the web version of this paper", "")\
                                 .replace("the reader is referred to the web version of the article", "")

    labeled_items.append(item)
print("Nr of samples:", len(labeled_items), " of ", counter)

# extract the data and labels and initialize the ensemble
data = [item["raw"] for item in labeled_items]
correct_labels = [item["label"] for item in labeled_items]
classifier = ensembleclassifier.Ensemble(labeling)
#print("Accuracy: %0.2f in %0.2f of the corpus" % classifier.evaluate(data, correct_labels))

# split the dataset into training and test part
X_train, X_test, y_train, y_test = train_test_split(data, correct_labels, test_size=0.25, random_state=42)

print("Retraining")
classifier.train(data, correct_labels, save=True)

# test the accuracy (F1-score) for several confidence thresholds
for i in range(5):
    print("Retrained Accuracy:%0.2f in %0.2f of the corpus with min-confidence %0.2f with on average %0.2f labels" % classifier.evaluate(X_test, y_test, 0.5+i*0.1)[0:4])
