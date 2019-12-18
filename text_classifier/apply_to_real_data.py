import csv
import json
import ensembleclassifier
from ensembleclassifier import stemming_tokenizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


labeled_items = []

labeling = ["Map", "Heatmap", "Area_Chart", "Column_Chart", "Proportion", "Ordination_Scatterplot", "Line_Chart", "NoViz", "Dendrogram", "Scatterplot", "Boxplot", "Timeseries", "Histogram", "Node-link_Diagram", "Stacked Area Chart", "Pie Chart"]
  
data = {}

csvfile =  open('labelData_new.csv', newline='', encoding="utf-8")
reader = csv.DictReader(csvfile)
for row in reader:
    data[row["id"].strip()] = {"label": row["label"].strip(), "caption": row["caption"].strip()}
csvfile.close()

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
print("Nr of samples:", len(labeled_items))

data = [item["raw"] for item in labeled_items]
correct_labels = [item["label"] for item in labeled_items]
classifier = ensembleclassifier.Ensemble(labeling)
#classifier.train(data, correct_labels, save=True) saved last time ^^

print("Retraining on whole labeled data finished")

unlabeled_data = json.loads(open("unlabeled_data.json","r").read())
labels = classifier.classify_batch([item["caption"] for item in unlabeled_data], 0.9)

percentage = sum([1 for label in labels if len(label)>0]) / len(labels)
print("Percentage of labeled captions:", percentage)
while percentage < 0.5:
    classifier.train(data + [datum["caption"] for (datum, label) in zip(unlabeled_data, labels) if len(label)==1], correct_labels + [label[0] for label in labels if len(label)==1], save=False)
    labels = classifier.classify_batch([item["caption"] for item in unlabeled_data], 0.9)

    percentage = sum([1 for label in labels if len(label)==1]) / len(labels)
    print("Percentage of labeled captions:", percentage)

labeled_data = [{"id": datum["id"], "caption": datum["caption"], "label": label} for (datum, label) in zip(unlabeled_data, labels)]
with open("labeled_data_iterative.json", "w") as file:
    file.write(json.dumps(labeled_data))
