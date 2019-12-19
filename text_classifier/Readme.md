* Binary visualization classifiers: All files with extension pickle preeceding with the visualization names are binary classifier except vectorizer.pickle
* vectorizer.pickle: 
* ensembleclassifier.py: Takes all the .pickle files and bind or compile it to one assembly classifier. 
* ensemble.py: Runs the ensembleclassifier.py on the dataset and do training and validation. 
* apply_to_real_data.py: It contains the algorithmn for incemental learning. It runs that on the trained model from ensemble.py to tag data.
* classify_metadata.py: To classify bexis metadta
* classify_metadata_befchina.py: To classify befchina metadata
* Binary text classification models are available at https://github.com/fusion-jena/Biodiv-Visualization-Classifier/releases/download/stable/binary_classifiers.rar . Please download them and run through the python executable files in this folder.
