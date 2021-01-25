# Hotel_Reviews_Classifier
Classifies hotel reviews as True/Fake and Positive/Negative
The word tokens in the reviews are used as features for the classification.

Classifiers: 
1. Naive Bayes Classifier
2. Perceptron Classifier

* nblearn - learns a naive bayes model from the data
* nbclassify - classifies the reviews and assigns labels as true/fake and positive/negative

Smoothing was performed to deal with unknown vocabulary in the test data

* perceplearn - learns vanilla and averaged models from the data
* percepclassify - uses the vanilla and averaged models to classify the data
