from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
from sklearn.base import clone

from utils import *
import numpy as np
import matplotlib.pyplot as plt

def build_data(tablets):
	data = []
	labels = []

	# Build data point for each tablet
	for tablet in tablets:
		data_point = {}

		# Current tablet obj
		c_tablet = tablets[tablet]

		if c_tablet.year == -999 or c_tablet.year == 0: continue

		for pn in c_tablet.pns:
			data_point[pn] = c_tablet.pns[pn]

		for gn in c_tablet.gns:
			data_point[gn] = c_tablet.gns[gn]

		data_point['period'] = c_tablet.period
		data.append(data_point)

		labels.append(c_tablet.year)

	return data, labels

def save_result(predictions, targets, le):
	# Write predictions out to file
	output = open("model_predictions.txt", "w+")
	for i in range(0, len(targets)):
		pr = le.inverse_transform(predictions[i])
		ac = le.inverse_transform(targets[i])
		if pr != ac:
			output.write("Predicted: " + str(pr) + " Actual: " + str(ac) + "\n")
	output.close()

def train_model(model, training_data, training_targets):
	model.fit(training_data, training_targets)

def make_predictions(model, data, targets, le):
	y_pred = model.predict(data)

	inc_total = (targets != y_pred).sum()
	acc = (1 - float(inc_total) / float(data.shape[0])) * 100

	#print "Total: %6i Incorrect: %6i" % (data.shape[0], inc_total)
	#print "Accuracy: %.2f%%" % (acc)
	save_result(y_pred, targets, le)

	return acc

####################
####################

# Build tablet dataset
tablets = collect_pns_gns()
collect_tablet_years(tablets)

# Parse tablets into datapoints
data, labels = build_data(tablets)
v = DictVectorizer()
vectorizer = make_pipeline(v, TfidfTransformer())
data = vectorizer.fit_transform(data)

# Encode/normalize labels
le = LabelEncoder()
targets = le.fit_transform(labels)

models = [
	("MNB",        MultinomialNB(alpha=.1)),
	("Perceptron", Perceptron(alpha=.1)),
	("SGD",        SGDClassifier(alpha=.1))
]

graph_acc = []

for k in range(0, len(models)):
	validation_acc = []

	for i in range(0, 10):
		# Fetch Sample sets for training/testing
		ss = cross_validation.ShuffleSplit(data.shape[0], n_iter=1, test_size=.10)

		for train, test in ss:
			train_indecies = train
			test_indecies = test

		train_samples = data[train_indecies]
		train_targets = targets[train_indecies]

		test_samples = data[test_indecies]
		test_targets = targets[test_indecies]

		###################
		# Train and Predict
		###################
		# Current Model
		c_model = clone(models[k][1], True)

		# Train model
		train_model(c_model, train_samples, train_targets)

		# Run test predictions and compare to answer set
		validation_acc.append(make_predictions(c_model, test_samples, test_targets, le, ))

	# Print results
	best_acc = 0.0 
	worst_acc = 100.0
	for acc in validation_acc:
		if acc < worst_acc:
			worst_acc = acc
		if acc > best_acc:
			best_acc = acc

	validation = sum(validation_acc) / 10.0

	graph_acc.append((validation_acc, [validation], [best_acc], [worst_acc]))

	print "Model: %s" % (models[k][0])
	print "10-Fold Cross Validation Acc: %.2f%%" % (validation)
	print "Sample Size: %i" % (test_samples.shape[0])
	print "Best: %.2f%%" % (best_acc)
	print "Worst: %.2f%% \n" % (worst_acc)

# Plot accuracies
graph_data = []
for i in range(0, len(graph_acc)):
	graph_data.append(np.concatenate(graph_acc[i], 0))

plt.figure(0)
plt.boxplot(graph_data, 0, '')
plt.xticks([1, 2, 3], ['MNB', 'Perc', 'SGD'])

plt.show()






