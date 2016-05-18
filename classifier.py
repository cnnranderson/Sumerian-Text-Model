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
import os
import matplotlib.pyplot as plt

def build_data(tablets, unknown=False):
	'''
	Input:
		tablets - Takes in the tablet object data that has all pns/gns/year of tablet

	This method creates data points (read: feature vectors) out of every tablet
	from a build up dictionary of all tablets. These should be read in from
	the Texts.csv file, and processed with collect_pns_gns, and collect_years
	methods. 

	Returns:
		data - the list of feature vectors (known as X)
		labels - the targets of each data point (known as Y)
	'''
	data = []
	labels = []

	# Build data point for each tablet
	for tablet in tablets:
		data_point = {}

		# Current tablet obj
		c_tablet = tablets[tablet]

		if unknown:
			if c_tablet.year == -999 or c_tablet.year == 0:
				for pn in c_tablet.pns:
					data_point[pn] = c_tablet.pns[pn]

				for gn in c_tablet.gns:
					data_point[gn] = c_tablet.gns[gn]

				data_point['period'] = c_tablet.period
				data.append(data_point)

				labels.append(c_tablet.year)
		else:
			if c_tablet.year == -999 or c_tablet.year == 0:
				continue

			for pn in c_tablet.pns:
				data_point[pn] = c_tablet.pns[pn]

			for gn in c_tablet.gns:
				data_point[gn] = c_tablet.gns[gn]

			data_point['period'] = c_tablet.period
			data.append(data_point)

			labels.append(c_tablet.year)

	return data, labels

def save_result(model, data, predictions, targets, le):
	'''
	Input:
		model - name of model being used
		predictions - predictions made by the model
		targets - answer set that the predictions were compared against
		le - label encoder to help inverse transform encoded labels
		
	Writes out the predictions compared against the targets.
	'''
	filename = 'prediction_output/' + model + '_predictions.txt'
	try:
		os.makedirs("prediction_output")
	except:
		pass
	output = open(filename, "w+")
	output.write(model + ' Predictions:\n')
	for i in range(0, len(targets)):
		pr = le.inverse_transform(predictions[i])
		ac = le.inverse_transform(targets[i])
		if pr != ac:
			output.write("Predicted: " + str(pr) + " Actual: " + str(ac) + "\n")
	output.close()

def plot_accuracies(graph_acc):
	graph_data = []
	for i in range(0, len(graph_acc)):
		graph_data.append(np.concatenate(graph_acc[i], 0))

	plt.figure(0)
	plt.boxplot(graph_data, 0, '')
	plt.xticks([1, 2, 3], ['MNB', 'Perc', 'SGD'])

	plt.show()

def plot_tablets_year(tablets):
	years = {}

	for tablet in tablets:
		c_tablet = tablets[tablet]
		years[c_tablet.year] = 1 if c_tablet.year not in years else years[c_tablet.year] + 1

	fig = plt.figure()
	ax = plt.subplot(111)
	ticks = []
	values = []
	for year in sorted(years):
		if year != 0 and year != "0":
			values.append(years[year])
			ticks.append(year)

	width = .8
	ax.set_title("Number of Tablets Per Year")
	ax.set_xlabel("Normalized Year Name")
	ax.set_ylabel("Number of Tablets")
	ax.bar(range(len(years) - 1), values)
	ax.set_xticks(np.arange(len(years) - 1) + width/2)
	ax.set_xticklabels(ticks, rotation = 90)
	ax.set_ylim([0, 350])
	plt.show()
	print sum(values)

def plot_pns_year(tablets):
	years = OrderedDict()
	pns_observed = []

	for tablet in tablets:
		c_tablet = tablets[tablet]
		
		if c_tablet.year not in years:
			years[c_tablet.year] = 0

		for pn in c_tablet.pns:
			if pn not in pns_observed:
				years[c_tablet.year] += 1
				pns_observed.append(pn)

	fig = plt.figure()
	ax = plt.subplot(111)
	ticks = []
	values = []
	for year in sorted(years):
		if year != 0 and year != "0":
			values.append(years[year])
			ticks.append(year)

	width = .8
	ax.set_title("Number of Unique Names Per Year")
	ax.set_xlabel("Normalized Year Name")
	ax.set_ylabel("Number of Unique Names")
	ax.bar(range(len(years) - 1), values)
	ax.set_xticks(np.arange(len(years) - 1) + width/2)
	ax.set_xticklabels(ticks, rotation = 90)
	ax.set_ylim([0, 175])
	plt.show()

	

def train_model(model, training_data, training_targets):
	'''
	Input:
		model - model to be trained
		training_data - training set to train the model on
		training_targets - correct labels of each data point
		
	This simply fits the model to the provided training set.
	'''
	model.fit(training_data, training_targets)

def make_predictions(model, data, targets, le, name):
	'''
	Input:
		model - model to use to predict
		data - data to predict on
		targets - actual answer set to compare against predictions
		le - label encoder to normalize/transform labels
		
	Perform predictions using the provided model, over the data, and determine the
	accuracy of our model's predictions. It will also save out the results for further 
	observations.

	Returns:
		acc - the accuracy of the predictions that were made
	'''
	y_pred = model.predict(data)

	inc_total = (targets != y_pred).sum()
	acc = (1 - float(inc_total) / float(data.shape[0])) * 100

	#print "Total: %6i Incorrect: %6i" % (data.shape[0], inc_total)
	#print "Accuracy: %.2f%%" % (acc)
	save_result(name, data, y_pred, targets, le)

	return acc

####################
####################

# Build tablet dataset
tablets = collect_pns_gns()
collect_tablet_years(tablets)

# Parse tablets into datapoints
uk_data, uk_labels = build_data(tablets, True)
data, labels = build_data(tablets, False)

v = DictVectorizer()
vectorizer = make_pipeline(v, TfidfTransformer())

data = vectorizer.fit_transform(data)
uk_data = vectorizer.transform(uk_data)

# Encode/normalize labels
le = LabelEncoder()
le.fit(labels + uk_labels)
targets = le.transform(labels)
uk_targets = le.transform(uk_labels)
print le.classes_

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
		validation_acc.append(make_predictions(c_model, test_samples, test_targets, le, models[k][0]))
		make_predictions(c_model, uk_data, uk_targets, le, models[k][0])

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
#plot_accuracies(graph_acc)
plot_tablets_year(tablets)
#plot_pns_year(tablets)







