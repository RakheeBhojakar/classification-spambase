'''
Author - Nandadeep Davuluru
Email - davuluru@pdx.edu
Course - CS 545 Machine learning
Instructor - Anthony Rhodes
Task - Naive Bayes and logistic regression
'''
import numpy, random, pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plotter
from sklearn.metrics import confusion_matrix
import random, itertools


# roughly 60/40 spam not spam split will have the following stats
# spam = 1813, 725
# not spam = 2788, 1672
pd.set_option('expand_frame_repr', False)
#### Confusion Matrix helper functions - sklearn metrics
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plotter.get_cmap('jet') or plotter.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """


    accuracy = numpy.trace(cm) / float(numpy.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plotter.get_cmap('Blues')

    plotter.figure(figsize=(10,10))
    plotter.imshow(cm, interpolation='nearest', cmap=cmap)
    plotter.title(title)
    plotter.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plotter.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plotter.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plotter.tight_layout()
    plotter.ylabel('True label')
    plotter.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plotter.show()
    # plotter.savefig('initial.png')

class spambase(object):

	def __init__(self):
		# OOP style for garbage collection, object handling
		self.data = pd.read_csv('spambase.data', header=None)
		self.data.rename(columns = {57 : 'is_spam'}, inplace=True)
		self.statisticsSpam = dict()
		self.statisticsNotSpam = dict()

	# using numpy instead of this function
	def productOfList(self, probabilities):
		product = 0.0
		for i in probabilities:
			product *= i
		return product

	# split train test data
	def generateData(self):
		# separate dataset into spam and not spam 
		spam = self.data[self.data['is_spam'] == 1]
		notSpam = self.data[self.data['is_spam'] == 0]
		# split using sklearn test train split - random 0.6 for training and 0.4 for testing
		spam_train, spam_test = train_test_split(spam, train_size=0.6)
		notSpam_train, notSpam_test = train_test_split(notSpam, train_size=0.6)
		# divide the data from labels for easier fitting
		trainData = notSpam_train.append(spam_train)
		trainLabels = trainData.pop('is_spam')
		testData = notSpam_test.append(spam_test)
		testLabels = testData.pop('is_spam')
		# return them
		return trainData, trainLabels, testData.values.tolist(), testLabels.values.tolist()

	def generateStatistics(self, train, trainLabels):

		''' builds the probabilistic model. Need to make it less verbose. '''

		# converting to list for better manipulation
		train = train.values.tolist()
		trainLabels = trainLabels.values.tolist()
		spam, notSpam = [], []
		for index, (row, label) in enumerate(zip(train, trainLabels)):
			if label == 1:
				spam.append(train[index])
			else:
				notSpam.append(train[index])

		# converting to pandas for better statistic calculation
		spamdf = pd.DataFrame(spam)
		notSpamdf = pd.DataFrame(notSpam)
		# print(notSpamdf)
		# find mean, std for each feature given its class
		for column in spamdf.columns:
			strColumn = str(column)
			meanOfCol = spamdf[column].mean()
			stdOfCol = spamdf[column].std()
			# to avoid divide by zero error in gaussian naive bayes
			if meanOfCol == 0:
				meanOfCol = 0.00001
			if stdOfCol == 0:
				stdOfCol = 0.000001
			self.statisticsSpam[strColumn] = [meanOfCol, stdOfCol]

		# find mean, std for each feature given its class
		for column in notSpamdf.columns:
			strColumn = str(column)
			meanOfCol = notSpamdf[column].mean()
			stdOfCol = notSpamdf[column].std()
			# to avoid divide by zero error in gaussian naive bayes
			if meanOfCol == 0:
				meanOfCol = 0.00001
			if stdOfCol == 0:
				stdOfCol = 0.000001
			self.statisticsNotSpam[strColumn] = [meanOfCol, stdOfCol]

		# just debugging
		# pprint(sorted(self.statisticsSpam.items(), key=lambda s: s[0]))
		# pprint(self.statisticsSpam[str(1)][0])
		return self.statisticsNotSpam, self.statisticsSpam
			

	# computes prior values for the two classes - count spam/not spam instances
	def computePrior(self, labels):
		priorSpam, priorNSpam = 0.0, 0.0
		priorSpam = (labels == 1).astype(int).sum() / len(labels)
		priorNSpam = (labels == 0).astype(int).sum() / len(labels)
		# 0.6, 0.4 approx. if you print them out. 
		return priorSpam, priorNSpam

	def gaussianProbability(self, feature, mean, std):
		# computes gaussian probability - given in the slides
		first_term = (1 / numpy.sqrt(2 * numpy.pi * numpy.power(std, 2)))
		second_term = numpy.exp(-numpy.power((feature - mean), 2) / 2 * numpy.power(std, 2))
		return numpy.log(first_term * second_term)

	# predicting test instances using gaussian probability
	def predict(self, testInstance, testLabel, priors):
		# appending the prior terms first. 
		positiveProbabilities = list()
		positiveProbabilities.append(numpy.log(priors[0]))
		negativeProbabilites = list()
		negativeProbabilites.append(numpy.log(priors[1]))

		for i, X in enumerate(testInstance):
			# converting to string because dict keys are str while enumerate returns an int
			i = str(i)
			# calc positive class probability using the dict built above
			mean = self.statisticsSpam[i][0]
			std = self.statisticsSpam[i][1]
			posp = self.gaussianProbability(X, mean, std)
			positiveProbabilities.append(posp)

			# calc negative class probability using the dict built above
			mean = self.statisticsNotSpam[i][0]
			std = self.statisticsNotSpam[i][1]
			negp = self.gaussianProbability(X, mean, std)
			negativeProbabilites.append(negp)
		positivePred = sum(numpy.array(positiveProbabilities))
		negativePred = sum(numpy.array(negativeProbabilites))
		if positivePred > negativePred:
			return 1
		else:
			return 0

	# need to write accuracy function explicitly. Although its covered in the confusion matrix method.

def main():

	classifier = spambase()
	train, trainLabels, test, testLabels = classifier.generateData()
	stats, priors = classifier.generateStatistics(train, trainLabels), classifier.computePrior(trainLabels)
	predictions = [classifier.predict(instance, label, priors) for instance, label in zip(test, testLabels)]

	# prints the accuracy, misclassified points, confusion matrix of the two classes. 
	plot_confusion_matrix(confusion_matrix(predictions, testLabels))


if __name__ == '__main__':
	main()