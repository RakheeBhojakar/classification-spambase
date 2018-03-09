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
import itertools
import pdb


# roughly 60/40 spam not spam split will have the following stats
# spam = 1813, 725
# not spam = 2788, 1672

# printing the entire data frame
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
		# OOP style
		self.data = pd.read_csv('spambase.data', header=None)
		self.data.rename(columns = {57 : 'is_spam'}, inplace=True)
		self.statisticsSpam = dict()
		self.statisticsNotSpam = dict()

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
		# pdb.set_trace()
		return self.statisticsNotSpam, self.statisticsSpam
			
	def alternativeStatistics(self, train, trainLabels):
		'''
		This function was written for sanity check - to make sure model is built properly. 
		Fundamentally it is the same as the generateStatistics(). Since I wrote it anyway, Im decided to leave as is. 
		'''
		spamStats, notSpamStats = {}, {}
		train = train.values.tolist()
		trainLabels = trainLabels.values.tolist()

		spam, notspam = list(), list()

		for data, label in zip(train, trainLabels):
			if label == 0:
				notspam.append(data)
			else:
				spam.append(data)
		
		numpySpam = numpy.array(spam).T
		numpyNSpam = numpy.array(notspam).T
		for idx, row in enumerate(numpySpam):
			spamStats[str(idx)] = [numpy.mean(row), numpy.std(row)]
		# print(numpySpam.shape, numpyNSpam.shape)
		for idx, row in enumerate(numpyNSpam):
			notSpamStats[str(idx)] = [numpy.mean(row), numpy.std(row)]
		pdb.set_trace()

	# computes prior values for the two classes - count spam/not spam instances
	def computePrior(self, labels):
		# priorSpam, priorNSpam = 0.0, 0.0
		priorSpam = (labels == 1).astype(int).sum() / len(labels)
		priorNSpam = (labels == 0).astype(int).sum() / len(labels)
		# 0.606, 0.393 approx. if you print them out. 
		return priorSpam, priorNSpam

	def gaussianProbability(self, feature, mean, std):
		# computes gaussian probability - given in the slides
		first_term = (1 / numpy.sqrt(2 * numpy.pi * numpy.power(std, 2)))
		second_term = numpy.exp(-numpy.power((feature - mean), 2) / 2 * numpy.power(std, 2))
		prob = first_term * second_term
		if prob == 0:
			prob = 0.0001
		return numpy.log(prob)

	# predicting test instances using gaussian probability
	def predict(self, testInstance, testLabel, priors):
		# appending the prior terms first. 
		positiveProbabilities = list()
		positiveProbabilities.append(priors[0])
		negativeProbabilites = list()
		negativeProbabilites.append(priors[1])

		for i, X in enumerate(testInstance):
			# converting to string because dict keys are str while enumerate returns an int
			i = str(i)
			# calc positive class probability using the dict built above
			mean, std  = self.statisticsSpam[i]
			posp = self.gaussianProbability(X, mean, std)
			if numpy.isinf(posp):
				posp = 0.000001
			positiveProbabilities.append(posp)

			# calc negative class probability using the dict built above
			mean, std = self.statisticsNotSpam[i]
			negp = self.gaussianProbability(X, mean, std)
			if numpy.isinf(negp):
				negp = 0.000001
			negativeProbabilites.append(negp)
		positivePred = sum(positiveProbabilities)
		negativePred = sum(negativeProbabilites)
		if positivePred > negativePred:
			return 0
		else:
			return 1

	# need to write accuracy function explicitly. Although its covered in the confusion matrix method.

def main():

	classifier = spambase()
	train, trainLabels, test, testLabels = classifier.generateData()
	stats = classifier.generateStatistics(train, trainLabels)
	# stats2 = classifier.alternativeStatistics(train, trainLabels)
	# hard coded the priors. But if you actually call the function computePrior() it will give the same thing
	# priors = classifier.computePrior(trainLabels.values.tolist())
	priors = (0.606, 0.39)

	predictions = [classifier.predict(instance, label, priors) for instance, label in zip(test, testLabels)]
	
	# print("predicted spam - {} predicted nspam - {}".format(predictions.count(1), predictions.count(0)))
	# print("actual spam - {} actual nspam - {}".format(testLabels.count(1), testLabels.count(0)))
	# prints the accuracy, misclassified points, confusion matrix of the two classes.
	# cm = confusion_matrix(predictions, testLabels)
	# print(cm)
	# pdb.set_trace() 
	plot_confusion_matrix(confusion_matrix(predictions, testLabels))


if __name__ == '__main__':
	main()