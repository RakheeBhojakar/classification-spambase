'''
Author - Nandadeep Davuluru
Email - davuluru@pdx.edu
Course - CS 545 Machine learning
Instructor - Anthony Rhodes
Task - Naive Bayes and logistic regression
'''
import numpy, random, pandas as pd
from sklearn.cross_validation import train_test_split
from pprint import pprint

# roughly 60/40 spam not spam split will have the following stats
# spam = 1813, 725
# not spam = 2788, 1672
pd.set_option('expand_frame_repr', False)
class spambase(object):

	def __init__(self):

		self.data = pd.read_csv('spambase.data', header=None)
		self.data.rename(columns = {57 : 'is_spam'}, inplace=True)
		self.statisticsSpam = dict()
		self.statisticsNotSpam = dict()
		# self.probabilities = dict()

	def generateData(self):

		# separate dataset into spam and not spam 
		spam = self.data[self.data['is_spam'] == 1]
		notSpam = self.data[self.data['is_spam'] == 0]
		# split using sklearn test train split - random 
		spam_train, spam_test = train_test_split(spam, train_size=0.6)
		notSpam_train, notSpam_test = train_test_split(notSpam, train_size=0.6)
		# divide the data from labels for easier fitting
		trainData = notSpam_train.append(spam_train)
		trainLabels = trainData.pop('is_spam')
		testData = notSpam_test.append(spam_test)
		testLabels = testData.pop('is_spam')
		# return them
		return trainData, trainLabels, testData, testLabels

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
			if stdOfCol == 0:
				stdOfCol = 0.000001
			self.statisticsSpam[strColumn] = [meanOfCol, stdOfCol]

		# find mean, std for each feature given its class
		for column in notSpamdf.columns:
			strColumn = str(column)
			meanOfCol = notSpamdf[column].mean()
			stdOfCol = notSpamdf[column].std()
			# to avoid divide by zero error in gaussian naive bayes
			if stdOfCol == 0:
				stdOfCol = 0.000001
			self.statisticsNotSpam[strColumn] = [meanOfCol, stdOfCol]

		# pprint(sorted(self.statisticsSpam.items(), key=lambda s: s[0]))
		# pprint(self.statisticsSpam[str(1)][0])
		return self.statisticsNotSpam, self.statisticsSpam
			


	def computePrior(self, labels):
		priorSpam, priorNSpam = 0.0, 0.0
		priorSpam = (labels == 1).astype(int).sum() / len(labels)
		priorNSpam = (labels == 0).astype(int).sum() / len(labels)
		# 0.6, 0.4 approx. if you print them out. 
		return priorSpam, priorNSpam

	def gaussianProbability(self, feature, mean, std):

		first_term = (1 / numpy.sqrt(2 * numpy.pi * numpy.power(std, 2)))
		second_term = numpy.exp(-numpy.power((feature - mean), 2) / 2 * numpy.power(std, 2))
		return numpy.log(first_term * second_term)

	def calcProbabilities(self, testdata, priors):
		# converting to lists for ease. cuz why not. 
		testdatalist = testdata.values.tolist()
		positiveProbabilities = list()
		negativeProbabilites = list()
		for row in testdatalist:
			for i, X in enumerate(row):
				# calc positive class probability using the dict built above
				i = str(i)
				# print(self.statisticsSpam[i])
				mean = self.statisticsSpam[i][0]
				std = self.statisticsSpam[i][1]
				posp = self.gaussianProbability(X, mean, std)
				positiveProbabilities.append(posp)

				# calc negative class probability using the dict built above
				mean = self.statisticsNotSpam[i][0]
				std = self.statisticsNotSpam[i][1]
				negp = self.gaussianProbability(X, mean, std)
				negativeProbabilites.append(negp)


def main():
	classifier = spambase()
	train, trainLabels, test, testLabels = classifier.generateData()
	stats = classifier.generateStatistics(train, trainLabels)
	priors = classifier.computePrior(trainLabels)
	classifier.calcProbabilities(test, priors)

if __name__ == '__main__':
	main()


