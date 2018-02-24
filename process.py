import numpy, random, pandas as pd
from sklearn.cross_validation import train_test_split
from addict import Dict

# roughly 60/40 spam not spam split will have the following stats
# spam = 1813, 725
# not spam = 2788, 1672

class spambase(object):

	def __init__(self):

		self.data = pd.read_csv('spambase.data', header=None)
		self.data.rename(columns = {57 : 'is_spam'}, inplace=True)
		self.statistics = dict()

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


	# create a df with 57 rows, and 2 columns where the 2 columns will have [mean, std]
	# data structure is as follows:

	# 0 : [m0, std0]
	# 1 : [m1, std1]
	# ...

	def generateStatistics(self, train):
		
		for column in train.columns:
			strColumn = str(column)
			meanOfCol = train[column].mean()
			stdOfCol = train[column].std()

			# to avoid divide by zero in naive bayes..
			if stdOfCol == 0:
				stdOfCol = 0.000001

			self.statistics[strColumn] = [meanOfCol, stdOfCol]
			
		return self.statistics

	def computePrior(self, labels):
		priorSpam, priorNSpam = 0.0, 0.0
		priorSpam = (labels == 1).astype(int).sum() / len(labels)
		priorNSpam = (labels == 0).astype(int).sum() / len(labels)
		# 0.6, 0.4 approx. 
		return priorSpam, priorNSpam


def main():
	classifier = spambase()
	train, trainLabels, test, testLabels = classifier.generateData()
	stats = classifier.generateStatistics(train)
	trainPrior = classifier.computePrior(trainLabels)
	# print(stats)

if __name__ == '__main__':
	main()


