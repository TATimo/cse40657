# Rosalyn Tan
# Logistic Regression Classifier with Bigrams

import autograd.numpy as np
import autograd.scipy.misc as ag
import random
from autograd import grad

trainDocs = []
devDocs = []
testDocs = []
wordSet = set()
speakerSet = set()
wordDict = {}
speakerDict = {}

def readTrainDocs():
	f = open("./hw1-data/train", "r")
	line = f.readline()
	line = line.strip()
	while line != "":
		trainDocs.append(line)
		line = f.readline()
	f.close()

def readDevDocs():
	f = open("./hw1-data/dev", "r")
	line = f.readline()
	line = line.strip()
	while line != "":
		devDocs.append(line)
		line = f.readline()
	f.close()

def readTestDocs():
	f = open("./hw1-data/test", "r")
	line = f.readline()
	line = line.strip()
	while line != "":
		testDocs.append(line)
		line = f.readline()
	f.close()

def initializeLambdas():
	# initialize lambdas to 0
	for doc in trainDocs:
		words = doc.split()
		numWords = len(words)
		speakerSet.add(words[0])
		if len(words) > 1:
			for i in range(1, numWords-1):
				words.append(words[i] + words[i+1])
			words.append('s' + words[1])
			words.append(words[numWords - 1] + 'e')
		for word in words[1:]:
			wordSet.add(word)
	wordSet.add('<bias>')

	i = 0
	for word in wordSet:
		wordDict[word] = i
		i += 1

	i = 0
	for speaker in speakerSet:
		speakerDict[speaker] = i
		i += 1

	lambdaArray = np.zeros((len(speakerSet), len(wordSet)))
	return lambdaArray

def negLogProbSpeakerGivenDoc(model, doc, speaker):
	words = doc.split()
	numWords = len(words)

	if numWords > 1:
		for i in range(1, numWords-1):
			words.append(words[i] + words[i+1])
		words.append('s' + words[1])
		words.append(words[numWords - 1] + 'e')

	wordVec = np.zeros([len(list(wordDict.keys())), 1])

	for i in range(1, len(words)):
		if words[i] not in wordDict:
			pass
		else:
			wordVec[wordDict[words[i]]] += 1
	wordVec[wordDict['<bias>']] = 1
	
	z_d = np.dot(model, wordVec)

	s_k_d = z_d[speakerDict[speaker]]
	
	p_k_d = s_k_d - ag.logsumexp(z_d)
	
	return (p_k_d * -1)

def stochGradDescent(func, model, bigrams, speaker):
	g_logProb = grad(func)
	model -= .01 * g_logProb(model, bigrams, speaker)

def classify(model, doc):
	maxProb = 0

	words = doc.split()

	for speaker in list(speakerDict.keys()):
		neg_p_k_d = negLogProbSpeakerGivenDoc(model, doc, speaker)
		p_k_d = np.exp(neg_p_k_d * -1)
		if p_k_d > maxProb:
			maxProb = p_k_d
			guess = speaker

	if guess == words[0]:
		return 1
	else:
		return 0		

def computeProb(model, doc):
	for speaker in list(speakerDict.keys()):
		neg_p_k_d = negLogProbSpeakerGivenDoc(model, doc, speaker)
		p_k_d = np.exp(neg_p_k_d * -1)
		print("P(%s|d) = %f" % (speaker, p_k_d))

if __name__ == "__main__":
	readTrainDocs()
	readDevDocs()
	readTestDocs()

	sumLogProb = 0
	totalCorrect = 0

	lambdaArray = initializeLambdas()
	
	for j in range(1, 21):
		for i in range(0, len(trainDocs)):
			sumLogProb += negLogProbSpeakerGivenDoc(lambdaArray, trainDocs[i], trainDocs[i].split()[0])
			stochGradDescent(negLogProbSpeakerGivenDoc, lambdaArray, trainDocs[i], trainDocs[i].split()[0])
			if i % 1000 == 0:
				print("Line %d" % i)
		print("Negative log-probability of train after iteration %d: %f" % (j, sumLogProb))
		sumLogProb = 0
		for doc in devDocs:
			totalCorrect += classify(lambdaArray, doc)
		print("Accuracy on dev: %f" % (totalCorrect / len(devDocs)))
		totalCorrect = 0
		random.shuffle(trainDocs)
	computeProb(lambdaArray, devDocs[0])
	for doc in testDocs:
		totalCorrect += classify(lambdaArray, doc)
	print("Accuracy on test: %f" % (totalCorrect / len(testDocs)))
