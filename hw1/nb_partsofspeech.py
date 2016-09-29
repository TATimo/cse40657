# Rosalyn Tan
# Naive Bayes Classifier with Parts of Speech

import math
import autograd.scipy.misc as ag
import nltk

trainDocs = []
devDocs = []
testDocs = []
speakerCount = {}
speakerProb = {}
speakerWordCount = {}
speakerWordProb = {}
uniqueWords = set()

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

def collectCounts():
	for doc in trainDocs:
		words = doc.split()
		if words[0] in speakerCount:
			speakerCount[words[0]] = speakerCount[words[0]] + 1
		else:
			speakerCount[words[0]] = 1
			speakerWordCount[words[0]] = {}
		for word in words[1:]:
			pos = nltk.pos_tag(word)
			if word in speakerWordCount[words[0]]:
				speakerWordCount[words[0]][word] = speakerWordCount[words[0]][word] + 1
			else:
				speakerWordCount[words[0]][word] = 1
			if pos[0][0] != pos[0][1] and pos[0][1] in speakerWordCount[words[0]]:
				speakerWordCount[words[0]][pos[0][1]] += 1
			elif pos[0][0] != pos[0][1] and pos[0][1] not in speakerWordCount[words[0]]:
				speakerWordCount[words[0]][pos[0][1]] = 1
			uniqueWords.add(word)
					
	print("clinton count: %d" % speakerCount['clinton'])
	print("trump count: %d " % speakerCount['trump'])		

	print("clinton,country count: %d" % speakerWordCount['clinton']['country'])
	print("clinton,president count: %d" % speakerWordCount['clinton']['president'])
	print("trump,country count: %d" % speakerWordCount['trump']['country'])
	print("trump,president count: %d" % speakerWordCount['trump']['president'])

	print("\n")

def trainProbabilities():
	speakerSum = 0
	delta = .1 # unseen word types, set equal to 0 for just training set probabilites

	for speaker in list(speakerCount.keys()):
		speakerSum = speakerSum + speakerCount[speaker]

	for speaker in list(speakerCount.keys()):
		speakerProb[speaker] = speakerCount[speaker] / speakerSum

	for speaker in list(speakerWordCount.keys()):
		speakerWordProb[speaker] = {}
		wordSum = delta * (len(uniqueWords) + 1)
		for word in list(speakerWordCount[speaker].keys()):
			wordSum = wordSum + speakerWordCount[speaker][word]
		for word in list(speakerWordCount[speaker].keys()):
			speakerWordProb[speaker][word] = (speakerWordCount[speaker][word] + delta) / wordSum
		speakerWordProb[speaker]['unk'] = delta / wordSum

	print("clinton probability: %f" % (speakerProb['clinton']))
	print("trump probability: %f" % (speakerProb['trump']))

	print("country given clinton probability: %f" % (speakerWordProb['clinton']['country']))
	print("president given probability: %f" % (speakerWordProb['clinton']['president']))
	print("country given trump probabilty: %f" % (speakerWordProb['trump']['country']))
	print("president given trump probability %f" % (speakerWordProb['trump']['president']))
	print("\n")

def classifySpeakerGivenDoc(doc):
	words = doc.split()
	sumSpeakProb = 0
	logProbSpeakerAndDoc = []
	probSpeakerGivenDoc = {}
	logProbSpeakerGivenDoc = {}
	sumProb = 0
	maxProb = 0

	numWords = len(words)
	
	# add parts of speech into test lines
	for i in range(1, numWords):
		pos = nltk.pos_tag(words[i])
		if pos[0][0] != pos[0][1]:
			words.append(pos[0][1])

	for speaker in list(speakerCount.keys()):
		sumLogWordProb = 0
		for word in words[1:]:
			if word not in speakerWordProb[speaker]:
				sumLogWordProb = sumLogWordProb + math.log(speakerWordProb[speaker]['unk'])
			else:
				sumLogWordProb = sumLogWordProb + math.log(speakerWordProb[speaker][word])
		logProbSpeakerAndDoc.append(sumLogWordProb + math.log(speakerProb[speaker]))

	for speaker in list(speakerCount.keys()):
		sumLogWordProb = 0
		for word in words[1:]:
			if word not in speakerWordProb[speaker]:
				sumLogWordProb = sumLogWordProb + math.log(speakerWordProb[speaker]['unk'])
			else:
				sumLogWordProb = sumLogWordProb + math.log(speakerWordProb[speaker][word])
		logProbSpeakerGivenDoc[speaker] = (sumLogWordProb + math.log(speakerProb[speaker])) - ag.logsumexp(logProbSpeakerAndDoc)
		probSpeakerGivenDoc[speaker] = math.exp(logProbSpeakerGivenDoc[speaker])

	for speaker in list(speakerCount.keys()):
		sumProb = sumProb + probSpeakerGivenDoc[speaker]
		if probSpeakerGivenDoc[speaker] > maxProb:
			maxProb = probSpeakerGivenDoc[speaker]
			guess = speaker

	if guess == words[0]:
		return 1
	else:
		return 0

if __name__ == '__main__':
	numCorrect = 0

	readTrainDocs()
	readDevDocs()
	readTestDocs()

	collectCounts()
	trainProbabilities()

# uncomment for accuracy on dev

#	for doc in devDocs:
#		numCorrect = numCorrect + classifySpeakerGivenDoc(doc)
#	print("Accuracy for dev is: %f" % (numCorrect / len(devDocs)))

	numCorrect = 0
	for doc in testDocs:
		numCorrect = numCorrect + classifySpeakerGivenDoc(doc)
	print("Accuracy for test is: %f" % (numCorrect / len(testDocs)))
