# Rosalyn Tan
# Naive Bayes Classifier

import math
import autograd.scipy.misc as ag

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
		# number of documents each speaker has
		if words[0] in speakerCount:
			speakerCount[words[0]] += 1
		else:
			speakerCount[words[0]] = 1
			speakerWordCount[words[0]] = {}
		# number of each word each speaker has
		for word in words[1:]:
			if word in speakerWordCount[words[0]]:
				speakerWordCount[words[0]][word] += 1
			else:
				speakerWordCount[words[0]][word] = 1
			uniqueWords.add(word)
					
	print("c(clinton) = %d" % speakerCount['clinton'])
	print("c(trump) = %d " % speakerCount['trump'])		

	print("c(clinton,country) = %d" % speakerWordCount['clinton']['country'])
	print("c(clinton,president) = %d" % speakerWordCount['clinton']['president'])
	print("c(trump,country) = %d" % speakerWordCount['trump']['country'])
	print("c(trump,president) = %d" % speakerWordCount['trump']['president'])

	print("\n")

def trainProbabilities():
	speakerSum = 0
	delta = .1 # unseen word types, set equal to 0 for training set probabilites

	for speaker in list(speakerCount.keys()):
		speakerSum = speakerSum + speakerCount[speaker]

	# probability that any document belongs to a speaker
	for speaker in list(speakerCount.keys()):
		speakerProb[speaker] = speakerCount[speaker] / speakerSum

	# probability of a word given a speaker
	for speaker in list(speakerWordCount.keys()):
		speakerWordProb[speaker] = {}
		wordSum = delta * (len(uniqueWords) + 1)
		for word in list(speakerWordCount[speaker].keys()):
			wordSum = wordSum + speakerWordCount[speaker][word]
		for word in list(speakerWordCount[speaker].keys()):
			speakerWordProb[speaker][word] = (speakerWordCount[speaker][word] + delta) / wordSum
		speakerWordProb[speaker]['unk'] = delta / wordSum

	print("p(clinton) = %f" % (speakerProb['clinton']))
	print("p(trump) = %f" % (speakerProb['trump']))

	print("p(country|clinton) = %f" % (speakerWordProb['clinton']['country']))
	print("p(president|clinton) = %f" % (speakerWordProb['clinton']['president']))
	print("p(country|trump) = %f" % (speakerWordProb['trump']['country']))
	print("p(president|trump) = %f" % (speakerWordProb['trump']['president']))
	print("\n")

def classifySpeakerGivenDoc(doc):
	words = doc.split()
	sumSpeakProb = 0
	logProbSpeakerAndDoc = []
	probSpeakerGivenDoc = {}
	logProbSpeakerGivenDoc = {}
	sumProb = 0
	maxProb = 0

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

# uncomment below to get probabilities per speaker for a document

#		print("probability of %s given document: %f" % (speaker, probSpeakerGivenDoc[speaker]))
#	print ("\n")
#	print("probability sum over all speakers: %f" % sumProb)

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

# uncomment to get probabilities per speaker for the first line of dev

#	print("probability of a speaker given the first line of dev:\n")
#	classifySpeakerGivenDoc(devDocs[0])

# uncomment to get accuracy rate on dev

#	for doc in devDocs:
#		numCorrect = numCorrect + classifySpeakerGivenDoc(doc)
#	print("Accuracy for dev is: %f" % (numCorrect / len(devDocs)))

	numCorrect = 0
	for doc in testDocs:
		numCorrect = numCorrect + classifySpeakerGivenDoc(doc)
	print("Accuracy for test is: %f" % (numCorrect / len(testDocs)))
