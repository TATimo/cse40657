# Rosalyn Tan
# Baseline (0th order) HMM

wordTagCount = {} # stores mapping of each word to the frequency of each tag assigned to that word
tagCount = {} # stores frequency of each tag, use for unknown words

def train():
	for line in open("./hw3-data/train.txt"):
		words = line.split()
		for word in words:
			pair = word.split('/')
			if pair[0] in wordTagCount:
				if pair[1] in wordTagCount[pair[0]]:
					wordTagCount[pair[0]][pair[1]] += 1
				else:
					wordTagCount[pair[0]][pair[1]] = 1
			else:
				wordTagCount[pair[0]] = {}
				wordTagCount[pair[0]][pair[1]] = 1
			if pair[1] in tagCount:
				tagCount[pair[1]] += 1
			else:
				tagCount[pair[1]] = 1

def guess(word):
	maxCount = 0
	sumCount = 0
	maxTag = ""

	if word not in wordTagCount: # unknown words
		for key in tagCount.keys():
			if maxCount < tagCount[key]:
				maxCount = tagCount[key]
				maxTag = key
		return maxTag
	else:
		for key in wordTagCount[word].keys():
			sumCount += wordTagCount[word][key]
			if maxCount < wordTagCount[word][key]:
				maxCount = wordTagCount[word][key]
				maxTag = key
		maxProb = maxCount / sumCount
		return maxTag

# Caluclates P(t|you) for all t
def p_t_you():
	sumTag = 0
	for tag in wordTagCount['you']:
		sumTag += wordTagCount['you'][tag]
	for tag in wordTagCount['you']:
		print("P(%s|you) = %f" % (tag, (wordTagCount['you'][tag]/sumTag)))

# Calculates accuracy on test set
def test_acc():
	numCorrect = 0
	total = 0
	for line in open("./hw3-data/test.txt"):
		words = line.split()
		for word in words:
			pair = word.split('/')
			tag = guess(pair[0])
			if tag == pair[1]:
				numCorrect += 1
			total += 1
	print("Accuracy on test: %f" % (numCorrect / total))

# Computes tags for second line of test set
def second_line_tags():
	lineCount = 0
	for line in open("./hw3-data/test.txt"):
		lineCount += 1
		if lineCount > 2:
			break
		elif lineCount < 2:
			continue
		words = line.split()
		for word in words:
			pair = word.split('/')
			tag = guess(pair[0])
			print(pair[0] + '/'+ tag)

if __name__ == '__main__':
	train()

#	p_t_you()
	test_acc()
#	second_line_tags()
