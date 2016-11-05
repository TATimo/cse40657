# Rosalyn Tan
# Bigram (1st order) HMM

tag_bigrams = {} # dictionary of dictionaries
tag_bigrams_prob = {} # dictionary of dictionaries of probabilities
tagWordCount = {} # stores mapping of each tag to the frequency of each word that appears with that tag
tagWordProb = {} # stores mapping of each tag to the probability of each workd that appears with that tag
viterbi = {} # stores weights
pointers = {} # stores best path back

def setup():
	tag_bigrams[0] = {} # unigrams--do I need this?
	tag_bigrams[1] = {} # bigrams
	tag_bigrams_prob[0] = {}
	tag_bigrams_prob[1] = {}

def train():
	get_counts()
	calc_probs()

def get_counts():
	for line in open("./hw3-data/train.txt"):
		bg_1 = '<s>'
		bg_2 = '<s>'
		words = line.split()
		for word in words:
			pair = word.split('/')
			bg_1 = bg_2
			bg_2 = pair[1]
			if pair[1] in tagWordCount:
				if pair[0] in tagWordCount[pair[1]]:
					tagWordCount[pair[1]][pair[0]] += 1
				else:
					tagWordCount[pair[1]][pair[0]] = 1
			else:
				tagWordCount[pair[1]] = {}
				tagWordCount[pair[1]][pair[0]] = 1
			if pair[1] in tag_bigrams[0]:
				tag_bigrams[0][pair[1]] += 1
			else:
				tag_bigrams[0][pair[1]] = 1
			if bg_1 in tag_bigrams[1]:
				if bg_2 in tag_bigrams[1][bg_1]:
					tag_bigrams[1][bg_1][bg_2] += 1
				else:
					tag_bigrams[1][bg_1][bg_2] = 1
			else:
				tag_bigrams[1][bg_1] = {}
				tag_bigrams[1][bg_1][bg_2] = 1
		if bg_1 in tag_bigrams[1]:
			if '</s>' in tag_bigrams[1][bg_1]:
				tag_bigrams[1][bg_1]['</s>'] += 1
			else:
				tag_bigrams[1][bg_1]['</s>'] = 1
		else:
			tag_bigrams[1][bg_1] = {}
			tag_bigrams[1][bg_1]['</s>'] = 1

def calc_probs():
	sum_count = 0
	for tag in tagWordCount.keys():
		for word in tagWordCount[tag].keys():
			sum_count += tagWordCount[tag][word]
		for word in tagWordCount[tag].keys():
			if tag not in tagWordProb:
				tagWordProb[tag] = {}
			tagWordProb[tag][word] = tagWordCount[tag][word] / sum_count
		sum_count = 0
	
#	sum_count = 0
	for tag in tag_bigrams[0].keys():
		sum_count += tag_bigrams[0][tag]
	for tag in tag_bigrams[0].keys():
		tag_bigrams_prob[0][tag] = tag_bigrams[0][tag] / sum_count

	sum_count = 0
	for tag in tag_bigrams[1].keys():
		for tag_2 in tag_bigrams[1][tag].keys():
			sum_count += tag_bigrams[1][tag][tag_2]
		for tag_2 in tag_bigrams[1][tag].keys():
			if tag not in tag_bigrams_prob[1]:
				tag_bigrams_prob[1][tag] = {}
			tag_bigrams_prob[1][tag][tag_2] = tag_bigrams[1][tag][tag_2] / sum_count
		sum_count = 0
	
def decode():
	pass

def print_matrix():
	print("P(t'|t)")
	print("t'/t\t<s>\t\tN\t\tF\t\tE\t\tD\t\tA\t\tR")
	print("N\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['N'], tag_bigrams_prob[1]['N']['N'], tag_bigrams_prob[1]['F']['N'], tag_bigrams_prob[1]['E']['N'], tag_bigrams_prob[1]['D']['N'], tag_bigrams_prob[1]['A']['N'], tag_bigrams_prob[1]['R']['N']))
	print("F\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['F'], tag_bigrams_prob[1]['N']['F'], tag_bigrams_prob[1]['F']['F'], tag_bigrams_prob[1]['E']['F'], tag_bigrams_prob[1]['D']['F'], tag_bigrams_prob[1]['A']['F'], tag_bigrams_prob[1]['R']['F']))
	print("E\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['E'], tag_bigrams_prob[1]['N']['E'], tag_bigrams_prob[1]['F']['E'], tag_bigrams_prob[1]['E']['E'], tag_bigrams_prob[1]['D']['E'], tag_bigrams_prob[1]['A']['E'], tag_bigrams_prob[1]['R']['E']))
	print("D\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['D'], tag_bigrams_prob[1]['N']['D'], tag_bigrams_prob[1]['F']['D'], tag_bigrams_prob[1]['E']['D'], tag_bigrams_prob[1]['D']['D'], tag_bigrams_prob[1]['A']['D'], tag_bigrams_prob[1]['R']['D']))
	print("A\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['A'], tag_bigrams_prob[1]['N']['A'], tag_bigrams_prob[1]['F']['A'], tag_bigrams_prob[1]['E']['A'], tag_bigrams_prob[1]['D']['A'], tag_bigrams_prob[1]['A']['A'], tag_bigrams_prob[1]['R']['A']))
	print("R\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['R'], tag_bigrams_prob[1]['N']['R'], tag_bigrams_prob[1]['F']['R'], tag_bigrams_prob[1]['E']['R'], tag_bigrams_prob[1]['D']['R'], tag_bigrams_prob[1]['A']['R'], tag_bigrams_prob[1]['R']['R']))
	print("</s>\t%f\t%f\t%f\t%f\t%f\t%f\t%f" % (tag_bigrams_prob[1]['<s>']['</s>'], tag_bigrams_prob[1]['N']['</s>'], tag_bigrams_prob[1]['F']['</s>'], tag_bigrams_prob[1]['E']['</s>'], tag_bigrams_prob[1]['D']['</s>'], tag_bigrams_prob[1]['A']['</s>'], tag_bigrams_prob[1]['R']['</s>']))

def p_you_t():
	print("P(you|N) = %f" % tagWordProb['N']['you'])
	print("P(you|F) = %f" % tagWordProb['F']['you'])
	print("P(you|E) = %f" % tagWordProb['E']['you'])
	print("P(you|D) = %f" % tagWordProb['D']['you'])
	print("P(you|A) = %f" % tagWordProb['A']['you'])
	print("P(you|R) = %f" % tagWordProb['R']['you'])

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
	setup()
	train()

	print_matrix()
	p_you_t()
#	test_acc()
#	second_line_tags()
