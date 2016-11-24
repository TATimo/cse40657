# Rosalyn Tan
# Extended bigram HMM model

import math

tag_bigrams = {} # dictionary of dictionaries
tag_bigrams_prob = {} # dictionary of dictionaries of probabilities
tagWordCount = {} # stores mapping of each tag to the frequency of each word that appears with that tag
tagWordProb = {} # stores mapping of each tag to the probability of each word that appears with that tag
viterbi = {} # stores weights
pointer = {} # stores best path back
test_acc = {} # stores accuracy

bigramTagWordCount = {} # stores mapping of each bigram tag to the frequency of each word that appears with that tag
bigramTagWordProb = {} # stores mapping of each bigram tag to the probability of each word that appears with that tag

def setup():
	tag_bigrams[0] = {} # unigrams
	tag_bigrams[1] = {} # bigrams
	tag_bigrams_prob[0] = {}
	tag_bigrams_prob[1] = {}
	test_acc[0] = 0 # number correct
	test_acc[1] = 0 # total number

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
			bg = bg_1 + bg_2
			if pair[1] in tagWordCount:
				if pair[0] in tagWordCount[pair[1]]:
					tagWordCount[pair[1]][pair[0]] += 1
				else:
					tagWordCount[pair[1]][pair[0]] = 1
			else:
				tagWordCount[pair[1]] = {}
				tagWordCount[pair[1]][pair[0]] = 1
			if bg in bigramTagWordCount:
				if pair[0] in bigramTagWordCount[bg]:
					bigramTagWordCount[bg][pair[0]] += 1
				else:
					bigramTagWordCount[bg][pair[0]] = 1
			else:
				bigramTagWordCount[bg] = {}
				bigramTagWordCount[bg][pair[0]] = 1
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
		if bg_2 in tag_bigrams[1]:
			if '</s>' in tag_bigrams[1][bg_2]:
				tag_bigrams[1][bg_2]['</s>'] += 1
			else:
				tag_bigrams[1][bg_2]['</s>'] = 1
		else:
			tag_bigrams[1][bg_2] = {}
			tag_bigrams[1][bg_2]['</s>'] = 1

def calc_probs():
	sum_count = 0
	for tag in tag_bigrams[0].keys():
		sum_count += tag_bigrams[0][tag]
	for tag in tag_bigrams[0].keys():
		tag_bigrams_prob[0][tag] = tag_bigrams[0][tag] / sum_count

	# smoothing
	for tag in tag_bigrams[0]:
		tagWordCount[tag]['unk'] = tag_bigrams_prob[0][tag] * 10

	sum_count = 0
	for bg in bigramTagWordCount.keys():
		for word in bigramTagWordCount[bg].keys():
			sum_count += bigramTagWordCount[bg][word]
		for word in bigramTagWordCount[bg].keys():
			if bg not in bigramTagWordProb:
				bigramTagWordProb[bg] = {}
			bigramTagWordProb[bg][word] = bigramTagWordCount[bg][word] / sum_count
		sum_count = 0

	for tag in tagWordCount.keys():
		for word in tagWordCount[tag].keys():
			sum_count += tagWordCount[tag][word]
		for word in tagWordCount[tag].keys():
			if tag not in tagWordProb:
				tagWordProb[tag] = {}
			tagWordProb[tag][word] = tagWordCount[tag][word] / sum_count
		sum_count = 0
	
	for tag in tag_bigrams[1].keys():
		for tag_2 in tag_bigrams[1][tag].keys():
			sum_count += tag_bigrams[1][tag][tag_2]
		for tag_2 in tag_bigrams[1][tag].keys():
			if tag not in tag_bigrams_prob[1]:
				tag_bigrams_prob[1][tag] = {}
			tag_bigrams_prob[1][tag][tag_2] = tag_bigrams[1][tag][tag_2] / sum_count
		sum_count = 0
	
def decode(line):
	count = 0
	viterbi[0] = {}
	viterbi[0]['<s>'] = 1
	words = line.split()
	for word in words:
		# initialize viterbi states
		count += 1
		viterbi[count] = {}
		for tag in tag_bigrams[0].keys():
			viterbi[count][tag] = 0
	count += 1
	viterbi[count] = {}
	viterbi[count]['</s>'] = 0

	# viterbi algorithm
	for x in range(1, len(words)+2):
		for state in viterbi[x].keys():
			for prev_state in viterbi[x-1].keys():
				if x == len(words) + 1:
					prob = 1
				else:
					pair = words[x-1].split('/')
					if prev_state + state not in bigramTagWordProb or pair[0] not in bigramTagWordProb[prev_state + state]:
						if pair[0] not in tagWordProb[state]:
							prob = tagWordProb[state]['unk']
						else:
							prob = tagWordProb[state][pair[0]]
					else:
						prob = bigramTagWordProb[prev_state + state][pair[0]]
				if viterbi[x-1][prev_state] * tag_bigrams_prob[1][prev_state][state] * prob > viterbi[x][state]:
					viterbi[x][state] = viterbi[x-1][prev_state] * tag_bigrams_prob[1][prev_state][state] * prob
					if x not in pointer.keys():
						pointer[x] = {}
					pointer[x][state] = prev_state

	# trace back pointers
	curr_state = '</s>'
	second_line = []
	for i in range(len(words)+1, 1, -1):
		pair = words[i-2].split('/')
		if pointer[i][curr_state] == pair[1]:
			test_acc[0] += 1
		test_acc[1] += 1
		second_line.append(pair[0] + '/' + pointer[i][curr_state])
		curr_state = pointer[i][curr_state]
# uncomment for second line tags
#	for i in range(len(second_line)-1, -1, -1):
#		print(second_line[i])

def guess():	
	for line in open("./hw3-data/test.txt"):
		decode(line)
	print("Accuracy on test: %f" % (test_acc[0] / test_acc[1]))

# Computes tags for second line of test set
def second_line_tags():
	lineCount = 0
	for line in open("./hw3-data/test.txt"):
		lineCount += 1
		if lineCount > 2:
			break
		elif lineCount < 2:
			continue
		decode(line)

if __name__ == '__main__':
	setup()
	train()

	guess()
#	second_line_tags()
