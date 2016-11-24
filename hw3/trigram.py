# Rosalyn Tan
# Trigram (2nd order) HMM

tag_trigrams = {} # dictionary of dictionaries
tag_trigrams_prob = {} # dictionary of dictionaries of probabilities
tagWordCount = {} # stores mapping of each tag to the frequency of each word that appears with that tag
tagWordProb = {} # stores mapping of each tag to the probability of each workd that appears with that tag
viterbi = {} # stores weights
pointer = {} # stores best path back
test_acc = {} # stores accuracy

def setup():
	tag_trigrams[0] = {} # unigrams
	tag_trigrams[1] = {} # bigrams
	tag_trigrams[2] = {} # trigrams
	tag_trigrams_prob[0] = {}
	tag_trigrams_prob[1] = {}
	tag_trigrams_prob[2] = {}
	test_acc[0] = 0 # number correct
	test_acc[1] = 0 # total number

def train():
	get_counts()
	calc_probs()

def get_counts():
	for line in open("./hw3-data/train.txt"):
		bg_0 = '<s>'
		bg_1 = '<s>'
		bg_2 = '<s>'
		words = line.split()
		for word in words:
			pair = word.split('/')
			bg_0 = bg_1
			bg_1 = bg_2
			bg_2 = pair[1]
			bg = bg_0+bg_1
			if pair[1] in tagWordCount:
				if pair[0] in tagWordCount[pair[1]]:
					tagWordCount[pair[1]][pair[0]] += 1
				else:
					tagWordCount[pair[1]][pair[0]] = 1
			else:
				tagWordCount[pair[1]] = {}
				tagWordCount[pair[1]][pair[0]] = 1
			if pair[1] in tag_trigrams[0]:
				tag_trigrams[0][pair[1]] += 1
			else:
				tag_trigrams[0][pair[1]] = 1
			if bg_1 in tag_trigrams[1]:
				if bg_2 in tag_trigrams[1][bg_1]:
					tag_trigrams[1][bg_1][bg_2] += 1
				else:
					tag_trigrams[1][bg_1][bg_2] = 1
			else:
				tag_trigrams[1][bg_1] = {}
				tag_trigrams[1][bg_1][bg_2] = 1
			if bg in tag_trigrams[2]:
				if bg_2 in tag_trigrams[2][bg]:
					tag_trigrams[2][bg][bg_2] += 1
				else:
					tag_trigrams[2][bg][bg_2] = 1
			else:
				tag_trigrams[2][bg] = {}
				tag_trigrams[2][bg][bg_2] = 1
		if bg_2 in tag_trigrams[1]:
			if '</s>' in tag_trigrams[1][bg_2]:
				tag_trigrams[1][bg_2]['</s>'] += 1
			else:
				tag_trigrams[1][bg_2]['</s>'] = 1
		else:
			tag_trigrams[1][bg_2] = {}
			tag_trigrams[1][bg_2]['</s>'] = 1
		bg = bg_1 + bg_2
		if bg in tag_trigrams[2]:
			if '</s>' in tag_trigrams[2][bg]:
				tag_trigrams[2][bg]['</s>'] += 1
			else:
				tag_trigrams[2][bg]['</s>'] = 1
		else:
			tag_trigrams[2][bg] = {}
			tag_trigrams[2][bg]['</s>'] = 1

def calc_probs():
	sum_count = 0
	for tag in tag_trigrams[0].keys():
		sum_count += tag_trigrams[0][tag]
	for tag in tag_trigrams[0].keys():
		tag_trigrams_prob[0][tag] = tag_trigrams[0][tag] / sum_count

	# smoothing
	for tag in tag_trigrams[0]:
		tagWordCount[tag]['unk'] = tag_trigrams_prob[0][tag] * 10

	sum_count = 0
	for tag in tagWordCount.keys():
		for word in tagWordCount[tag].keys():
			sum_count += tagWordCount[tag][word]
		for word in tagWordCount[tag].keys():
			if tag not in tagWordProb:
				tagWordProb[tag] = {}
			tagWordProb[tag][word] = tagWordCount[tag][word] / sum_count
		sum_count = 0
	
	for tag in tag_trigrams[1].keys():
		for tag_2 in tag_trigrams[1][tag].keys():
			sum_count += tag_trigrams[1][tag][tag_2]
		for tag_2 in tag_trigrams[1][tag].keys():
			if tag not in tag_trigrams_prob[1]:
				tag_trigrams_prob[1][tag] = {}
			tag_trigrams_prob[1][tag][tag_2] = tag_trigrams[1][tag][tag_2] / sum_count
		sum_count = 0
	
	for bg in tag_trigrams[2].keys():
		for tag in tag_trigrams[2][bg].keys():
			sum_count += tag_trigrams[2][bg][tag]
		for tag in tag_trigrams[2][bg].keys():
			if bg not in tag_trigrams_prob[2]:
				tag_trigrams_prob[2][bg] = {}
			tag_trigrams_prob[2][bg][tag] = tag_trigrams[2][bg][tag] / sum_count
		sum_count = 0
	
def decode(line):
	count = 0
	viterbi[0] = {}
	viterbi[0]['<s><s>'] = 1
	words = line.split()
	for word in words:
		# initialize viterbi states
		count += 1
		viterbi[count] = {}
		for tag in tag_trigrams[2].keys():
			viterbi[count][tag] = 0
	count += 1
	viterbi[count] = {}
	viterbi[count]['</s>'] = 0

	# viterbi algorithm
	for x in range(1, len(words)+2):
		for state in viterbi[x].keys():
			if state == '</s>':
				st = state
			elif len(state) == 4:
				st = state[3]
			elif len(state) == 6:
				continue
			else:
				st = state[1]
			for prev_state in viterbi[x-1].keys():
				if prev_state == '</s>':
					print('something is wrong')
					continue
				if x == len(words) + 1:
					prob = 1
				else:
					pair = words[x-1].split('/')
					if pair[0] not in tagWordProb[st]:
						prob = tagWordProb[st]['unk']
					else:
						prob = tagWordProb[st][pair[0]]
				# account for unknown trigrams
				if st not in tag_trigrams_prob[2][prev_state]:
					if len(prev_state) == 4:
						p_t = tag_trigrams_prob[1]['<s>'][prev_state[3]]*tag_trigrams_prob[1][prev_state[3]][st]
					elif len(prev_state) == 6:
						p_t = 1
					else:
						p_t = tag_trigrams_prob[1][prev_state[0]][prev_state[1]]*tag_trigrams_prob[1][prev_state[1]][st]
				else:
					p_t = tag_trigrams_prob[2][prev_state][st]
				if viterbi[x-1][prev_state] * p_t * prob > viterbi[x][state]:
					viterbi[x][state] = viterbi[x-1][prev_state] * p_t * prob
					if x not in pointer.keys():
						pointer[x] = {}
					pointer[x][state] = prev_state

	# trace back pointers
	curr_state = '</s>'
	second_line = []
	for i in range(len(words)+1, 1, -1):
		pair = words[i-2].split('/')
		if pointer[i][curr_state][1] == pair[1]:
			test_acc[0] += 1
#		else:
#			print("Guess: %s\t Actual: %s" % (pointer[i][curr_state][1], pair[1]))
		test_acc[1] += 1
		second_line.append(pair[0] + '/' + pointer[i][curr_state])
		curr_state = pointer[i][curr_state]
# uncomment for second line tags and log-prob
#	for i in range(len(second_line)-1, -1, -1):
#		print(second_line[i])

def guess():
	counter = 0
	for line in open("./hw3-data/test.txt"):
		counter += 1
	#	if counter % 1000 == 0:
	#		print(counter)
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
