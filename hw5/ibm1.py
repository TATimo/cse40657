# Rosalyn Tan
# IBM Model 1 -- word alignment for training

import math
import numpy as np
import random

training = []
lambdas = {} # lambda of f given e (f and e are in the same line)
inverse = {} # maps e to every f in the same line (for looking up lambdas)

def read():
	for line in open("./hw5-files/data-a/episode1.zh-en"):
		training.append(line)
		zh_en = line.split('\t')

		if len(zh_en) > 2: # sanity check, can probably remove this later
			print('More than one tab in data')

		zh_words = zh_en[0].split()
		en_words = zh_en[1].split()

		inverse['NULL'] = set()

		for zh in zh_words:
			if zh not in lambdas:
				lambdas[zh] = {}
			for en in en_words:
				if en not in inverse:
					inverse[en] = set()
				inverse[en].add(zh)
				lambdas[zh][en] = 0 # initialize all lambdas to 0
			inverse['NULL'].add(zh)
			lambdas[zh]['NULL'] = 0

def log_prob(sentence):
	zh_en = sentence.split('\t')

	zh_words = zh_en[0].split()
	en_words = zh_en[1].split()

	# forward algorithm
	forward = np.empty(len(zh_words) + 1)
	forward[0] = 1
	for j in range(1, len(zh_words) + 1):
		forward[j] = 0
		for i in range(1, len(en_words) + 1):
			forward[j] += forward[j-1] * (1/(len(en_words) + 1)) * t_f_e(zh_words[j-1], en_words[i-1])
		forward[j] += forward[j-1] * (1/(len(en_words) + 1)) * t_f_e(zh_words[j-1], 'NULL')

	return math.log((1/100)*forward[len(zh_words)])

def t_f_e(f, e):
	summation = 0
	for zh in inverse[e]:
		summation += math.exp(lambdas[zh][e])
	return (math.exp(lambdas[f][e]) / summation)

def log_prob_5():
	counter = 0

	for line in training:
		counter += 1
		if(counter > 5):
			break
		print('Line ' + str(counter) + ' log-probability: ' + str(log_prob(line)))

def sga(): # definitely need to verify ranges
	summation = 0
	for line in training:
		summation += log_prob(line)
	print('Iteration 0 log-probability sum: ' + str(summation)) # before training

	T = 10 # number of iterations
	for t in range(1, T+1):
		eta = 1/t
		log_like = 0
		random.shuffle(training)
		for line in training:
			zh_en = line.split('\t')
			zh_words = zh_en[0].split()
			en_words = zh_en[1].split()

			log_like += log_prob(line)
			for j in range(1, len(zh_words) + 1):
				summation = 0
				for l in range(1, len(en_words) + 1):
					summation += t_f_e(zh_words[j-1], en_words[l-1])
				z = t_f_e(zh_words[j-1], 'NULL') + summation
				for i in range(0, len(en_words) + 1):
					if i == 0:
						p = t_f_e(zh_words[j-1], 'NULL') / z
						lambdas[zh_words[j-1]]['NULL'] += (eta * p)
						for word in zh_words:
							lambdas[word]['NULL'] -= (eta * p * t_f_e(word, 'NULL'))
					else:
						p = t_f_e(zh_words[j-1], en_words[i-1]) / z
						lambdas[zh_words[j-1]][en_words[i-1]] += (eta * p)
						for word in zh_words:
							lambdas[word][en_words[i-1]] -= (eta * p * t_f_e(word, en_words[i-1]))

		print('Iteration ' + str(t) + ' log-probability sum: ' + str(log_like))

def report_word_pairs():
	print('t(绝地|jedi) = ' + str(t_f_e('绝地', 'jedi')))
	print('t(机械人|droid) = ' + str(t_f_e('机械人', 'droid')))
	print('t(原力|force) = ' + str(t_f_e('原力', 'force')))
	print('t(原虫|midi-chlorians) = ' + str(t_f_e('原虫', 'midi-chlorians')))
	print('t(你|yousa) = ' + str(t_f_e('你', 'yousa')))

if __name__ == '__main__':
	read()
#	log_prob_5()
	sga()
	report_word_pairs()
