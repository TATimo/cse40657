# Rosalyn Tan
# IBM Model 1 -- word alignment for training

import math
import numpy as np
import random

training = []
testing = []
lambdas = {} # lambda of f given e (f and e are in the same line)
inverse = {} # maps e to every f in the same line (for looking up lambdas)
t_values = {} # maps e to f to t(f|e) value

def read():
	inverse['NULL'] = set()
	for line in open("./hw5-files/data-a/episode1.zh-en"):
		training.append(line)
		testing.append(line)
		zh_en = line.split('\t')

		zh_words = zh_en[0].split()
		en_words = zh_en[1].split()

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
		
	for e in inverse.keys():
		t_values[e] = {}
		t_f_e(e)

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
			forward[j] += forward[j-1] * (1/(len(en_words) + 1)) * t_values[en_words[i-1]][zh_words[j-1]]
		forward[j] += forward[j-1] * (1/(len(en_words) + 1)) * t_values['NULL'][zh_words[j-1]]

	return math.log((1/100)*forward[len(zh_words)])

def t_f_e(e):
	summation = 0
	for zh in inverse[e]:
		summation += math.exp(lambdas[zh][e])
	for zh in inverse[e]:
		t_values[e][zh] = math.exp(lambdas[zh][e]) / summation

def log_prob_5():
	counter = 0

	for line in training:
		counter += 1
		if(counter > 5):
			break
		print('Line ' + str(counter) + ' log-probability: ' + str(log_prob(line)))

def sga():
	summation = 0
	for line in training:
		summation += log_prob(line)
	print('Iteration 0 log-probability: ' + str(summation)) # before training

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
					summation += t_values[en_words[l-1]][zh_words[j-1]]
				z = t_values['NULL'][zh_words[j-1]] + summation
				t_f_e('NULL')
				for i in range(0, len(en_words) + 1):
					if i == 0:
						p = t_values['NULL'][zh_words[j-1]] / z
						lambdas[zh_words[j-1]]['NULL'] += (eta * p)
						for zh in inverse['NULL']:
							lambdas[zh]['NULL'] -= (eta * p * t_values['NULL'][zh])
					else:
						t_f_e(en_words[i-1])
						p = t_values[en_words[i-1]][zh_words[j-1]] / z
						lambdas[zh_words[j-1]][en_words[i-1]] += (eta * p)
						for zh in inverse[en_words[i-1]]:
							lambdas[zh][en_words[i-1]] -= (eta * p * t_values[en_words[i-1]][zh])
				t_f_e('NULL')
				t_f_e(en_words[i-1])

		print('Iteration ' + str(t) + ' log-probability: ' + str(log_like))

def report_word_pairs():
	print('t(绝地|jedi) = ' + str(t_values['jedi']['绝地']))
	print('t(机械人|droid) = ' + str(t_values['droid']['机械人']))
	print('t(原力|force) = ' + str(t_values['force']['原力']))
	print('t(原虫|midi-chlorians) = ' + str(t_values['midi-chlorians']['原虫']))
	print('t(你|yousa) = ' + str(t_values['yousa']['你']))

def test():
	for line in testing:
		zh_en = line.split('\t')
		zh_words = zh_en[0].split()
		en_words = zh_en[1].split()
		
		a = np.empty(len(zh_words), dtype = int)
		for n in range(0, len(a)):
			a[n] = -1

		first = 1
		for i, zh in enumerate(zh_words):
			arg_max = 0
			for j, en in enumerate(en_words):
				t = t_values[en][zh]
				if t > arg_max:
					arg_max = t
					a[i] = j
			t = t_values['NULL'][zh]
			if t > arg_max:
				arg_max = t
				a[i] = -1

		for k in range(0, len(a)):
			if a[k] != -1:
				if first == 1:
					print(str(k) + '-' + str(a[k]), end = '')
					first = 0
				else:
					print(' ' + str(k) + '-' + str(a[k]), end = '')
		print()

if __name__ == '__main__':
	read()
#	log_prob_5()
	sga()
#	report_word_pairs()
	test()
