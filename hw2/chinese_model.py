import random
import math
import collections as c
import tkinter as tk

class Chinese(object):
    """Barebones example of a language model class."""

    def __init__(self, m):
        self.vocab = set() # will contain all unique chars as well as bigrams, trigrams, etc.
        self.counts = {} # dictionary of dictionaries of dictionaries
        self.m = m # m-gram model
        for j in range(0, self.m):
            self.counts[j] = c.defaultdict()
        self.state = ''
        self.unigram = True if self.m == 1 else False

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            line = line.strip()
            for i, w in enumerate(line):
                temp = w
                for j in range(0, self.m):
                    # add start characters
                    if i-j < 0:
                        temp = line[0:i+1] #check indices
                        for k in range(0, j-i):
                            temp = '~' + temp
                    else:
                        temp = line[i-j:i+1]
                    # get context of character
                    if len(temp) > 1:
                        u = temp[:-1]
                    else:
                        u = ''
                    if u not in self.counts[j]:
                        self.counts[j][u] = {}
                        self.counts[j][u][w] = 1
                    elif w not in self.counts[j][u]:
                        self.counts[j][u][w] = 1
                    else:
                        self.counts[j][u][w] += 1
                    self.vocab.add(temp) # unique grams
                    temp = w                    

    # The following two methods make the model work like a finite
    # automaton.

    def start(self):
        """Resets the state."""
        self.state = ''
        for i in range(1, self.m):
            self.state = self.state + '~'

    def read(self, w):
        """Reads in character w, updating the state."""
        if self.unigram == True:
            self.state = ''
        else:
            self.state = self.state[1:] + w

    # The following two methods add probabilities to the finite automaton.

    # using Witten-Bell smoothing
    def prob(self, w):
        """Returns the probability of the next character being w given the
        current state."""
        probgram = {}
        countSum = 0
        for key in list(self.counts[0][''].keys()):
            countSum += self.counts[0][''][key]
        lamb = countSum / (countSum + len(self.counts[0]['']))
        if w in self.counts[0][''].keys():
            probgram[0] = lamb * (self.counts[0][''][w]/countSum) + (1-lamb)*(1/len(self.counts[0]['']))
        else:
            probgram[0] = (1-lamb)*(1/len(self.counts[0]['']))
        if self.unigram == True:
            return probgram[0]
        else:
            for z in range(1, self.m):
                # get number of word types following context
                if self.state[-z:] in self.counts[z]:
                    numWordTypes = len(self.counts[z][self.state[-z:]].keys())
                else:
                    numWordTypes = 0
                if numWordTypes == 0:
                    lamb = 0
                else:
                    countSum = 0
                    if self.state[-z:] in self.counts[z]:
                        for key in self.counts[z][self.state[-z:]]:
                            countSum += self.counts[z][self.state[-z:]][key]
                    lamb = countSum / (countSum + numWordTypes)
                if lamb == 0:
                    probgram[z] = probgram[z-1]
                else:
                    # recursive Witten-Bell smoothing formula
                    if self.state[-z:] in self.counts[z] and w in self.counts[z][self.state[-z:]]:
                        probgram[z] = lamb*(self.counts[z][self.state[-z:]][w]/countSum) + (1-lamb)*probgram[z-1]
                    else:
                        probgram[z] = (1-lamb)*probgram[z-1]
        return probgram[self.m-1]

    def probs(self):
        """Returns a dict mapping from all characters in the vocabulary to the
probabilities of each character."""
        return {w: self.prob(w) for w in self.vocab}

# class below given by Professor David Chiang
class Application(tk.Frame):
    def __init__(self, model, master=None):
        self.model = model

        tk.Frame.__init__(self, master)
        self.pack()

        self.INPUT = tk.Text(self)
        self.INPUT.pack()

        self.chars = ['qwertyuiop',
                      'asdfghjkl',
                      'zxcvbnm,.',
                      ' ']

        self.KEYS = tk.Frame(self)
        for row in self.chars:
            r = tk.Frame(self.KEYS)
            for w in row:
                # trick to make button sized in pixels
                f = tk.Frame(r, height=32)
                b = tk.Button(f, text=w, command=lambda w=w: self.press(w))
                b.pack(fill=tk.BOTH, expand=1)
                f.pack(side=tk.LEFT)
                f.pack_propagate(False)
            r.pack()
        self.KEYS.pack()

        self.TOOLBAR = tk.Frame()

        self.BEST = tk.Button(self.TOOLBAR, text='Best', command=self.best, 
                              repeatdelay=500, repeatinterval=1)
        self.BEST.pack(side=tk.LEFT)

        self.WORST = tk.Button(self.TOOLBAR, text='Worst', command=self.worst, 
                               repeatdelay=500, repeatinterval=1)
        self.WORST.pack(side=tk.LEFT)

        self.RANDOM = tk.Button(self.TOOLBAR, text='Random', command=self.random, 
                                repeatdelay=500, repeatinterval=1)
        self.RANDOM.pack(side=tk.LEFT)

        self.QUIT = tk.Button(self.TOOLBAR, text='Quit', command=self.quit)
        self.QUIT.pack(side=tk.LEFT)

        self.TOOLBAR.pack()

        self.update()
        self.resize_keys()

    def resize_keys(self):
        for bs, ws in zip(self.KEYS.winfo_children(), self.chars):
            wds = [150*self.model.prob(w)+15 for w in ws]
            wds = [int(wd+0.5) for wd in wds]

            for b, wd in zip(bs.winfo_children(), wds):
                b.config(width=wd)

    def press(self, w):
        self.INPUT.insert(tk.END, w)
        self.INPUT.see(tk.END)
        self.model.read(w)
        self.resize_keys()

    def best(self):
        _, w = max((p, w) for (w, p) in self.model.probs().items())
        self.press(w)

    def worst(self):
        _, w = min((p, w) for (w, p) in self.model.probs().items())
        self.press(w)

    def random(self):
        s = 0.
        r = random.random()
        p = self.model.probs()
        for w in p:
            s += p[w]
            if s > r:
                break
        self.press(w)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='train')
    args = parser.parse_args()

    ##### Replace this line with an instantiation of your model #####
    m = English(10)
    m.train(args.train)
    m.start()

    counter = 0
    maxProb = 0
    maxChar = ''
    numCorrect = 0
    totalGuess = 0
    lineCount = 0
    logProbSum = 0

    for line in open("./hw2-files/english/test"):
        line = line.strip()
        if lineCount % 10 == 0:
            print(lineCount)
        for c in line:
#            if counter >= 10:
#                break
#            logProbSum += math.log(m.prob(c))
            for key in list(m.counts[0][''].keys()):
                prob = m.prob(key)
                if prob > maxProb:
                    maxProb = prob
                    maxChar = key
            if c == maxChar:
                numCorrect += 1
            m.read(c)
#            print("%s, %f" % (maxChar, maxProb))
#            counter += 1
            maxProb = 0
            totalGuess += 1
        m.start()
        lineCount += 1
#    print("Perplexity on test: %f" % (math.exp(-1*logProbSum/counter)))
    print("Accuracy on test: %f" % (numCorrect / totalGuess))

#    root = tk.Tk()
#    app = Application(m, master=root)
#    app.mainloop()
#    root.destroy()
