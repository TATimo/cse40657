import random
import collections as c
import tkinter as tk

class English(object):
    """Barebones example of a language model class."""

    def __init__(self, m):
        self.vocab = set() # will contain all unique chars as well as bigrams, trigrams, etc.
        self.counts = {}
        self.m = m # m-gram model
        for j in range(0, self.m):
            self.counts[j] = c.defaultdict(int)
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
                            temp = '<s>' + temp
                    else:
                        temp = line[i-j:i+1]
                    if temp not in self.counts[j]:
                        self.counts[j][temp] = 1
                    else:
                        self.counts[j][temp] += 1
                    self.vocab.add(temp) # unique characters
                    # add stop characters
                    if len(line)-i-1 < j:
                        temp = line[i:len(line)]
                        for k in range(0, j-(len(line)-i-1)):
                            temp = temp + '</s>'
                        if temp not in self.counts[j]:
                            self.counts[j][temp] = 1
                        else:
                            self.counts[j][temp] += 1
                        self.vocab.add(temp)
                    temp = w                    

    # The following two methods make the model work like a finite
    # automaton.

    def start(self):
        """Resets the state."""
        self.state = ''
        for i in range(1, self.m):
            self.state = self.state + '<s>'

    def read(self, w):
        """Reads in character w, updating the state."""
        if self.unigram == True:
            self.state = ''
        elif self.state[0] == '<' and self.state[1] == 's' and self.state[2] == '>':
            self.state = self.state[3:] + w
        else:
            self.state = self.state[1:] + w

    # The following two methods add probabilities to the finite automaton.

    def prob(self, w):
        """Returns the probability of the next character being w given the
        current state."""
        probgram = {}
        countSum = 0
        for key in list(self.counts[0].keys()):
            countSum += self.counts[0][key]
        probgram[0] = self.counts[0][w]/countSum
        if self.unigram == True:
            return probgram[0]
        else:
            for z in range(1, self.m):
                numWordTypes = 0
                for i in list(self.counts[z-1].keys()):
                    if self.state[(self.m-z-1):] in i:
                        numWordTypes += 1
                lamb = self.counts[z-1][self.state[(self.m-z-1):]]/(self.counts[z-1][self.state[(self.m-z-1):]] + numWordTypes)
                probgram[z] = lamb*(self.counts[z-1][self.state[(self.m-z-1):]+w]/self.counts[z-1][self.state[(self.m-z-1):]]) + (1-lamb)*probgram[z-1]
        print (probgram[self.m-1])
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
    m = English(3)
    m.train(args.train)
    m.start()
    m.read('t')
    m.read('h')
    m.prob('e')

    root = tk.Tk()
    app = Application(m, master=root)
    app.mainloop()
    root.destroy()
