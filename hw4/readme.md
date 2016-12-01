Rosalyn Tan
HW 4

Execution Instructions:
python preprocess.py train.trees > train.trees.pre
python unknown.py train.trees.pre > train.trees.pre.unk
python part[n].py > test.parses
python postprocess.py test.parses > test.parses.post
python evalb.py test.parses.post test.trees

Submission:
part1.py -- code to learn a probabilistic CFG from trees
part2.py -- CKY parser
part3.py -- CKY parser with modifications (add-delta smoothing, horizontal/vertical markovization)
tree_part3.py -- implements horizontal and vertical markovization
