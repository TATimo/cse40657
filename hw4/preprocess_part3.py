#!/usr/bin/env python

import sys, fileinput
import tree_part3 as tree

for line in fileinput.input():
    t = tree.Tree.from_str(line)

    # Binarize, inserting 'X*' nodes.
    t.binarize()

    # Remove unary nodes
    t.remove_unit()

    # Horizontal/vertical markovization
#    t.horizontal()
    t.vertical()

    # The tree is now strictly binary branching, so that the CFG is in Chomsky normal form.

    # Make sure that all the roots still have the same label.
    assert t.root.label == 'TOP'

    print(t)
    
    
