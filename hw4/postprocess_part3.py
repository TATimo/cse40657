#!/usr/bin/env python

import sys, fileinput
import tree_part3 as tree

for line in fileinput.input():
    try:
        t = tree.Tree.from_str(line)

        t.untag()
        t.restore_unit()
        t.unbinarize()

        print(t)
    except:
        print("")
    
    
