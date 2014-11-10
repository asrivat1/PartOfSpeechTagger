#!/usr/bin/pypy

# Akshay Srivatsan
# Problem 2: Viterbi Tagger

import sys
import math
from collections import defaultdict

count_tt = defaultdict(int)
count_wt = defaultdict(int)
vocab = defaultdict(int)

# Find the counts from test data
def train(fileName):
    data = file(fileName, "r")
    t_old = None
    for line in data:
        # Remove \n character
        line = line.rstrip("\n")

        # Increment count_wt
        count_wt[line] += 1
        
        # Increment count_tt and add to vocab
        line = line.split("/")
        vocab[line[0]] += 1
        if t_old is not None:
            key = "/".join((line[1], t_old))
            count_tt[key] += 1

        # Keep track of t_i-1
        t_old = line[1]

# Use Viterbi to get probabilities
def test(fileName):
    # Read file into memory
    data = file(fileName, "r")
    data = data.readlines()

    # Initialize things
    mu = defaultdict(lambda: float('-inf'), {})
    mu["###/0"] = 1.0
    w = [None] * (len(data) + 1)
    w[0] = "###"
    t = [None] * (len(data) + 1)
    t[0] = "###"
    backpointer = {}

    for i in range(1, len(data)):
        # Remove \n character
        data[i] = data[i].rstrip("\n")
        w[i] , t[i] = data[i].split("/")
        for tc in tag_dict(w[i]):
            for tp in tag_dict(w[i - 1]):
                # Add because we're using log prob
                p = p_tt(tc, tp, w[i]) + p_wt(w[i], tc)
                mu_temp = mu["/".join((tp, str(i - 1)))] + p
                if mu_temp > mu["/".join((tc, str(i)))]:
                    mu["/".join((tc, str(i)))] = mu_temp
                    backpointer["/".join((tc, str(i)))] = tp

    t[len(data)] = "###"
    for j in range(1, len(data)):
        i = len(data) - j
        t[i - 1] = backpointer["/".join((t[i], str(i)))]

    for i in range(0, len(t) - 1):
        print "%s/%s" % (w[i], t[i])

def p_tt(t1, t2, w):
    numerator = count_tt["/".join((t1, t2))]
    denominator = 0
    for t in tag_dict(w):
        denominator += count_tt["/".join((t, t2))]
    return math.log(float(numerator) / float(denominator))

def p_wt(w_i, t):
    numerator = count_wt["/".join((w_i, t))]
    denominator = 0
    for w in vocab:
        denominator += count_wt["/".join((w, t))]
    return math.log(float(numerator) / float(denominator))

def tag_dict(w):
    if w == "###":
        return ["###"]
    else:
        return ["C", "H"]
        
def main():
    train(sys.argv[1])
    test(sys.argv[2])

if __name__ ==  "__main__":
    main()
