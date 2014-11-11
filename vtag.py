#!/usr/bin/pypy

# Akshay Srivatsan
# Problem 2: Viterbi Tagger

import sys
import math
from collections import defaultdict

count_tt = defaultdict(int)
count_wt = defaultdict(int)
count = defaultdict(int)
vocab = defaultdict(int)
tag_dict = defaultdict(list)
all_tags = {}
sing_tt = defaultdict(int)
sing_wt = defaultdict(int)
n = int()

# Find the counts from test data
def train(fileName):
    data = file(fileName, "r")
    data.readline()
    t_old = None
    for line in data:
        # Remove \n character
        line = line.rstrip("\n")

        # Increment count_wt
        count_wt[line] += 1
        # Increment count_tt and count and add to vocab
        line = line.split("/")
        count[line[1]] += 1
        vocab[line[0]] += 1
        if t_old is not None:
            count_tt["/".join((line[1], t_old))] += 1

        # Add the tag to the dictionary if it's the first time
        if count_wt["/".join((line[0], line[1]))] == 1:
            tag_dict[line[0]].append(line[1])
           
        # Also update singleton counts appropriately
        if t_old is not None:
            if count_tt["/".join((line[1], t_old))] == 1:
                sing_tt[t_old] += 1
            elif count_tt["/".join((line[1], t_old))] == 2:
                sing_tt[t_old] -= 1

        if count_wt["/".join((line[0], line[1]))] == 1:
            sing_wt[line[1]] += 1
        elif count_wt["/".join((line[0], line[1]))] == 2:
            sing_wt[line[1]] -= 1

        # Maintain a list of all tags we have seen 
        if line[1] != "###":
            all_tags[line[1]] = 1

        # Keep track of t_i-1
        t_old = line[1]

# Use Viterbi to get probabilities
def test(fileName):
    # Read file into memory
    data = file(fileName, "r")
    data = data.readlines()
    global n
    n = len(data) - 1

    # Initialize things
    mu = defaultdict(lambda: float('-inf'), {})
    mu["###/0"] = math.log(1.0)
    w = [None] * len(data)
    w[0] = "###"
    backpointer = {}
    actualTags = [None] * len(data)
    logProb = {}

    # Run Viterbi's algorithm
    for i in range(1, len(data)):
        # Remove \n character
        data[i] = data[i].rstrip("\n")
        w[i], actualTags[i] = data[i].split("/")
        for tc in tag_dict.get(w[i], all_tags.keys()):
            for tp in tag_dict.get(w[i - 1], all_tags.keys()):
                # Add because we're using log prob
                p = p_tt(tc, tp) + p_wt(w[i], tc)
                mu_temp = mu["/".join((tp, str(i - 1)))] + p
                if mu_temp >= mu["/".join((tc, str(i)))]:
                    mu["/".join((tc, str(i)))] = mu_temp
                    logProb["/".join((tc, str(i)))] = p
                    backpointer["/".join((tc, str(i)))] = tp

    # Follow backpointers and store the best path in t
    t = [None] * (len(data))
    t[len(data) - 1] = "###"
    totalProb = 0.0
    for j in range(1, len(data)):
        i = len(data) - j
        t[i - 1] = backpointer["/".join((t[i], str(i)))]
        totalProb += logProb["/".join((t[i], str(i)))]

    # Check accuracy
    novelCorrect = 0.0
    novelTotal = 0.0
    knownCorrect = 0.0
    knownTotal = 0.0
    for i in range(0, len(w)):
        # Ignore ###/###
        if w[i] == "###":
            continue
        # Increment correct words and total
        if actualTags[i] == t[i]:
            if vocab[w[i]] == 0:
                novelCorrect += 1
            else:
                knownCorrect += 1
        if vocab[w[i]] == 0:
            novelTotal += 1
        else:
            knownTotal += 1

    # Print the results
    '''for i in range(0, len(t)):
        print "%s/%s" % (w[i], t[i])'''
    # If no novel words encountered, we are vacuously 100% accuracy
    print "Tagging accuracy (Viterbi decoding): %.2f%% (known: %.2f%% novel: %.2f%%)" % \
          (100 * (novelCorrect + knownCorrect) / (novelTotal + knownTotal),
           100 * knownCorrect / knownTotal, 100 if novelTotal == 0 else 100 * novelCorrect / novelTotal)
    print "Perplexity per Viterbi-tagged test word: %.3f" % math.exp(- totalProb / (len(w) - 1))

def p_tt(t1, t2):
    numerator = count_tt["/".join((t1, t2))] + (1 + sing_tt[t1]) * p_tt_b(t1, t2)
    denominator = count[t1] + 1 + sing_tt[t1]
    return math.log(float(numerator) / float(denominator))

def p_tt_b(t1, t2):
    return float(count[t1]) / float(n)

def p_wt(w, t):
    if w == "###" and t == "###":
        return 0
    numerator = count_wt["/".join((w, t))] + (1 + sing_wt[t]) * p_wt_b(w, t)
    denominator = count[t] + 1 + sing_wt[t]
    return math.log(float(numerator) / float(denominator))

def p_wt_b(w, t):
    return float(vocab[w] + 1) / float(n + len(vocab) + 1)

def main():
    train(sys.argv[1])
    test(sys.argv[2])

if __name__ ==  "__main__":
    main()
