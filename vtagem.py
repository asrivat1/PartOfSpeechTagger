#!/usr/bin/pypy

# Akshay Srivatsan
# Problem 2: Viterbi Tagger

import sys
import math
from collections import defaultdict

count_tt_o = defaultdict(int)
count_wt_o = defaultdict(int)
count_o = defaultdict(int)
count_tt = defaultdict(int)
count_wt = defaultdict(int)
count = defaultdict(int)
vocab = defaultdict(int)
vocab_r = defaultdict(int)
tag_dict = defaultdict(list)
all_tags = {}
sing_tt = defaultdict(int)
sing_wt = defaultdict(int)
n = int()

# Find the counts from test data
def train(fileName):
    global count_tt_o
    global count_wt_o
    global count_o
    global count_tt
    global count_wt
    global count
    global vocab
    global vocab_r
    global tag_dict
    global all_tags
    global sing_tt
    global sing_wt

    data = file(fileName, "r")
    t_old = None
    for line in data:
        # Remove \n character
        line = line.rstrip("\n")

        # Increment count_wt
        count_wt_o[line] += 1
        # Increment count_tt and count and add to vocab
        line = line.split("/")
        count_o[line[1]] += 1
        vocab[line[0]] += 1
        if t_old is not None:
            count_tt_o["/".join((line[1], t_old))] += 1

        # Add the tag to the dictionary if it's the first time
        if count_wt_o["/".join((line[0], line[1]))] == 1:
            tag_dict[line[0]].append(line[1])
           
        # Also update singleton counts appropriately
        if t_old is not None:
            if count_tt_o["/".join((line[1], t_old))] == 1:
                sing_tt[t_old] += 1
            elif count_tt_o["/".join((line[1], t_old))] == 2:
                sing_tt[t_old] -= 1

        if count_wt_o["/".join((line[0], line[1]))] == 1:
            sing_wt[line[1]] += 1
        elif count_wt_o["/".join((line[0], line[1]))] == 2:
            sing_wt[line[1]] -= 1

        # Maintain a list of all tags we have seen 
        if line[1] != "###":
            all_tags[line[1]] = 1

        # Keep track of t_i-1
        t_old = line[1]

        # Set the current
        count_tt = defaultdict(int, count_tt_o)
        count_wt = defaultdict(int, count_wt_o)
        count = defaultdict(int, count_o)

# Use Viterbi to get probabilities
def test(fileName):
    global n

    # Read file into memory
    data = file(fileName, "r")
    data = data.readlines()
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
    seenCorrect = 0.0
    seenTotal = 0.0
    for i in range(0, len(w)):
        # Ignore ###/###
        if w[i] == "###":
            continue
        # Increment correct words and total
        if actualTags[i] == t[i]:
            if vocab[w[i]] > 0:
                knownCorrect += 1
            elif vocab_r[w[i]] > 0:
                seenCorrect += 1
            else:
                novelCorrect += 1
        if vocab[w[i]] > 0:
            knownTotal += 1
        elif vocab_r[w[i]] > 0:
            seenTotal += 1
        else:
            novelTotal += 1

    # Print the results
    '''for i in range(0, len(t)):
        print "%s/%s" % (w[i], t[i])'''
    # If no novel words encountered, print 0% accuracy
    print "Tagging accuracy (Viterbi decoding): %.2f%% (known: %.2f%% seen: %.2f%% novel: %.2f%%)" % \
          (100 * (novelCorrect + seenCorrect + knownCorrect) / (novelTotal + seenCorrect + knownTotal),
           100 * knownCorrect / knownTotal,
           0 if seenTotal == 0 else 100 * seenCorrect / seenTotal,
           0 if novelTotal == 0 else 100 * novelCorrect / novelTotal)
    print "Perplexity per Viterbi-tagged test word: %.3f" % math.exp(- totalProb / (len(w) - 1))


def logadd(x, y):
    if y == 0:
        return x
    if x == 0:
        return y
    if y <= x:
        return x + math.log1p(math.exp(y - x))
    else:
        return y + math.log1p(math.exp(x - y))

def forward_backward(fileName, iteration):
    global count_tt
    global count_wt
    global count
    global n

    # Read file into memory
    data = file(fileName, "r")
    data = data.readlines()
    n = len(data) - 1

    # Initialize things
    a = defaultdict(lambda: float('-inf'), {})
    a["###/0"] = math.log(1.0)
    b = defaultdict(lambda: float('-inf'), {})
    w = [None] * len(data)
    w[0] = "###"
    logProb = [("###", float('-inf'))] * len(data)
    transProb = defaultdict(lambda: ("###", float('-inf')))
    count_tt_n = defaultdict(int, count_tt_o)
    count_wt_n = defaultdict(int, count_wt_o)
    count_n = defaultdict(int, count_o)

    # Run forward part
    for i in range(1, len(data)):
        # Remove \n character
        data[i] = data[i].rstrip("\n")
        w[i] = data[i]
        for tc in tag_dict.get(w[i], all_tags.keys()):
            for tp in tag_dict.get(w[i - 1], all_tags.keys()):
                # Add because we're using log prob
                p = p_tt(tc, tp) + p_wt(w[i], tc)
                a["/".join((tc, str(i)))] = logadd(a["/".join((tc, str(i)))],
                                                   a["/".join((tp, str(i - 1)))] + p)
    S = a["/".join(("###", str(n)))]
    b["/".join(("###", str(n)))] = math.log(1.0)

    # Run backward part
    for j in range(1, len(data)):
        i = len(data) - j
        for tc in tag_dict.get(w[i], all_tags.keys()):
            # Compute p(ti | w), but only remember the tag with max prob
            if logProb[i][1] < a["/".join((tc, str(i)))] + b["/".join((tc, str(i)))] - S:
                logProb[i] = (tc, a["/".join((tc, str(i)))] + b["/".join((tc, str(i)))] - S)
            for tp in tag_dict.get(w[i - 1], all_tags.keys()):
                # Add because we're using log prob
                p = p_tt(tc, tp) + p_wt(w[i], tc)
                b["/".join((tp, str(i - 1)))] = logadd(b["/".join((tp, str(i - 1)))],
                                                       b["/".join((tc, str(i)))] + p)
                if transProb["/".join((tc, str(i)))][1] < a["/".join((tp, str(i - 1)))] + p \
                                                     + b["/".join((tc, str(i)))] - S:
                    transProb["/".join((tc, str(i)))] = (tp, a["/".join((tp, str(i - 1)))] + p \
                                                     + b["/".join((tc, str(i)))] - S)
        count_tt_n["/".join((logProb[i][0], transProb["/".join((tc, str(i)))][0]))] += 1
        count_n[logProb[i][0]] += 1
        count_wt_n["/".join((w[i], logProb[i][0]))] += 1

    # Calculate perplexity
    runningP = 0.0
    for tag in all_tags:
        runningP = logadd(runningP, a.get("/".join((tag, "1")), 0) + b.get("/".join((tag, "1")), 0))
    perplexity = math.exp(- runningP / n)
    print "Iteration %d: Perplexity per untagged raw word: %.2f" % (iteration, perplexity)

    # Update counts
    count_tt = defaultdict(int, count_tt_n)
    count_wt = defaultdict(int, count_wt_n)
    count = defaultdict(int, count_n)

def p_tt(t1, t2):
    numerator = count_tt["/".join((t1, t2))] + (1 + sing_tt[t2]) * p_tt_b(t1, t2)
    denominator = count[t2] + 1 + sing_tt[t2]
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
    return float(vocab[w] + vocab_r[w] + 1) / float(n + len(vocab) + len(vocab_r) + 1)

def supplementVocab(fileName):
    global vocab_r

    data = file(fileName, "r")
    for line in data:
        # Remove \n character
        line = line.rstrip("\n")
        line = line.split("/")
        vocab_r[line[0]] += 1

def main():
    train(sys.argv[1])
    supplementVocab(sys.argv[3])
    # Now run forward-backward
    for i in range(0, 4):
        test(sys.argv[2])
        print ""
        forward_backward(sys.argv[3], i)
        print ""

if __name__ ==  "__main__":
    main()
