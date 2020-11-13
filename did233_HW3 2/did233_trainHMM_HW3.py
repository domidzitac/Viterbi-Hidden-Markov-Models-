import re
import sys
import argparse
import collections
import numpy
import nltk
from math import log

SMALL_NUMBER = 1e-5


# In POS tagging, functional words (e.g. prepositions, conjunctions, determiners)
# should not be counted upper and lower cased. 
# They should be counted case-insensitive.
def make_word_lower(word, POS):
    # If there are any other upper-case POSes, add them here.
    if POS.startswith("NNP") or \
        POS.startswith("JJ") or word == "I":
        return word
    else:
        return word.lower()

# Function that gets all the words and POSs from the file (works for more that just one file)
def get_words_POS(file):
    all_words, POS_history = [''],['Begin_Sent']
    lcount = 0

    for line in file:
        lcount += 1

        if lcount % 100000 == 0:
            print("Read {0} lines from the training corpus...".format(lcount), file = sys.stderr)

        if line != '\n':
            word_POS = line.strip().split('\t')
            all_words.append(make_word_lower(word_POS[0], word_POS[1])) #append word
            POS_history.append(word_POS[1]) #append state
        else:
            all_words.append('')
            POS_history.append('End_Sent')
            all_words.append('')
            POS_history.append('Begin_Sent')

    return all_words, POS_history

# Function for the likelyhood of a word to be a certain POS as requested in hash 1
def likelihood(word_POS):
    likes = collections.defaultdict(list)

    for state, like in word_POS:
        likes[state].append(like)

    return {state: list_to_dict(likes) for state, likes in likes.items()}

# Function for the transition of states for hash 2
def transition(POS):
    prev_list = collections.defaultdict(list)

    for i in range(len(POS) - 1):
        pos, prev = POS[i], POS[i + 1]
        prev_list[pos].append(prev)

    return {pos: list_to_dict(prev_list) for pos, prev_list in prev_list.items()}

# Function that transforms a list to a dictionary with the frequencies 
def list_to_dict(list):
    counts = collections.defaultdict(int)

    for value in list:
        counts[value] += 1

    return {key: count for key, count in counts.items()}

# Function that loops thru hash table and converts frequencies into probabilities
def convert_frequencies(hash):
    for (pos, val) in hash.items():
        sum = 0

        for word in val:
            sum += val[word]

        sum = float(sum)

        for word in val:
            # R: this is incorrect, you're trying to compute P(w|t)
            # val[word]= float(val[word]/(len(hash)))
            val[word]= float(val[word]) / sum

    return hash
#####################################################################

#  gets the list of possible POS tags for a given word.
# If word is OOV, return a list of possible states.
# Note that in POS tagging, most of the unknown words are nouns.
# At any rate, they are not functional words (e.g. prepositions, conjunctions, etc.)
def get_word_states(word, ambiguity_classes):
    if word in ambiguity_classes:
        return ambiguity_classes[word]

    word = word.lower()

    if word in ambiguity_classes:
        return ambiguity_classes[word]

    # Like I said, only certain POSes are likely if word is OOV.
    return [
        "CD", "FW", "JJ",
        "JJR", "JJS", "NN",
        "NNP", "NNPS", "NNS",
        "RB", "RBR", "RBS",
        "VB", "VBD", "VBG",
        "VBN", "VBP", "VBZ",
        ":"
    ]

def get_transition_prob(prev_pos, pos, transition):
    if prev_pos in transition and \
        pos in transition[prev_pos]:
        return transition[prev_pos][pos]

    return SMALL_NUMBER;

#  get word emission probability P(word|pos) and 
# implement OOV strategies that were given in the homework description.
def get_word_emission_prob(word, pos, likelihood, oov_strategy):
    pos_words = likelihood[pos]

    if word in pos_words:
        # Known word
        return pos_words[word]

    wordlc = word.lower()

    if wordlc in pos_words:
        # Known word (lower-cased)
        return pos_words[wordlc]

    rxp_upper = re.compile("^[A-Z]")
    rxp_number = re.compile("^[0-9]+([,.][0-9]+)*$")
    rxp_punct = re.compile("^\\W+$")

    # This is an OOV word.
    # Let's guess its P(word|pos)
    # This is the guidance from the HW description!
    # Better approaches are available...
    if oov_strategy == "SMALL":
        return SMALL_NUMBER
    elif oov_strategy == "HEURISTIC":
        if (wordlc.endswith("s") or wordlc.endswith("es")) \
            and pos == "NNS":
            return 0.6
        if (wordlc.endswith("d") or wordlc.endswith("ed")) \
            and (pos == "VBD" or pos == "VBN" or pos == "JJ"):
            return 0.6
        elif rxp_upper.search(word) and \
            (pos == "NNP" or pos == "NNPS"):
            return 0.6
        elif rxp_number.search(word) and pos == "CD":
            return 1.0
        elif rxp_punct.search(word) and pos == ":":
            return 1.0
        else:
            return SMALL_NUMBER
    # implement remaining OOV strategies here!

#  this algorithm returns the most probable sequence
# of states (i.e. POS tags) that best explains the output (i.e.
# the sequence of words).
def viterbi(sentence, likelihood, transition, ambiguity_classes, oov_strategy):
    """
    likelihood -> is the P(w|t) emission probability table,
    transition -> is the P(t|previous t) transition probability table,
    ambiguity_classes -> is the list of possible states (POS tags) for a given word.
    """
    #  sentinels for BOS and EOS (end of sentence)
    # Initialize
    # Each state has the index of the previous state which leads
    # to the best decoding score so far: -1 for not being initialized
    # and the value of the best decoding so far: 0.0 for not being initialized.
    # We will use ln(P) so that multiplication of low numbers does not lead to 0s.
    # Instead of multiplying the probs, we will add the logs and maximize the sum.
    # This is the standard approach in HMM decoding to avoid numerical underflow.
    V = [[["Begin_Sent", -1, 0.0]]]

    # For each word, construct the trellis of states.
    for word in sentence:
        w_stats = get_word_states(word, ambiguity_classes)
        w_struct = []

        for pos in w_stats:
            w_struct.append([pos, -1, 0.0])

        V.append(w_struct)

    V.append([["End_Sent", -1, 0.0]])

    # Viterbi decoder: for each two adjacent trellis columns, compute
    # the best path, so far, for each cell.
    for t in range(1, len(V)):
        prev_states = V[t - 1]
        curr_states = V[t]

        if t - 1 < len(sentence):
            wt = sentence[t - 1]
        else:
            # We are at the EOS marker now,
            # not part of the original sentence.
            # Cooresponding word is the empty string.
            wt = ""

        # These are two adjacent trellis columns.
        # For each cell in column j, compute the
        # the best partial sum from all previous is
        # and record the best i and the max sum.
        for j in range(len(curr_states)):
            pj = curr_states[j][0]
            # If it's 1, it will be initialized.
            best_sum = 0
            best_i = -1
            # P(wt|pj) = emission probability
            ep = log(get_word_emission_prob(wt, pj, likelihood, oov_strategy))

            for i in range(len(prev_states)):
                pi = prev_states[i][0]
                si = prev_states[i][2]
                # P(pj|pi) = transition probability
                tp = log(get_transition_prob(pi, pj, transition))

                if best_sum == 0:
                    best_sum = si + tp + ep
                    best_i = i
                elif best_sum < si + tp + ep:
                    best_sum = si + tp + ep
                    best_i = i
            # end for i
            # Record best i/best sum for j
            curr_states[j][1] = best_i
            curr_states[j][2] = best_sum
        # end for j
    # end for t

    # Viterbi decoder: recuperate the best path 
    # through the computed trellis, right to left.
    # Last element of V only contains one state, 'End_Sent'
    # whose best index points back to the best path.
    best_tag_path = []
    sentence_index = len(V) - 1
    best_prev_state_index = V[sentence_index][0][1]

    while sentence_index >= 2:
        best_tag_path.insert(0, V[sentence_index - 1][best_prev_state_index][0])
        sentence_index -= 1
        best_prev_state_index = V[sentence_index][0][1]

    return best_tag_path

#  function that reads in the file to be tagged.
def read_to_be_tagged_file(file):
    all_sentences = []
    curr_sentence = []

    with open(file, "r") as f:
        lcount = 0

        for line in f:
            lcount += 1

            if lcount % 100000 == 0:
                print("Read {0} lines from the file {1}...".format(lcount, file), file = sys.stderr)

            if line != '\n':
                word = line.strip()
                curr_sentence.append(word) #append word
            else:
                all_sentences.append(curr_sentence)
                curr_sentence = []
        # end for
    # end with
    return all_sentences

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 nlp3.py <file to be tagged, one word/line, one empty line/sentence> <SMALL|HEURISTIC>", file = sys.stderr)
        exit(-1)

    to_be_tagged_file = sys.argv[1]
    oov_strategy = sys.argv[2]

    # MODEL TRAINING PART ###################################################################
    with open("WSJ_02-21.pos", "r") as training_file:
        (words, POS) = get_words_POS(training_file)
    # end with

    #HASH TABLES
    #1
    #POS -> table of frequencies of words that occur with that POS 
    #Example: likelihood['DT'] -> {'the':1500,'a':200,'an':100, ...}
    #Hash table of POSs with each value a hash table from words to frequencies
    print("Computing the emission probabilities...", file = sys.stderr)
    POS_hash_freq = likelihood(zip(POS, words))

    #loop through hash table and transform to probababilities
    POS_hash_prob = convert_frequencies(POS_hash_freq)

    # R: we need to retrieve the list of possible
    # states for a given, known word.
    print("Computing the ambiguity classes for words...", file = sys.stderr)
    WORD_to_states = {}

    for pos in POS_hash_prob:
        for word in POS_hash_prob[pos]:
            if not word in WORD_to_states:
                WORD_to_states[word] = []

            if not pos in WORD_to_states[word]:
                WORD_to_states[word].append(pos) 
        # end for word
    # end for pos

    #2
    #STATE -> table of frequencies of following states
    #Example: Transition['Begin_Sent'] -> {'DT':1000,'NNP':500,'VB':200, ...}
    #Example: Transition['DT'] -> {'NN':500,'NNP:'200,'VB':30,,...}
    #Hash table of states with each a value a hash table from states to frequencies

    print("Computing the transition probabilities...", file = sys.stderr)
    STATE_hash_freq = transition(POS)

    #loop through hash table and transform to probababilities

    STATE_hash_prob = convert_frequencies(STATE_hash_freq)
    # END MODEL TRAINING PART ################################################################

    #test_sentence = ['I',
    #    'am',
    #    'very',
    #    'happy',
    #    'that',
    #    'the',
    #    'Viterbi',
    #    'decoder',
    #    'works',
    #    'now',
    #    '!'
    #]

    # This list has the same number of elements as test_sentence.
    #test_tagging = viterbi(test_sentence, POS_hash_prob, STATE_hash_prob, WORD_to_states, "HEURISTIC")

    #for i in range(len(test_sentence)):
    #    print("{0}\t{1}".format(test_sentence[i], test_tagging[i]), file = sys.stdout)

    sentences_to_be_tagged = read_to_be_tagged_file(to_be_tagged_file)

    for sentence in sentences_to_be_tagged:
        # Tag the sentence
        tagging = viterbi(sentence, POS_hash_prob, STATE_hash_prob, WORD_to_states, oov_strategy)

        # Dump the tagging to stdout
        for i in range(len(sentence)):
            print("{0}\t{1}".format(sentence[i], tagging[i]), file = sys.stdout)

        print("", file = sys.stdout)
