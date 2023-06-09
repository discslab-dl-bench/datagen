import random
import nltk
from nltk.corpus import wordnet
import argparse

def get_synonym(word):
    """

    :param word:
    :return: the list of synonyms of the given word
    """
    synset = wordnet.synsets(word)
    synwords = [synset[i].lemmas()[0].name() for i in range(len(synset)) if synset[i].lemmas()[0].name() != word]
    return synwords


def random_swap(text: str, number: int) -> str:
    """
    (RS)
    Randomly choose two words in the sentence and swap their positions. Do this n times.
    :param text:
    :param number:
    :return:
    """
    words = text.split(' ')
    wc = len(words)
    if wc < 2:
        return text
    words[-1] = words[-1][:len(words[-1])-1] # get rid of the newline character
    for i in range(number):
        pos1, pos2 = random.sample(range(wc), 2)
        words[pos1], words[pos2] = words[pos2], words[pos1]
    return ' '.join(words) + '\n'





def random_insertion(text, number):
    """
    (RI)
    Find a random synonym of a random word in the sentence that is not a stop word.
    Insert that synonym into a random position in the sentence. Do this n times.

    :param text:
    :param number:
    :return:
    """
    words = text.split(' ')
    words[-1] = words[-1][:len(words[-1])-1] # get rid of the newline character
    for i in range(number):
        wc = len(words)
        p = random.sample(range(wc), 1)[0]  # get the position of a random word
        synwords = get_synonym(words[p])
        swc = len(synwords)
        if swc > 1:
            syn_pos = random.sample(range(swc), 1)[0]  # get a random synonym of the random word
            to_insert = random.sample(range(wc), 1)[0]  # get a random position to insert the synonym
            words.insert(to_insert, synwords[syn_pos])
    return ' '.join(words) + '\n'


def random_deletion(text: str, number) -> str:
    """
    (RD)
    Randomly remove each word in the sentence with a certain probability.
    :param text:
    :param number: probability that a word will be removed from the text
    :return:
    """
    words = text.split(' ')
    words[-1] = words[-1][:len(words[-1])-1]
    new_words = []

    for w in words:
        if random.uniform(0, 1) > number:
            new_words.append(w)

    if len(new_words) < 1:
        pos = random.sample(range(len(words)), 1)[0]
        new_words.append(words[pos])
    return ' '.join(new_words) + '\n'


def synonym_replacement(text: str, n: int) -> str:
    """
    Randomly choose n words from the sentence.
    Replace each of these words with one of its synonyms chosen at random.
    :param text:
    :param n: randomly replace n words with synonym
    :return: a sentence
    """
    words = text.split(' ')
    words[-1] = words[-1][:len(words[-1])-1]
    wc = len(words)
    if wc < n:
        return text
    indices = list(range(wc))
    pos = random.sample(indices, n)

    for p in pos:
        synwords = get_synonym(words[p])
        swc = len(synwords)
        if swc > 1:
            syn_pos = random.sample(range(swc), 1)[0]
            words[p] = synwords[syn_pos]
    return ' '.join(words) + '\n'


FUNCTIONS = {
"random_swap": random_swap,
"random_insertion": random_insertion,
"random_deletion": random_deletion,
"synonym_replacement": synonym_replacement
}

def text_augmentation(input, output, method, number):
    """

    :param input: path to the input file
    :param output: path to the output file
    :param aug_method: the function for data augmentation
    :param number: either an integer for number of times (RI, RS, SR) or a float for probability (RD)
    :return:
    """
    f_original = open(input, "r")
    f_new = open(output, "w")

    if method == "random_deletion":
        number = float(number)
    else:
        number = int(number)

    while True:
        text = f_original.readline()
        if text == "":
            break
        func = FUNCTIONS[method] # the augmentation function
        new_text = func(text, number)
        f_new.write(new_text)

    f_original.close()
    f_new.close()


if __name__ == "__main__":
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--input', dest='input', required=True)
    PARSER.add_argument('--output', dest='output', required=True)
    PARSER.add_argument('--method', dest='method', choices=["random_swap", "random_insertion", "random_deletion", "synonym_replacement"])
    PARSER.add_argument('--number', dest='number')
    args = PARSER.parse_args()

    text_augmentation(args.input, args.output, args.method, args.number)

