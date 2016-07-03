__author__ = 'John'

import re, sys, getopt

# clean for BioASQ
bioclean = lambda t: ' '.join(re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                     t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                                   '').strip().lower()).split())

# read a text-per-line file and write to a <word, ..., word> file
def texts2words(textsfile, wordsfile):
    with open(textsfile, 'r') as f:
        texts = f.readlines()
    with open(wordsfile, 'w') as f:
        for t in texts: f.write('%s ' % bioclean(t))


def write_deep_vectors(model_filepath, words_filepath, vectors_filepath):
    try:
        import gensim, numpy as np
    except:
        sys.exit('Please, install gensim & numpy')
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(model_filepath, binary=True)
    with open(words_filepath, 'r') as f:
        terms = [bioclean(line) for line in f.readlines()]
    missed = [w for w in terms if w not in model]
    deep_words = [model[w] if w not in missed else np.array([-1]) for w in terms]
    with open(vectors_filepath, 'w') as f:
        for vector in deep_words: f.write('%s\n' % ' '.join([str(d) for d in vector]))
    return missed


def main(argv=None):
    # Parse the input
    opts, args = getopt.getopt(argv, "hm:i:o:", ["help", "model", "input=", "output="])
    use_msg = '''Use as:
	>> python toolkit.py -i textperline.txt
	and it will output a textperline.txt.lm file, which will contain <word, ..., word>
	Else, you can call it as:
	>> python toolkit.py -m binfile -i wordsfile -o vectorsfileout
	and it will read a W2V.bin model file, a file with word per line, and a filename to save the corresponding deep vectors of the words'''
    if len(opts) == 0: sys.exit(use_msg)
    filein, model_in, fileout = None, None, None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit(use_msg)
        elif opt in ('-i', "--input"):
            filein = arg
        elif opt in ('-o', "--output"):
            fileout = arg
        elif opt in ('-m', "--model"):
            model_in = arg
    if filein and model_in and fileout:
        missed = write_deep_vectors(model_in, filein, fileout)
        print missed
    elif filein and fileout:
        texts2words(filein, fileout)
    else:
        sys.exit(use_msg)


if __name__ == "__main__": main(sys.argv[1:])
