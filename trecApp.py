import gensim, logging
import re

def test():
	with open('Trec2016QuerySummaryCoreTxtEdtion', 'r') as fp:
		for line in fp.readlines():
			print re.split(r'#\d', line.strip())

def trec2vec():

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
	model = gensim.models.Word2Vec.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)

	with open('Trec2016QueryNoteCoreTxtEdtion', 'r') as fp:
		for line in fp.readlines():
			wordlist = line.strip().lower().split()
			worddict = {}
			for word in wordlist:
				try:
					worddict[word] = model.similar_by_word(word, topn=3, restrict_vocab=None)
				except:
					continue
			print worddict

if __name__ == '__main__':
	# trec2vec()
	test()