from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import Word2Vec
import multiprocessing


def embedding(data):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(
        sentences=data,
        vector_size=150,
        window=4,
        min_count=2,
        negative=10,
        workers=cores-1,
        sg=1)
    # model.train(
    #     corpus_iterable=data,
    #     total_examples=len(data),
    #     epochs=20)
    return model.wv     # return keyed vectors


def ngrams(data, n):
    train, vocab = padded_everygram_pipeline(n, data)
    model = MLE(n)
    model.fit(train, vocab)
    return model


def generate(model):
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for i in range(4):
        line = model.generate(8, text_seed=['<s>'])
        for token in line:
            if token == '</s>':
                content.append('\n')
                break
            content.append(token)
    return detokenize(content)


def main():
    # test
    text = [['<s>', 'a', 'b', 'c', '<\s>'], ['<s>', 'a', 'c', 'd', 'c', 'e', 'f', '<\s>']]
    # model = ngrams(text, 2)
    # print(generate(model))

    print(embedding(text).most_similar(positive=['c']))


if __name__ == '__main__':
    main()