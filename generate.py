from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
import gensim


def embedding(data):
    model = gensim.models.Word2Vec(
        data,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)


def ngrams(data, n):
    train, vocab = padded_everygram_pipeline(n, data)
    print(train)
    print(list(vocab))
    model = MLE(n)
    model.fit(train, vocab)
    return model


def generate(model):
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for i in range(4):
        for token in model.generate(6, text_seed=['<s>'], random_seed=3):
            if token == '</s>':
                break
            content.append(token)
        content.append('\n')
    return detokenize(content)


def main():
    text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
    model = ngrams(text, 2)
    print('generate:', generate(model))


if __name__ == '__main__':
    main()