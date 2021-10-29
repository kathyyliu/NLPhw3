from nltk import pos_tag
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import Word2Vec
import multiprocessing
import json
import random


def pos_tagging(data):
  pos_tags = {}
  for line in data:
      for pair in pos_tag(line, tagset='universal'):
          if pair[1] in pos_tags:
            pos_tags[pair[1]].add(pair[0])
          else:
            pos_tags[pair[1]] = {pair[0]}
  return pos_tags


def embedding(data):
    cores = multiprocessing.cpu_count()
    model = Word2Vec(
        sentences=data,
        vector_size=150,
        window=4,
        min_count=0,
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


def generate(data, n=2, num_lines=4, max_line_len=8, similarity_threshold=.998, semantics=True):
    model = ngrams(data, n)
    song_tags = pos_tagging(data)
    vectors = embedding(data)
    new_song = []
    for i in range(num_lines):
        line = model.generate(max_line_len, text_seed=['<s>'])
        new_tags = pos_tag(line, tagset='universal')
        for pair in new_tags:
            new_token = pair[0]
            new_tag = pair[1]
            if new_token == '</s>':
                break
            elif new_token == '<s>':
                continue
            elif semantics and new_tag in ('ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'PRT', 'PRON'):
                top_similar =[]
                for token in song_tags[new_tag]:
                    similarity = vectors.similarity(new_token, token)
                    if similarity > similarity_threshold and token != new_token:
                        top_similar.append(token)
                if top_similar:
                    i = random.randint(0, len(top_similar)-1)
                    new_song.append(top_similar[i])
                    continue
            new_song.append(new_token)
        new_song.append('\n')
    detokenize = TreebankWordDetokenizer().detokenize
    return detokenize(new_song)


def main():
    # for n in (1, 2, 3):
    #     print('##########', n, 'GRAM ##########')
    #     for genre in ('country', 'metal', 'pop', 'rock'):
    #         file_name = genre + '.json'
    #         f = open(file_name, )
    #         data = json.load(f)['data']
    #         print(f"{genre}:\n{generate(data, n=n)}\n")

    for i in range(10):
        f = open('rock.json', )
        data = json.load(f)['data']
        print(f"########## rock {i+1} ##########\n"
              f"1: {generate(data)}\n"
              f"\t\t\t--------\n"
              f"2: {generate(data, semantics=False)}\n")


if __name__ == '__main__':
    # nltk.download('universal_tagset')
    main()