import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
import json


def main():
    for genre in ('Country', 'Metal', 'Pop', 'Rock'):
        path = './Homework 3 Data/' + genre + '/'
        lyrics = []
        i = 0
        for f in os.listdir(path):
            try:
                with open(path + f, 'r') as file:
                    lines = []
                    for line in file:
                        lyric = str(file.readline().lower())
                        lines.append(lyric)
                lyrics.append(lines)
                i += 1
            except UnicodeDecodeError:
                with open(path + f, 'rb') as file:
                    lines = []
                    for line in file:
                        lyric = str(file.readline().lower())
                        lines.append(lyric)
                lyrics.append(lines)
                i += 1

        final_lyrics = []
        i = 0
        length = len(lyrics)
        for song in lyrics:
            for line in song:
                final_line = []
                new_line = re.sub('\\n', '', line)
                new_line = word_tokenize(new_line)
                for token in new_line:
                    if token not in string.punctuation and len(new_line) > 1:
                        final_line.append(token)
                if len(final_line) > 1:
                    final_lyrics.append(final_line)
            i += 1
            percent = round(((i / length) * 100), 2)
            print(f"{percent}% done")
            if i == length:
                print('Done with all documents')
                break

        json_file = genre.lower() + '.json'
        with open(json_file, 'w') as save_file:
            json.dump({'data': final_lyrics}, save_file, indent=4)



if __name__ == '__main__':
    main()
