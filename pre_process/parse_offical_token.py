import argparse
import csv
import json

label_file = "../data/official_data/train.json"
token_file = '../data/official_data/dictionary.txt'

def parse_symbols2(truth):
    unique_symbols = set()
    i = 0
    while i < len(truth):
        char = truth[i]
        i += 1
        if char.isspace():
            continue
        else:
            unique_symbols.add(char)
    return unique_symbols

def merge(file):
    unique_symbols = set()
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.strip().split()
            print(tmp[0])
            unique_symbols.add(tmp[0])
    return unique_symbols

def create_tokens(groundtruth, output="tokens.txt"):
    with open(groundtruth, "r") as fd:
        unique_symbols = set()
        lines = fd.readlines()

        for line in lines:
            tmp = line.strip().split()
            truth_symbols = parse_symbols2(tmp[1:])
            unique_symbols = unique_symbols.union(truth_symbols)

        unique_symbols = unique_symbols.union(merge("./dictionary.txt"))

        # This is somehow wrong, as it should be recognised as "less than N"
        # It would be two symbols, which are both already present.

        symbols = list(unique_symbols)
        symbols.sort()
        num = 0
        with open(output, "w") as output_fd:
            for symbol in symbols:
                output_fd.write(str(symbol) + "\t" + str(num) + "\n")
                num += 1

def gen_token():

    data_dict = {}
    unique_symbols = set()
    with open(label_file, 'r') as f_label:
        data_dict = json.load(f_label)
        for item in data_dict.values():
            truth_symbols = parse_symbols2(item.split())
            unique_symbols = unique_symbols.union(truth_symbols)

    symbols = list(unique_symbols)
    print("total symbols: %d " % len(unique_symbols))
    symbols.sort()

    num = 0
    with open(token_file, 'w') as out_f:
        for symbol in symbols:
            out_f.write(str(symbol) + "\t" + str(num) + "\n")
            num += 1


def get_token_from_token_file():
    token_file_ex = "../data/dictionary.txt"
    token_file_now = "../data/official_data/dic_now.txt"
    data_dict = {}
    unique_symbols = set()
    with open(token_file_ex, 'r') as f_token:
        lines = f_token.readlines()
        for line in lines:
            item = line.strip().split()[0]
            unique_symbols = unique_symbols.union(item)
    symbols = list(unique_symbols)
    print("total symbols: %d " % len(unique_symbols))
    symbols.sort()

    num = 0
    with open(token_file_now, 'w') as out_f:
        for symbol in symbols:
            out_f.write(str(symbol) + "\t" + str(num) + "\n")
            num += 1


if __name__ == "__main__":

    # gen_token()
    get_token_from_token_file()
