import argparse
import os
import random
import csv
from multiprocessing import Pool

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--chat_dir", type=str, default='data/chat')
parser.add_argument("--emote_file", type=str, default='data/emotes2.txt')
parser.add_argument("--out_dir", type=str, default='dataset')

args = parser.parse_args()


def _remove_and_get_emotes(string, emote_list):
    string_list = string.split()
    ret = []
    emotes = []
    for word in string_list:
        if word not in emote_list:
            ret.append(word)
        else:
            emotes.append(word)
    return " ".join(ret).strip(), emotes


def _read_file_into_list(path):
    lines = []
    with open(path, 'r', errors='ignore', encoding='utf-8') as file:
        for line in file:
            line = line.strip().lower()
            if line not in lines:
                lines.append(line)
    return lines


def write_to(dir, stripped_strings, label_idxs):
    with open(os.path.join(dir, "labels.txt"), "w", encoding='utf-8') as f:
        f.write('\n'.join(label_idxs))

    with open(os.path.join(dir, "chat.txt"), "w", encoding='utf-8') as f:
        f.writelines('\n'.join(stripped_strings))


def get_result(args):
    path, emotes = args
    string_set = set()
    print(path, "starting")
    stripped_strings, labels = [], []
    path = os.path.join('data/chat', path)
    chat = []
    if '.txt' in path:
        chat = _read_file_into_list(path)
    elif '.csv' in path:
        with open(path, 'r', encoding='utf-8') as f:
            d = csv.DictReader(f)
            for row in d:
                a = row['Message']
                a = a.split()
                c = []
                for b in a:
                    if len(b) < 36:
                        c.append(b)
                chat.append(' '.join(c))
    for string in chat:
        ret_string, ret_emotes = _remove_and_get_emotes(string, emotes)
        if len(ret_string.split()) < 2:
            continue

        if ret_string in string_set:
            continue
        string_set.add(ret_string)
        for emote in ret_emotes:
            stripped_strings.append(ret_string)
            labels.append(emote)

    print(path, "done")
    return stripped_strings, labels


if __name__ == '__main__':
    stripped_strings = []
    labels = []
    emotes = set(_read_file_into_list(args.emote_file))
    l = os.listdir(args.chat_dir)
    random.shuffle(l)
    process_pool = Pool(16)
    string_set = set()
    result = process_pool.map(get_result, zip(l, [emotes] * len(l)))
    stripped_strings = []
    labels = []
    for i in range(len(result)):
        a, b = result[i]
        stripped_strings += a
        labels += b
    emote_counts = {}
    for emote in labels:
        emote_counts[emote] = emote_counts.get(emote, 0) + 1

    sorted_emote_names = [k for k, v in sorted(emote_counts.items(), key=lambda item: item[1], reverse=True)]
    label_idxs = [str(sorted_emote_names.index(emote)) for emote in labels]

    with open(os.path.join(args.out_dir, "class_names.txt"), "w") as f:
        f.writelines('\n'.join(sorted_emote_names))

    X_train, X_test, y_train, y_test = train_test_split(stripped_strings, label_idxs, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    write_to(os.path.join(args.out_dir, "train"), X_train, y_train)
    write_to(os.path.join(args.out_dir, "test"), X_test, y_test)
    write_to(os.path.join(args.out_dir, "val"), X_val, y_val)
