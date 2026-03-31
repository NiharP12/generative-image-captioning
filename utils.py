"""
Utility functions for Image Caption Generator.
"""
import os, re, json
import pandas as pd
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        for w in ["<pad>", "<start>", "<end>", "<unk>"]:
            self._add_word(w)

    def _add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocabulary(self, captions):
        counter = Counter()
        for cap in captions:
            counter.update(self.tokenize(cap))
        for word, count in counter.items():
            if count >= self.freq_threshold:
                self._add_word(word)

    @staticmethod
    def tokenize(text):
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip().split()

    def numericalize(self, text):
        return [self.word2idx.get(t, self.word2idx["<unk>"]) for t in self.tokenize(text)]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx,
                        "idx2word": {str(k): v for k, v in self.idx2word.items()},
                        "freq_threshold": self.freq_threshold}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        v = cls(freq_threshold=data["freq_threshold"])
        v.word2idx = data["word2idx"]
        v.idx2word = {int(k): v2 for k, v2 in data["idx2word"].items()}
        v.idx = len(v.word2idx)
        return v

def load_captions(caption_file):
    df = pd.read_csv(caption_file)
    captions = {}
    for _, row in df.iterrows():
        img = row["image"].strip()
        cap = str(row["caption"]).lower().strip()
        cap = re.sub(r"[^a-zA-Z\s]", "", cap)
        cap = re.sub(r"\s+", " ", cap).strip()
        captions.setdefault(img, []).append(cap)
    return captions

def get_all_captions(captions_dict):
    return [c for caps in captions_dict.values() for c in caps]
