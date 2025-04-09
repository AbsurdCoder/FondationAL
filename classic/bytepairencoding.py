# """
# This function is implementation of BPE tokenizer
# Step-by-step:
# 1. Tokenization
# 2. Find Frequent Pairs
# 3. Find the next frequent pair and merge it, until you have a vocabulary size limit (e.g., 50).
# 4. Final Tokenized Words
# """


# import re
# from collections import defaultdict, Counter

# class BPE:
#     def __init__(self, vocab_size=50):
#         self.vocab_size = vocab_size
#         self.vocab = {}
#         self.bpe_codes = []

#     def train(self, corpus_texts):
#         corpus = Counter()
#         for line in corpus_texts:
#             for word in line.strip().split():
#                 tokens = " ".join([word]) + " </w>"
#                 corpus[tokens] += 1

#         while len(self.vocab) < self.vocab_size:
#             pairs = self.get_stats(corpus)
#             if not pairs:
#                 break
#             best_pair = max(pairs, key=pairs.get)
#             self.bpe_codes.append(best_pair)
#             corpus = self.merge_vocab(best_pair, corpus)
#         self.vocab = {w: i for i, w in enumerate(corpus.keys())}


#     def get_stats(self, corpus):
#         pairs = defaultdict(int)
#         for word,freq in corpus.items():
#             characters = word.split()
#             for i in range(len(characters)-1):
#                 pair = (characters[i], characters[i+1])
#                 pairs[pair] += freq
#         return pairs
    

#     def merge_vocab(self, pair, corpus):
#         new_corpus = {}
#         bigram = re.escape(" ".join(pair))
#         pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#         for word in corpus:
#             new_word = pattern.sub("".join(pair), word)
#             new_corpus[new_word] = corpus[word]
#         return new_corpus

#     def encode_word(self, word):
#             word = list(word) + ['</w>']
#             while True:
#                 pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
#                 candidates = [p for p in pairs if p in self.bpe_codes]
#                 if not candidates:
#                     break
#                 pair = min(candidates, key=lambda x: self.bpe_codes.index(x))
#                 i = pairs.index(pair)
#                 word = word[:i] + [''.join(pair)] + word[i+2:]
#             return word

#     def encode(self, text):
#         tokens = []
#         for word in text.strip().split():
#             tokens.extend(self.encode_word(word))
#         return tokens
    




import re
from collections import defaultdict, Counter

class BPE:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_codes = []

    def get_stats(self, corpus):
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        new_corpus = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in corpus:
            new_word = pattern.sub("".join(pair), word)
            new_corpus[new_word] = corpus[word]
        return new_corpus

    def train(self, corpus_texts):
        # Tokenize words and count frequencies
        corpus = Counter()
        for line in corpus_texts:
            for word in line.strip().split():
                tokens = " ".join(list(word)) + " </w>"
                corpus[tokens] += 1

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.bpe_codes.append(best_pair)
            corpus = self.merge_vocab(best_pair, corpus)

        self.vocab = {w: i for i, w in enumerate(corpus.keys())}

    def encode_word(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            candidates = [p for p in pairs if p in self.bpe_codes]
            if not candidates:
                break
            pair = min(candidates, key=lambda x: self.bpe_codes.index(x))
            i = pairs.index(pair)
            word = word[:i] + [''.join(pair)] + word[i+2:]
        return word

    def encode(self, text):
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens

data = [
"low",
"lower",
"newest",
"widest"
]

bpe = BPE(vocab_size=50)
bpe.train(data)

print("Learned BPE codes:", bpe.bpe_codes)

text = "lower widest"
tokens = bpe.encode(text)
print("Encoded:", tokens)
