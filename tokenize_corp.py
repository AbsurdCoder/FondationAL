from classic.tokenizerAL import BPETokenizer as BPE


corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the fox is quick and smart",
    "lazy dogs lie in the sun"
]

bpe = BPE(vocab_size=100)
bpe.train(corpus)



# Encode a new sentence
sentence = "the quick dog jumps high"
tokens = bpe.encode(sentence)
print("BPE Tokens:", tokens)
print("BPE Decoded:", bpe.decode(tokens))