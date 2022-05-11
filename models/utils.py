class WordIndexer:
    PADDING_VALUE = 0
    UNKNOWN_VALUE = 1

    def __init__(self, vocab_and_freq_dict, min_frequency=20, max_words=None):
        vocab_sorted_by_freq = sorted(vocab_and_freq_dict.keys(),
                                      key=lambda x: vocab_and_freq_dict[x], reverse=True)

        if min_frequency is not None:
            vocab_sorted_by_freq = [word for word in vocab_sorted_by_freq
                                    if vocab_and_freq_dict[word] >= min_frequency]
        if max_words is not None:
            vocab_sorted_by_freq = vocab_sorted_by_freq[:max_words]

        self.vocab_index_dict = {k: (v + 2) for v, k in enumerate(vocab_sorted_by_freq)}

    @property
    def padding_val(self):
        return WordIndexer.PADDING_VALUE

    @property
    def unknown_val(self):
        return WordIndexer.UNKNOWN_VALUE

    def index_of(self, word):
        return self.vocab_index_dict.get(word, WordIndexer.UNKNOWN_VALUE)

    def __len__(self):
        return len(self.vocab_index_dict) + 2
