import pretty_midi

from torch import Tensor
from collections import Counter
from utils.constants import *
from torchtext.vocab import vocab


"""
Refer to https://github.com/dvruette/figaro/blob/2b253eb0476453e197ee0599a5c58f87d82a3890/src/input_representation.py
"""


class Tokens:
    @staticmethod
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        tokens = [f"{key}_{pretty_midi.program_to_instrument_name(i)}" for i in range(128)]
        tokens.append(f"{key}_Drum")

        return tokens

    @staticmethod
    def get_midi_tokens(
        bar_key=BAR_KEY,
        pitch_key=PITCH_KEY,
        tempo_key=TEMPO_KEY,
        velocity_key=VELOCITY_KEY,
        duration_key=DURATION_KEY,
        position_key=POSITION_KEY,
        instrument_key=INSTRUMENT_KEY,
    ):
        instrument_tokens = Tokens.get_instrument_tokens(instrument_key)

        bar_tokens = [f"{bar_key}_0"]
        tempo_tokens = [f"{tempo_key}_{i}" for i in range(len(DEFAULT_TEMPO_BINS))]
        position_tokens = [f"{position_key}_{i}" for i in range(DEFAULT_POS_PER_BAR)]
        duration_tokens = [f"{duration_key}_{i}" for i in range(len(DEFAULT_DURATION_BINS))]
        velocity_tokens = [f"{velocity_key}_{i}" for i in range(len(DEFAULT_VELOCITY_BINS))]

        pitch_tokens = [f"{pitch_key}_{i}" for i in range(128)] + [
            f"{pitch_key}_Drum_{i}" for i in range(128)
        ]

        return (
            bar_tokens
            + tempo_tokens
            + instrument_tokens
            + pitch_tokens
            + position_tokens
            + duration_tokens
            + velocity_tokens
        )


class Vocab:
    def __init__(
        self,
        counter,
        specials=[PAD_TOKEN, UNK_TOKEN, EOB_TOKEN, MASK_TOKEN],
        unk_token=UNK_TOKEN,
    ):
        self.vocab = vocab(counter)

        self.specials = specials
        for i, token in enumerate(self.specials):
            self.vocab.insert_token(token, i)

        if unk_token in specials:
            self.vocab.set_default_index(self.vocab.get_stoi()[unk_token])

    def to_i(self, token):
        return self.vocab.get_stoi()[token]

    def to_s(self, idx):
        if idx >= len(self.vocab):
            return UNK_TOKEN
        else:
            return self.vocab.get_itos()[idx]

    def __len__(self):
        return len(self.vocab)

    def encode(self, seq):
        return self.vocab(seq)

    def decode(self, seq):
        if isinstance(seq, Tensor):
            seq = seq.numpy()

        return self.vocab.lookup_tokens(seq)


class RemiVocab(Vocab):
    def __init__(self):
        tokens = Tokens.get_midi_tokens()

        counter = Counter(tokens)
        super().__init__(counter)
