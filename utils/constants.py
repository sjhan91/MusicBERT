import numpy as np

# key name set
BAR_KEY = "Bar"
PITCH_KEY = "Pitch"
TEMPO_KEY = "Tempo"
VELOCITY_KEY = "Velocity"
DURATION_KEY = "Duration"
POSITION_KEY = "Position"
INSTRUMENT_KEY = "Instrument"

# default grids
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_QUARTERS_PER_BAR = 4
DEFAULT_POS_PER_BAR = DEFAULT_POS_PER_QUARTER * DEFAULT_QUARTERS_PER_BAR

# default bins
DEFAULT_TEMPO_BINS = np.linspace(30, 210, 32, dtype=int)
DEFAULT_VELOCITY_BINS = np.linspace(1, 127, 32, dtype=int)
DEFAULT_DURATION_BINS = np.sort(
    np.concatenate(
        [
            np.arange(1, 13),  # smallest possible units up to 1 quarter
            np.arange(12, 24, 3)[1:],  # 16th notes up to 1 bar
            np.arange(13, 24, 4)[1:],  # triplets up to 1 bar
            np.arange(24, 48, 6),  # 8th notes up to 2 bars
            np.arange(48, 4 * 48, 12),  # quarter notes up to 8 bars
            np.arange(4 * 48, 16 * 48 + 1, 24),  # half notes up to 16 bars
        ]
    )
)

NUM_INST = 129
MAX_TOKEN_LEN = 512

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_TONE = ["maj", "min", "dim", "aug", "dom7", "maj7", "min7"]

# tokens
BAR_TOKEN = "Bar_0"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
EOB_TOKEN = "<eob>"
