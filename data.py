import os
import copy
import time
import torch
import random
import pickle
import pytorch_lightning as pl

from utils.vocab import *
from utils.constants import *

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class DataModule(pl.LightningDataModule):
    """
    Refer to https://github.com/dvruette/figaro/blob/2b253eb0476453e197ee0599a5c58f87d82a3890/src/datasets.py
    """

    def __init__(
        self,
        file_list,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        masking=0.8,
        replace=0.1,
        phase="train",
    ):
        super().__init__()

        self.file_list = file_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.masking = masking
        self.replace = replace

        self.shuffle = True if phase == "train" else False

        self.path = "/".join(file_list[0].split("/")[:3])
        self.vocab = RemiVocab()
        self.setup()

    def setup(self):
        self.data = DatasetSampler(self.file_list)

        self.collator = SeqCollator(
            self.path,
            self.vocab,
            self.vocab.to_i(PAD_TOKEN),
            self.masking,
            self.replace,
        )

    def return_dataloader(self):
        return DataLoader(
            self.data,
            collate_fn=self.collator,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class DatasetSampler(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # load data
        with open(file_path, "rb") as f:
            x = pickle.load(f)

        file_name = file_path.split("/")[-1].split("_")[0]
        x["file_name"] = file_name

        return x


class SeqCollator:
    """
    Refer to https://github.com/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.py
    """

    def __init__(self, path, vocab, pad_token, masking, replace):
        self.path = path
        self.vocab = vocab
        self.pad_token = pad_token

        self.min_pitch = 0
        self.max_pitch = 127
        self.min_vel = 0
        self.max_vel = 31

        self.aug_pitch = range(-6, 7)
        self.aug_vel = range(-3, 4)

        self.masking = masking
        self.replace = replace

    def get_masked_seq(self, feature):
        inputs = copy.deepcopy(feature["events"])
        outputs = copy.deepcopy(feature["events"])

        num_pred = int(round(len(inputs) * 0.15))

        mask_pos = [
            i for i, token in enumerate(inputs) if token != BAR_TOKEN and token != EOB_TOKEN
        ]
        random.shuffle(mask_pos)

        masked_pos = []
        for pos in mask_pos[:num_pred]:
            masked_pos.append(pos)

            # masking
            if random.random() < self.masking:
                inputs[pos] = MASK_TOKEN
            # randomly replace
            elif random.random() < (self.replace / (1 - self.masking)):
                rand_idx = random.randint(0, len(self.vocab) - 1)
                inputs[pos] = self.vocab.to_s(rand_idx)

        unmasked_pos = [pos for pos in range(len(inputs)) if pos not in masked_pos]

        for pos in unmasked_pos:
            outputs[pos] = PAD_TOKEN

        return inputs, outputs

    def check_extreme(self, events, type, low, high):
        for i in range(len(events)):
            event_split = events[i].split("_")

            # skip drum for pitch augmentation
            if "Pitch_Drum" in events[i]:
                continue

            if event_split[0] == type:
                low = min(low, int(event_split[1]))
                high = max(high, int(event_split[1]))

        return low, high

    def augment(self, events, type, min_limit, max_limit, aug_range):
        min_val, max_val = self.check_extreme(events, type=type, low=max_limit, high=min_limit)

        num_key = random.choice(aug_range)
        while min_val + num_key < min_limit or max_val + num_key > max_limit:
            num_key = random.choice(aug_range)

        for i in range(len(events)):
            event_split = events[i].split("_")

            # skip drum for pitch augmentation
            if "Pitch_Drum" in events[i]:
                continue

            if event_split[0] == type:
                new_event = event_split[0] + "_" + str(int(event_split[1]) + num_key)
                events[i] = new_event

        return events

    def get_augment_seq(self, feature):
        events = copy.deepcopy(feature["events"])

        # pitch augmentation
        events = self.augment(
            events,
            type="Pitch",
            min_limit=self.min_pitch,
            max_limit=self.max_pitch,
            aug_range=self.aug_pitch,
        )

        # velocity augmentation
        events = self.augment(
            events,
            type="Velocity",
            min_limit=self.min_vel,
            max_limit=self.max_vel,
            aug_range=self.aug_vel,
        )

        return events

    def get_neighbor_seq(self, feature):
        folder_path = os.path.join(self.path, feature["file_name"])
        file_list = os.listdir(folder_path)

        file_name = random.choice(file_list)
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "rb") as f:
            events = pickle.load(f)["events"]

        return events

    def __call__(self, features):
        # start_time = time.time()

        (
            x_list,
            x_aug_list,
            x_neigh_list,
            x_mask_list,
            y_mask_list,
            file_list,
        ) = ([] for _ in range(6))

        (
            inst_list,
            chord_list,
            tempo_list,
            vel_list,
            dur_list,
            gp_list,
        ) = ([] for _ in range(6))

        batch = {}
        for feature in features:
            ### original inputs
            x = copy.deepcopy(feature["events"])

            ### augment inputs
            x_aug = self.get_augment_seq(feature)

            ### neighbor inputs
            x_neigh = self.get_neighbor_seq(feature)

            ### masking inputs
            x_mask, y_mask = self.get_masked_seq(feature)

            # convert events to index
            x = torch.tensor(self.vocab.encode(x), dtype=torch.long)
            x_aug = torch.tensor(self.vocab.encode(x_aug), dtype=torch.long)
            x_neigh = torch.tensor(self.vocab.encode(x_neigh), dtype=torch.long)
            x_mask = torch.tensor(self.vocab.encode(x_mask), dtype=torch.long)
            y_mask = torch.tensor(self.vocab.encode(y_mask), dtype=torch.long)

            x_list.append(x)
            x_aug_list.append(x_aug)
            x_neigh_list.append(x_neigh)
            x_mask_list.append(x_mask)
            y_mask_list.append(y_mask)

            ### instrument
            inst = torch.zeros(NUM_INST)
            inst[feature["meta_info"]["inst"]] = 1
            inst_list.append(inst)

            ### chord
            chord = torch.zeros(len(PITCH_CLASSES) * len(CHORD_TONE))
            chord_idx = list(
                map(
                    lambda x: PITCH_CLASSES.index(x.split(":")[0]) * len(CHORD_TONE)
                    + CHORD_TONE.index(x.split(":")[1]),
                    feature["meta_info"]["chord"],
                )
            )

            chord[chord_idx] = 1
            chord_list.append(chord)

            ### tempo
            tempo = torch.zeros((len(DEFAULT_TEMPO_BINS),))
            tempo_idx = np.where(DEFAULT_TEMPO_BINS == feature["meta_info"]["tempo"])[0]
            tempo[tempo_idx] = 1
            tempo_list.append(tempo)

            ### velocity & duration
            vel_list.append(feature["meta_info"]["mean_velocity"])
            dur_list.append(feature["meta_info"]["mean_duration"])

            ### groove pattern
            gp = torch.zeros(DEFAULT_POS_PER_BAR)
            gp_idx = feature["meta_info"]["groove_pattern"]
            gp[gp_idx] = 1
            gp_list.append(gp)

            file_list.append(feature["file_name"])

        x_list = pad_sequence(x_list, batch_first=True, padding_value=self.pad_token)
        x_aug_list = pad_sequence(x_aug_list, batch_first=True, padding_value=self.pad_token)
        x_neigh_list = pad_sequence(x_neigh_list, batch_first=True, padding_value=self.pad_token)
        x_mask_list = pad_sequence(x_mask_list, batch_first=True, padding_value=self.pad_token)
        y_mask_list = pad_sequence(y_mask_list, batch_first=True, padding_value=self.pad_token)

        batch["x"] = x_list
        batch["x_aug"] = x_aug_list
        batch["x_neigh"] = x_neigh_list
        batch["x_mask"] = x_mask_list
        batch["y_mask"] = y_mask_list

        batch["inst"] = torch.vstack(inst_list)
        batch["chord"] = torch.vstack(chord_list)
        batch["tempo"] = torch.vstack(tempo_list)

        batch["mean_velocity"] = torch.as_tensor(vel_list)
        batch["mean_duration"] = torch.as_tensor(dur_list)
        batch["groove_pattern"] = torch.vstack(gp_list)

        batch["file_name"] = file_list

        # print(f"Data loader : {time.time() - start_time:.3f} sec")

        return batch
