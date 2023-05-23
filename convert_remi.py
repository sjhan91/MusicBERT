import os
import pickle
import argparse
import multiprocessing

from glob import glob
from time import time

from utils.remi import REMI
from utils.vocab import *


def get_bars(events):
    return [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]


def convert_type(value, var_type=int):
    if isinstance(value, list):
        results = [var_type(elem) for elem in value]
    else:
        results = var_type(value)

    return results


def convert(file_path):
    try:
        remi = REMI(file_path)
        events, meta_info = remi.get_remi_events()

        # get bar boundary
        bars = get_bars(events)

        contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(events))]
        contexts = [
            (start, end)
            if (end - start) <= (MAX_TOKEN_LEN - 1)
            else (start, start + (MAX_TOKEN_LEN - 1))
            for (start, end) in contexts
        ]

        for j, (start, end) in enumerate(contexts):
            src = events[start:end] + [EOB_TOKEN]

            if len(src) < 30:
                continue

            x = {"events": src, "meta_info": {k: v[j] for k, v in meta_info.items()}}

            # type conversion
            x["meta_info"]["inst"] = convert_type(x["meta_info"]["inst"], int)
            x["meta_info"]["tempo"] = convert_type(x["meta_info"]["tempo"], int)
            x["meta_info"]["mean_velocity"] = convert_type(x["meta_info"]["mean_velocity"], float)
            x["meta_info"]["mean_duration"] = convert_type(x["meta_info"]["mean_duration"], float)
            x["meta_info"]["groove_pattern"] = convert_type(x["meta_info"]["groove_pattern"], int)

            file_name = file_path.split("/")[-1].split(".")[0]
            dest_folder_path = os.path.join(dest_path, file_name)

            if not os.path.exists(dest_folder_path):
                os.makedirs(dest_folder_path)

            with open(os.path.join(dest_folder_path, file_name + "_" + str(j) + ".pkl"), "wb") as f:
                pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("File error occured!")


if __name__ == "__main__":
    global dest_path
    global remi_vocab

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="./data/lmd_full/")
    parser.add_argument("--dest_path", type=str, default="./data/lmd_full_remi/")
    parser.add_argument("--num_process", type=int, default=100)

    args = parser.parse_args()
    dest_path = args.dest_path

    # define vocab
    remi_vocab = RemiVocab()

    path = []
    start_time = time()
    for folder_path in glob(args.src_path + "*"):
        for file_path in glob(folder_path + "/*"):
            path.append(file_path)

    pool = multiprocessing.Pool(args.num_process)
    pool.map(convert, path)
    pool.close()
    pool.join()

    print(f"Time : {time() - start_time:.3f} sec")
