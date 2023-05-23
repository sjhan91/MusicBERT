# MusicBERT Offical

This repository is the implementation of "Systematic Analysis of Music Representations from BERT".


## Getting Started

### Environments

* Python 3.8.8
* Ubuntu 20.04.2 LTS
* Read [requirements.txt](/requirements.txt) for other Python libraries

### Data Download

* [Lakh MIDI Dataset (LMD-full)](https://colinraffel.com/projects/lmd/)

### Data Preprocess

[convert_remi.py](/convert_remi.py) is to obtain the bar-level REMI+ representations from LMD. It supports the parallel processing by specifying the number of processes to --num_process argument.
```
python convert_remi.py --src_path "./data/lmd_full/" --dest_path "./data/lmd_full_remi/" --num_process 100
```

### Model Training
You should modify [config.json](/config.json) for mode change ("BERT", "BERT-aug", "BERT-neighbor", "BERT-dropout"). By setting "strategy" (ex. ddp) in [train.py](/train.py) and "gpus" in [config.json](/config.json) (ex. [0, 1, 2]), you can train the models with distributed GPU settings of pytorch-lightining. Here is an example of BERT-neighbor configurations.

```json
{
    "random_seed": 0,
    "batch_size": 24,
    "num_workers": 16,
    "dim": 768,
    "depth": 12,
    "heads": 12,
    "max_len": 512,
    "rate": 0,
    "masking": 0.8,
    "replace": 0.1,
    "loss_weights": [1, 0.1],
    "lr": 1e-4,
    "epochs": 3,
    "warm_up": 10000,
    "temp": 0.1,
    "gpus": [0],
    "mode": "BERT-neighbor"
}
```


For training the BERT-variants models, the command is as below;
```
python train.py
```

### Model Inference
You can obtain seven evaluation metrics (chords, groove patterns, instruments, tempo, mean velocity, mean duration, song clustering) from [test.ipynb](/test.ipynb).


## Appreciation
I have learned a lot and reused available codes from [dvruette FIGARO](https://github.com/dvruette/figaro), [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch), and [sthalles SimCLR](https://github.com/sthalles/SimCLR/blob/master/simclr.py). Also, I have applied [gautierdag noam scheduler](https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01) for learning warm-up, and positional encodings from [dreamgonfly transformer-pytorch](https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py).


## References
Sangjun Han, Hyeongrae Ihm, Woohyung Lim (LG AI Research), "Systematic Analysis of Music Representations from BERT"