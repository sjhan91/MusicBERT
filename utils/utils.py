import os
import glob
import torch
import random
import numpy as np

from collections import Counter
from joblib import parallel_backend

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error


def folder_to_file(folder_list):
    file_list = [glob.glob(os.path.join(folder, "*")) for folder in folder_list]
    file_list = [elem for sublist in file_list for elem in sublist]

    return file_list


def folder_to_multiple_file(folder_list, k=1):
    results = []
    origin_k = k

    for folder in folder_list:
        file_list = glob.glob(os.path.join(folder, "*"))

        k = origin_k
        if k > len(file_list):
            k = len(file_list)

        file = random.sample(file_list, k)
        results += file

    return results


def dataset_split(dataset, train_ratio=0.8, val_ratio=0.1):
    num_train = int(len(dataset) * train_ratio)
    num_val = int(len(dataset) * val_ratio)

    train_idx = num_train
    val_idx = num_train + num_val

    train_set = dataset[:train_idx]
    val_set = dataset[train_idx:val_idx]
    test_set = dataset[val_idx:]

    return train_set, val_set, test_set


def to_numpy(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.numpy()
    elif isinstance(x, list):
        x = np.asarray(x)

    return x


def get_feat(data_set, model, device):
    data = {}
    label = {}

    label_list = [
        "inst",
        "chord",
        "tempo",
        "mean_velocity",
        "mean_duration",
        "groove_pattern",
        "file_name",
    ]

    for label_name in label_list:
        label[label_name] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_set):
            _, _, h_list = model(batch["x"].to(device))

            for i, h in enumerate(h_list):
                if i not in data:
                    data[i] = []

                data[i].append(to_numpy(h))

            for label_name in label_list:
                label[label_name].append(to_numpy(batch[label_name]))

    for label_name in label_list:
        elem = label[label_name]
        elem = np.hstack(elem) if elem[-1].ndim == 1 else np.vstack(elem)
        label[label_name] = elem

    return data, label


def normalize(train, test):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)

    train = (train - mean) / std
    test = (test - mean) / std

    return train, test


def probing(train, test, metrics, task):
    train_feat, train_labels = train
    test_feat, test_labels = test

    for label_list in train_labels.keys():
        # print(label_list, task[label_list])
        if label_list == "file_name":
            continue

        if label_list not in metrics:
            metrics[label_list] = []

        eval_dict = {
            "regression": mean_squared_error,
            "multi-label": roc_auc_score,
            "multi-class": accuracy_score,
        }

        reg_flag = task[label_list] == "regression"
        clf_flag = task[label_list] == "multi-label" or task[label_list] == "multi-class"

        # define model
        if reg_flag:
            clf = Ridge(alpha=100, max_iter=1000)
        elif clf_flag:
            clf = RidgeClassifier(alpha=100, max_iter=1000)

        # evaluation function
        eval = eval_dict[task[label_list]]

        # remove zero labels
        f = lambda x: np.where(np.sum(x, axis=1) != 0)

        if clf_flag:
            train_idx = f(train_labels[label_list])[0]
            test_idx = f(test_labels[label_list])[0]

        train_feat_clean = train_feat[train_idx]
        train_labels_clean = train_labels[label_list][train_idx]

        test_feat_clean = test_feat[test_idx]
        test_labels_clean = test_labels[label_list][test_idx]

        # normalize
        train_feat_clean, test_feat_clean = normalize(train_feat_clean, test_feat_clean)

        # train model
        clf.fit(train_feat_clean, train_labels_clean)

        # evaluate
        if label_list == "chord":
            conf = clf.decision_function(test_feat_clean)
            pred = 2 * (conf > -0.8) - 1
            pred[pred == -1] = 0
        else:
            pred = clf.predict(test_feat_clean)

        if reg_flag:
            metric = np.sqrt(eval(test_labels_clean, pred))
        elif task[label_list] == "multi-label":
            metric = eval(test_labels_clean, pred, average="weighted")
        elif task[label_list] == "multi-class":
            metric = eval(test_labels_clean, pred)

        metrics[label_list].append(metric)

    return metrics


def clustering(train, test, metrics, num_k):
    train_feat, train_labels = train
    test_feat, test_labels = test

    label_list = "song_clustering"
    if label_list not in metrics:
        metrics[label_list] = []

    # normalize
    train_feat, test_feat = normalize(train_feat, test_feat)

    # training
    with parallel_backend("threading", n_jobs=30):
        kmeans = MiniBatchKMeans(n_clusters=num_k).fit(train_feat)

    # predict
    pred = kmeans.predict(test_feat)

    song_class = {}
    for i in range(pred.shape[0]):
        if test_labels[i] not in song_class:
            song_class[test_labels[i]] = []

        song_class[test_labels[i]].append(pred[i])

    metric = []
    for key in song_class.keys():
        count = Counter(song_class[key])
        total_count = sum(list(count.values()))

        prob = np.zeros(num_k)
        for k, v in count.items():
            prob[k] = v / total_count

        # entropy
        prob += 1e-6
        entropy = -np.sum(prob * np.log(prob)) / np.log(num_k)
        metric.append(entropy)

    metrics[label_list].append(np.mean(metric))

    return metrics
