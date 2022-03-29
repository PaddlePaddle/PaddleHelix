import os
import random
import argparse
import time
from datetime import datetime
from tqdm import tqdm

import paddle

paddle.disable_static()
import paddle.nn.functional as F
import numpy as np

from model import ProteinSIGN
from dataset import GoTermDataset, GoTermDataLoader
from custom_metrics import do_compute_metrics
from utils import add_saved_args_and_params


def do_compute(model, batch):
    logits = model(*batch[:-1])
    return logits, batch[-1]


def run_batch(model, data_loader, desc):
    logits_list = []
    ground_truth = []

    for batch in tqdm(data_loader, desc=f"{desc}"):
        logits, labels = do_compute(model, batch)
        logits_list.append(F.sigmoid(logits).tolist())
        ground_truth.append(labels.tolist())

    logits_list = np.concatenate(logits_list)
    ground_truth = np.concatenate(ground_truth)
    metrics = do_compute_metrics(ground_truth, logits_list)

    return metrics


def test(model, test_data_loader):
    model.eval()
    with paddle.no_grad():
        test_metrics = run_batch(model, test_data_loader, "test")
        print(f"#### Test results")
        print("f_max: {0:.4f}, auprc: {1:.4f}".format(*test_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument(
        "--test_file",
        type=str,
        default=f"./data/nrPDB-GO_2019.06.18_test.txt",
        help="File containing training protein chains",
    )
    parser.add_argument(
        "--protein_chain_graphs",
        type=str,
        default="./data/chain_graphs",
        help="Path to graph reprsentations of proteins",
    )
    parser.add_argument(
        "--label_data_path",
        type=str,
        required=True,
        help="Mapping containing protein chains with associated labeels",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Mapping containing protein chains with associated labeels",
    )
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    args.activation = F.relu
    task_name = os.path.split(args.label_data_path)[-1]
    task_name = os.path.splitext(task_name)[0]
    args.task = task_name
    if int(args.cuda) == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu:%s" % args.cuda)

    test_chain_list = [p.strip() for p in open(args.test_file)]

    saved_state_dict = paddle.load(args.model_name)
    # In-place assignment
    add_saved_args_and_params(args, saved_state_dict)
    test_dataset = GoTermDataset(
        test_chain_list,
        args.num_angle,
        args.n_channels,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
    )
    test_loader = GoTermDataLoader(test_dataset, batch_size=args.batch_size)

    args.n_labels = test_dataset.n_labels

    model = ProteinSIGN(args)
    model.set_state_dict(saved_state_dict["model"])
    
    model.eval()

    print(f"\n{args.task}: Testing on {len(test_dataset)} protein samples.")
    print(f"Starting  at {datetime.now()}\n")
    print(args)

    test(model, test_loader)
