import os
import random
import argparse
import time
from datetime import datetime
from tqdm import tqdm

import paddle

paddle.disable_static()
import paddle.nn.functional as F
import paddle.optimizer as optim
from pgl.utils.data import Dataloader
import numpy as np

from models import DeepFRI
from data_preprocessing import MyDataset
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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cuda", type=str, default="0", help="GPU ID to train on.")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/nrPDB-GO_2019.06.18_test.txt",
        help="File with list of protein chains for training.",
    )

    parser.add_argument(
        "--protein_chain_graphs",
        type=str,
        default="data/chain_graphs",
        help="Path to graph reprsentations of proteins.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to saved/trained methods with parameters.",
    )
    parser.add_argument(
        "--label_data_path",
        type=str,
        required=True,
        help="Mapping containing protein chains with associated their labels. Choose from [molecular_function.npz, cellular_component.npz, biological_process.npz]",
    )
    parser.add_argument(
        "-lm",
        "--lm_model_name",
        type=str,
        help="Path to the pre-trained LSTM-Language Model.",
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to save protein graph in memory for fast reading.",
    )
    args = parser.parse_args()

    args.use_cache = bool(args.use_cache)

    if int(args.cuda) == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu:%s" % args.cuda)

    test_chain_list = [p.strip() for p in open(args.test_file)]

    saved_state_dict = paddle.load(args.model_name)
    # In-place assignment
    add_saved_args_and_params(args, saved_state_dict)
    test_dataset = MyDataset(
        test_chain_list,
        args.n_channels,
        args.pad_len,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
        args.use_cache,
    )

    test_loader = Dataloader(
        test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn
    )

    args.n_labels = test_dataset.n_labels
    model = DeepFRI(args)
    model.set_state_dict(saved_state_dict["model"])
    model.eval()

    print(f"\n{args.task}: Testing on {len(test_dataset)} protein samples.")
    print(f"Starting  at {datetime.now()}\n")
    print(args)

    test(model, test_loader)
