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
from paddle.nn import BCEWithLogitsLoss
from pgl.utils.data import Dataloader
import numpy as np

from models import DeepFRI
from data_preprocessing import MyDataset
from custom_metrics import do_compute_metrics
from utils import print_metrics, get_model_params_state

paddle.seed(123)


def setup_seed(seed):
    # paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def do_compute(model, batch):
    logits = model(*batch[:-1])
    return logits, batch[-1]


def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn):
    total_loss = 0
    logits_list = []
    ground_truth = []

    for batch in tqdm(data_loader, desc=f"{desc} Epoch {epoch_i}"):
        logits, labels = do_compute(model, batch)

        loss = loss_fn(logits, labels)
        loss = paddle.mean(paddle.sum(loss, -1))
        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        total_loss += loss.item()

        logits_list.append(F.sigmoid(logits).tolist())
        ground_truth.append(labels.tolist())

    total_loss /= len(data_loader)

    logits_list = np.concatenate(logits_list)
    ground_truth = np.concatenate(ground_truth)

    metrics = None
    if not model.training:
        metrics = do_compute_metrics(ground_truth, logits_list)

    return total_loss, metrics


def train(
    model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, model_name
):
    best_auprc = -1
    for epoch_i in range(1, n_epochs + 1):

        start = time.time()
        model.train()
        ## Training
        train_loss, train_metrics = run_batch(
            model, optimizer, train_data_loader, epoch_i, "train", loss_fn
        )

        model.eval()
        with paddle.no_grad():

            ## Validation
            if val_data_loader:
                val_loss, val_metrics = run_batch(
                    model, optimizer, val_data_loader, epoch_i, "val", loss_fn
                )
                if best_auprc < val_metrics[1]:
                    current_sate = get_model_params_state(
                        model, args, epoch_i, *val_metrics
                    )
                    paddle.save(current_sate, f"{model_name}.pdparams")
                    best_auprc = val_metrics[1]

        if train_data_loader:
            print(f"\n#### Epoch {epoch_i} time {time.time() - start:.4f}s")
            print_metrics(train_loss, 0, 0)

        if val_data_loader:
            print(f"#### Validation epoch {epoch_i}")
            print_metrics(val_loss, *val_metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cuda", type=str, default="0", help="GPU ID to train on.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "-gcd",
        "--gc_dims",
        type=int,
        default=[512, 512, 512],
        nargs="+",
        help="Dimensions of GraphConv layers.",
    )
    parser.add_argument(
        "-fcd",
        "--fc_dims",
        type=int,
        default=[1024],
        nargs="+",
        help="Dimensions of fully connected layers (after GraphConv layers).",
    )
    parser.add_argument(
        "-drop", "--drop", type=float, default=0.3, help="Dropout rate."
    )
    parser.add_argument(
        "-l2",
        "--weight_decay",
        type=float,
        default=2e-5,
        help="L2 regularization coefficient.",
    )
    parser.add_argument("-lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument(
        "-gc",
        "--gc_layer",
        type=str,
        default="GraphConv",
        choices=["GraphConv", "SAGEConv", "GAT"],
        help="Graph Conv layer.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=200, help="Number of epochs to train."
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "-pd",
        "--pad_len",
        type=int,
        default=1000,
        help="Padd length (max len of protein sequences in train set).",
    )
    parser.add_argument(
        "-lm",
        "--lm_model_name",
        type=str,
        help="Path to the pre-trained LSTM-Language Model.",
    )
    parser.add_argument(
        "-lm_dim",
        "--lm_dim",
        type=int,
        default=1024,
        help="Output dimension of the pre-trained Language Model.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/nrPDB-GO_2019.06.18_train.txt",
        help="File with list of protein chains for training.",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="data/nrPDB-GO_2019.06.18_valid.txt",
        help="File with list of protein chains for validation.",
    )
    parser.add_argument(
        "--protein_chain_graphs",
        type=str,
        default="data/chain_graphs",
        help="Path to graph reprsentations of proteins.",
    )
    parser.add_argument(
        "--label_data_path",
        type=str,
        default="data/labels/molecular_function.npz",
        help="Mapping containing protein chains with associated their labels. "
        "Choose from [molecular_function.npz, cellular_component.npz, biological_process.npz]",
    )
    parser.add_argument(
        "--cmap_thresh",
        type=int,
        default=10,
        help="Distance (in armstrong) threshold for concat map construction.",
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=26,
        help="Total number of distinct amino acids symbols.",
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
    if args.seed:
        setup_seed(args.seed)

    if int(args.cuda) == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu:%s" % args.cuda)

    train_chain_list = [p.strip() for p in open(args.train_file)]
    valid_chain_list = [p.strip() for p in open(args.valid_file)]

    train_dataset = MyDataset(
        train_chain_list,
        args.n_channels,
        args.pad_len,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
        args.use_cache,
    )
    valid_dataset = MyDataset(
        valid_chain_list,
        args.n_channels,
        args.pad_len,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
        args.use_cache,
    )

    train_loader = Dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = Dataloader(
        valid_dataset, batch_size=args.batch_size, collate_fn=valid_dataset.collate_fn
    )

    args.n_labels = train_dataset.n_labels
    model = DeepFRI(args)
    task_name = os.path.split(args.label_data_path)[-1]
    task_name = os.path.splitext(task_name)[0]
    args.task = task_name
    time_stamp = str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    args.model_name = (
        f"models/{model.__class__.__name__}_{args.gc_layer}_{args.task}_{time_stamp}"
    )

    loss_fn = BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(
        parameters=model.parameters(),
        learning_rate=args.lr,
        beta1=0.95,
        beta2=0.99,
        weight_decay=args.weight_decay,
    )

    model_save_dir = os.path.split(args.model_name)[0]
    if model_save_dir:
        try:
            os.makedirs(model_save_dir)
        except FileExistsError:
            pass
    print(
        f"\n{args.task}: Training on {len(train_dataset)} protein samples and {len(valid_dataset)} for validation."
    )
    print(f"Starting  at {datetime.now()}\n")
    print(args)

    train(
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        args.epochs,
        args.model_name,
    )
