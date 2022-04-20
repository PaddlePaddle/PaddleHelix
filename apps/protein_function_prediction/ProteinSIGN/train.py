import os
import time
import argparse
import random
from datetime import datetime
from tqdm import tqdm

import numpy as np
import paddle

paddle.disable_static()
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.nn import BCEWithLogitsLoss

from dataset import GoTermDataset, GoTermDataLoader
from model import ProteinSIGN
from custom_metrics import do_compute_metrics
from utils import get_model_params_state, print_metrics


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
    # data_source = '/home/arnold/Implementations/PROT-function-prediction'
    data_source = "."
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--train_file",
        type=str,
        default="./data/nrPDB-GO_2019.06.18_train.txt",
        help="File containing training protein chains.",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="./data/nrPDB-GO_2019.06.18_valid.txt",
        help="File containing validation protein chains.",
    )
    parser.add_argument(
        "--protein_chain_graphs",
        type=str,
        default="./data/chain_graphs",
        help="Path to graph reprsentations of proteins.",
    )
    parser.add_argument(
        "--label_data_path",
        type=str,
        default="./data/labels/molecular_function.npz",
        help="Mapping containing protein chains with associated labeels.",
    )
    parser.add_argument("--feat_drop", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument(
        "--n_channels",
        type=int,
        default=26,
        help="Number of amino acids (residues) symbols.",
    )
    parser.add_argument("--dense_dims", type=int, default=[512], nargs="+")
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument(
        "--cmap_thresh",
        type=int,
        default=10,
        help="Distance (in armstrong) threshold for concat map construction.",
    )
    parser.add_argument(
        "--num_angle", type=int, default=4, help="Number of angle domains."
    )
    parser.add_argument(
        "--merge_e2e",
        type=str,
        default="cat",
        help="How to merge output from edge to edge layer.",
    )
    parser.add_argument(
        "--merge_e2n",
        type=str,
        default="mean",
        help="How to merge output from edge to node layer.",
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to save protein graph in memory for fast reading.",
    )

    args = parser.parse_args()
    args.activation = F.relu
    if args.seed:
        setup_seed(args.seed)

    args.use_cache = bool(args.use_cache)

    if int(args.cuda) == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu:%s" % args.cuda)

    train_chain_list = [p.strip() for p in open(args.train_file)]
    valid_chain_list = [p.strip() for p in open(args.valid_file)]
    train_dataset = GoTermDataset(
        train_chain_list,
        args.num_angle,
        args.n_channels,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
        use_cache=args.use_cache,
    )
    valid_dataset = GoTermDataset(
        valid_chain_list,
        args.num_angle,
        args.n_channels,
        args.protein_chain_graphs,
        args.cmap_thresh,
        args.label_data_path,
        use_cache=args.use_cache,
    )

    train_loader = GoTermDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = GoTermDataLoader(valid_dataset, batch_size=args.batch_size)
    args.n_labels = train_dataset.n_labels

    model = ProteinSIGN(args)
    task_name = os.path.split(args.label_data_path)[-1]
    task_name = os.path.splitext(task_name)[0]
    args.task = task_name
    time_stamp = str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    args.model_name = f"models/{model.__class__.__name__}_{args.task}_{time_stamp}"

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
