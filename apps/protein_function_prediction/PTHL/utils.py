import os

import paddle


def print_metrics(loss, f_max, auprc):
    print(f"loss: {loss:.4f}, f_max: {f_max:.4f}, auprc: {auprc:.4f}")


def get_model_params_state(model, args, epoch, f_max, auprc):
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "n_h_dim": args.n_h_dim,
        "s_h_dim": args.s_h_dim,
        "v_h_dim": args.v_h_dim,
        "v_n_feats": args.v_n_feats,
        "e_s_h_dim": args.e_s_h_dim,
        "e_v_h_dim": args.e_v_h_dim,
        "e_v_n_feats": args.e_v_n_feats,
        "n_blocks": args.n_blocks,
        "drop": args.drop,
        "fc_dims": args.fc_dims,
        "n_channels": args.n_channels,
        "cmap_thresh": args.cmap_thresh,
        "task": args.task,
        "n_labels": args.n_labels,
        "lr": args.lr,
        "val_f_max": f"{f_max:.4f}",
        "val_auprc": f"{auprc:.4f}",
    }

def add_saved_args_and_params(args, state_dict):
    args.n_blocks = state_dict["n_blocks"]
    args.fc_dims = state_dict["fc_dims"]
    args.drop = state_dict["drop"]
    args.n_h_dim = state_dict["n_h_dim"]
    args.s_h_dim = state_dict["s_h_dim"]
    args.v_h_dim = state_dict["v_h_dim"]
    args.v_n_feats = state_dict["v_n_feats"]
    args.e_s_h_dim = state_dict["e_s_h_dim"]
    args.e_v_h_dim = state_dict["e_v_h_dim"]
    args.e_v_n_feats = state_dict["e_v_n_feats"]
    args.n_channels = state_dict["n_channels"]
    args.cmap_thresh = state_dict["cmap_thresh"]
    args.task = state_dict["task"]
    args.n_labels = state_dict["n_labels"]
    args.lr = state_dict["lr"]

    return None  # In-place modification of args


def _norm_no_nan(x, axis=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = paddle.clip(paddle.sum(paddle.square(x), axis=axis, keepdim=keepdim), min=eps)
    return paddle.sqrt(out) if sqrt else out
