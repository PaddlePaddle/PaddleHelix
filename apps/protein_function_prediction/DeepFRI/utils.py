import os


def print_metrics(loss, f_max, auprc):
    print(f"loss: {loss:.4f}, f_max: {f_max:.4f}, auprc: {auprc:.4f}")


def get_model_params_state(model, args, epoch, f_max, auprc):
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "gc_dims": args.gc_dims,
        "fc_dims": args.fc_dims,
        "drop": args.drop,
        "gc_layer": args.gc_layer,
        "pad_len": args.pad_len,
        "lm_dim": args.lm_dim,
        "n_channels": args.n_channels,
        "cmap_thresh": args.cmap_thresh,
        "task": args.task,
        "n_labels": args.n_labels,
        "val_f_max": f"{f_max:.4f}",
        "val_auprc": f"{auprc:.4f}",
    }


def add_saved_args_and_params(args, state_dict):
    args.gc_dims = state_dict["gc_dims"]
    args.fc_dims = state_dict["fc_dims"]
    args.drop = state_dict["drop"]
    args.gc_layer = state_dict["gc_layer"]
    args.pad_len = state_dict["pad_len"]
    args.lm_dim = state_dict["lm_dim"]
    args.n_channels = state_dict["n_channels"]
    args.cmap_thresh = state_dict["cmap_thresh"]
    args.task = state_dict["task"]
    args.n_labels = state_dict["n_labels"]

    return None  # In-place modification of args
