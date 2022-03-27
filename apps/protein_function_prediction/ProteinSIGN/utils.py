import os


def print_metrics(loss, f_max, auprc):
    print(f"loss: {loss:.4f}, f_max: {f_max:.4f}, auprc: {auprc:.4f}")


def get_model_params_state(model, args, epoch, f_max, auprc):
    return {
        "model": model.state_dict(),
        "epoch": epoch,
        "num_convs": args.num_convs,
        "dense_dims": args.dense_dims,
        "feat_drop": args.feat_drop,
        "hidden_dim": args.hidden_dim,
        "n_channels": args.n_channels,
        "cmap_thresh": args.cmap_thresh,
        "task": args.task,
        "n_labels": args.n_labels,
        "num_heads": args.num_heads,
        "num_angle": args.num_angle,
        "merge_e2e": args.merge_e2e,
        "merge_e2n": args.merge_e2n,
        "val_f_max": f"{f_max:.4f}",
        "val_auprc": f"{auprc:.4f}",
    }


def add_saved_args_and_params(args, state_dict):
    args.num_convs = state_dict["num_convs"]
    args.dense_dims = state_dict["dense_dims"]
    args.feat_drop = state_dict["feat_drop"]
    args.hidden_dim = state_dict["hidden_dim"]
    args.num_heads = state_dict["num_heads"]
    args.num_angle = state_dict["num_angle"]
    args.merge_e2e = state_dict["merge_e2e"]
    args.merge_e2n = state_dict["merge_e2n"]
    args.n_channels = state_dict["n_channels"]
    args.cmap_thresh = state_dict["cmap_thresh"]
    args.task = state_dict["task"]
    args.n_labels = state_dict["n_labels"]

    return None  # In-place modification of args
