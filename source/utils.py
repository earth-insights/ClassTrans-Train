import os
import torch
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# mpl.use("Agg")
# plt.ioff()


def progress(train_logs, valid_logs, loss_nm, metric_nm, nepochs, outdir, fn_out):
    loss_t = [dic[loss_nm] for dic in train_logs]
    loss_v = [dic[loss_nm] for dic in valid_logs]
    score_t = [dic[metric_nm] for dic in train_logs]
    score_v = [dic[metric_nm] for dic in valid_logs]

    epochs = range(0, len(score_t))
    plt.figure(figsize=(12, 5))

    # Train and validation metric
    # ---------------------------
    plt.subplot(1, 2, 1)

    idx = np.nonzero(score_t == max(score_t))[0][0]
    label = f"Train, {metric_nm}={max(score_t):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_t, "b", label=label)

    idx = np.nonzero(score_v == max(score_v))[0][0]
    label = f"Valid, {metric_nm}={max(score_v):6.4f} in Epoch={idx}"
    plt.plot(epochs, score_v, "r", label=label)

    plt.title("Training and Validation Metric")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel(metric_nm)
    plt.ylim(0, 1)
    plt.legend()

    # Train and validation loss
    # -------------------------
    plt.subplot(1, 2, 2)
    ymax = max(max(loss_t), max(loss_v))
    ymin = min(min(loss_t), min(loss_v))
    ymax = 1 if ymax <= 1 else ymax + 0.5
    ymin = 0 if ymin <= 0.5 else ymin - 0.5

    idx = np.nonzero(loss_t == min(loss_t))[0][0]
    label = f"Train {loss_nm}={min(loss_t):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_t, "b", label=label)

    idx = np.nonzero(loss_v == min(loss_v))[0][0]
    label = f"Valid {loss_nm}={min(loss_v):6.4f} in Epoch:{idx}"
    plt.plot(epochs, loss_v, "r", label=label)

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.xlim(0, nepochs)
    plt.ylabel("Loss")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig(f"{outdir}/{fn_out}.png", bbox_inches="tight")
    plt.clf()
    plt.close()

    return


def setup_seed(seed, return_old_state=False):
    old_state = list()
    if return_old_state:
        old_state.append(random.getstate())
        old_state.append(np.random.get_state())
        old_state.append(torch.get_rng_state())
        old_state.append(torch.cuda.get_rng_state())
        old_state.append(torch.cuda.get_rng_state_all())
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return old_state


def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.round(W * cut_rat)
    cut_h = np.round(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W).astype(np.uint8)
    bby1 = np.clip(cy - cut_h // 2, 0, H).astype(np.uint8)
    bbx2 = np.clip(cx + cut_w // 2, 0, W).astype(np.uint8)
    bby2 = np.clip(cy + cut_h // 2, 0, H).astype(np.uint8)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, prob=0.5):
    if random.random() < prob:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        rand_index = torch.randperm(x.size()[0]).to(x.device)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        y[:, bbx1:bbx2, bby1:bby2] = y[rand_index, bbx1:bbx2, bby1:bby2]
    return x, y