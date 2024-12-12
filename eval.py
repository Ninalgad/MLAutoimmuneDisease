import numpy as np
from utils import inflate_mask
from tqdm.auto import tqdm


def l2_error(target_vals, preds):
    return float(np.mean((preds - target_vals) ** 2))


def l1_error(target_vals, preds):
    return float(np.abs((preds - target_vals)))


def evaluate_metric(args, val_loader, model, device):

    p, y = [], []
    for batch in val_loader:
        prd_reg = model(batch['img'].to(device).float())['reg']
        prd_reg = prd_reg.detach().cpu().numpy()
        tar_reg = batch['adata'].detach().cpu().numpy()

        p.append(prd_reg)
        y.append(tar_reg)

        if args.debug:
            break
    p, y = np.concatenate(p, axis=0, dtype="float32"), np.concatenate(y, axis=0, dtype="float32")

    res = {
        'overall_l2': l2_error(y, p),
        'y1_l2': l2_error(y[y > 0], p[y > 0]),
        'y0_l2': l2_error(y[y == 0], p[y == 0]),
    }
    return res


def evaluate_metric_conj(args, val_loader, model, device):

    p, y = [], []
    for batch in val_loader:

        prd_reg = model(batch['img'].to(device).float(),
                        batch['mask'].to(device).float()
                        )['reg']
        prd_reg = prd_reg.detach().cpu().numpy()
        tar_reg = batch['adata'].detach().cpu().numpy()

        p.append(prd_reg)
        y.append(tar_reg)

        if args.debug:
            break
    p, y = np.concatenate(p, axis=0, dtype="float32"), np.concatenate(y, axis=0, dtype="float32")

    res = {
        'overall_l2': l2_error(y, p),
        'y1_l2': l2_error(y[y > 0], p[y > 0]),
        'y0_l2': l2_error(y[y == 0], p[y == 0]),
    }
    return res


def offload(tensor):
    return tensor.detach().cpu().numpy()


def evaluate_metric_optimized(args, val_loader, model, device):

    pred_reg, pred_bin, labels = [], [], []
    for batch in val_loader:
        outputs = model(batch['img'].to(device).float())
        outputs = {k: offload(v) for (k, v) in outputs.items()}
        tar = offload(batch['adata'])

        labels.append(tar)
        pred_reg.append(outputs['reg'])
        pred_bin.append(outputs['bin'])

        if args.debug:
            break

    pred_reg = np.concatenate(pred_reg, axis=0, dtype="float32")
    pred_bin = np.concatenate(pred_bin, axis=0, dtype="float32")
    labels = np.concatenate(labels, axis=0, dtype="float32")

    best_pred, best_thresh, best_score = 0, 0, float('inf')
    for t in np.linspace(pred_bin.min(), pred_bin.max(), num=200):
        pred_t = pred_reg * (pred_bin > t).astype('float32')
        score = l2_error(labels, pred_t)
        if score < best_score:
            best_thresh = t
            best_score = score
            best_pred = pred_t

    res = {
        'overall_l2': best_score,
        'threshold': best_thresh,

        'y1_l2': l2_error(labels[labels > 0], best_pred[labels > 0]),
        'y0_l2': l2_error(labels[labels == 0], best_pred[labels == 0]),
    }
    return res
