import numpy as np


def l2_error(target_vals, preds):
    return float(np.mean((preds - target_vals) ** 2))


def l1_error(target_vals, preds):
    return float(np.abs((preds - target_vals)))


def evaluate_metric(args, val_loader, model, device):

    p, y = [], []
    for batch in val_loader:
        prd_reg = model(batch['img'].to(device).float())['reg']
        prd_reg = offload(prd_reg)
        tar_reg = offload(batch['adata'])

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
        prd_reg = offload(prd_reg)
        tar_reg = offload(batch['adata'])

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
