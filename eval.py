import numpy as np
import torch
from tqdm.auto import tqdm


def l2_error(target_vals, preds):
    return float(np.mean((preds - target_vals) ** 2))


def l1_error(target_vals, preds):
    return float(np.abs((preds - target_vals)))


def extract(dataloader, key, debug=False):
    x = []
    for batch in dataloader:
        x.append(offload(batch[key]))
        if debug:
            break
    return np.concatenate(x, axis=0)


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
    y = extract(val_loader, 'adata', debug=args.debug)
    p = predict(val_loader, model, device,
                input_keys=['img'], output_keys=['reg'],
                debug=args.debug)['reg']

    res = {
        'overall_l2': l2_error(y, p),
        'y1_l2': l2_error(y[y > 0], p[y > 0]),
        'y0_l2': l2_error(y[y == 0], p[y == 0]),
    }
    return res


def offload(tensor):
    return tensor.detach().cpu().numpy()


def predict(dataloader, model, device, input_keys=None, output_keys=None, half_precision=False, debug=False):
    outputs = None
    for batch in tqdm(dataloader, total=len(dataloader)):

        if input_keys is not None:
            batch = {k: batch[k] for k in input_keys}
        batch = {k: v.to(device).float() for (k, v) in batch.items()}

        model_out = model(**batch)

        if output_keys is not None:
            model_out = {k: model_out[k] for k in output_keys}

        if half_precision:
            model_out = {k: v.half() for (k, v) in model_out.items()}

        model_out = {k: offload(v) for (k, v) in model_out.items()}

        if not outputs:
            outputs = model_out
        else:
            outputs = {k: np.append(v, model_out[k], axis=0)
                       for (k, v) in outputs.items()}

        if debug:
            break

    return outputs


def predict_features(dataloader, model, device, subset=None, half_precision=False, debug=False):

    outputs = []
    for batch in tqdm(dataloader, total=len(dataloader)):

        feat = model(batch['img'].to(device).float())['features']

        if subset is not None:
            min_, max_ = subset
            feat = feat[:, min_:max_]

        if half_precision:
            feat = feat.half()

        outputs.append(offload(feat))

        if debug:
            break

    return np.concatenate(outputs, axis=0)
