import numpy as np
from tqdm.auto import tqdm
from dataset import create_dataloader


def l2_error(target_vals, preds):
    return float(np.mean((preds - target_vals) ** 2))


def evaluate_metric(args, val_df, genes, model, device):
    val_loader = create_dataloader(
        val_df.patches_path.values, val_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker, training=True,
    )
    errors = []
    for batch in tqdm(val_loader):
        _, prd_reg = model(batch['img'].to(device).float())
        prd_reg = prd_reg.detach().cpu().numpy()
        tar_reg = batch['adata'].detach().cpu().numpy()
        errors.append(l2_error(tar_reg, prd_reg))

        if args.debug:
            break

    return np.mean(errors)
