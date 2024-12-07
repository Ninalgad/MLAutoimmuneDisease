import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json
from torch.optim import Adam
from tqdm.notebook import tqdm

from model import SimpNet
from dataset import create_dataloader
from eval import evaluate_metric


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceBCELoss(nn.Module):
    def __init__(self, from_logits=False, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, smooth=0.1):
        targets = targets.float()

        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return bce_loss + dice_loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class FocalLoss(nn.Module):
    def __init__(self, from_logits=False, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.from_logits = from_logits

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        if self.from_logits:
            inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce

        return focal_loss


def segmentation_loss_func(outputs, tar):
    return DiceBCELoss(from_logits=True)(outputs, tar) + \
           3 * FocalLoss(from_logits=True)(outputs, tar)


def regression_loss_func(outputs, tar):
    return nn.MSELoss()(outputs, tar)


def train_step(batch, model, optimizer, device):
    batch = {k: v.to(device).float() for (k, v) in batch.items() if k != 'coords'}
    optimizer.zero_grad()

    prd_seg, prd_reg = model(batch['img'])

    loss_ = segmentation_loss_func(prd_seg, batch['mask']) + regression_loss_func(prd_reg, batch['adata'])

    loss_.backward()
    optimizer.step()

    return loss_.detach().cpu().numpy()


def train_finetune(args, train_df, validation_df):
    with open(os.path.join(args.dir_dataset, args.gene_list), 'r') as f:
        genes = json.load(f)['genes']

    train_loader = create_dataloader(
        train_df.patches_path.values, train_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker, training=True
    )

    print('training')
    weights = None
    if args.pretrained:
        weights = 'IMAGENET1K_V1'
    model = SimpNet(weights=weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_ep = 0
    global_step = 0
    best_val = float('inf')

    steps_per_epoch = len(train_loader)

    num_epochs, val_epoch = args.num_epochs, args.val_epoch
    if args.debug:
        num_epochs, val_epoch = 1, 0

    for epoch in range(num_epochs):
        print(
            f'Iter: {global_step}, Ep: {global_step / steps_per_epoch}, '
            f'Current Best: {best_val}, Best Ep: {best_ep}')
        model.train()
        train_loss = []

        for i, batch in tqdm(enumerate(train_loader), total=steps_per_epoch):
            loss = train_step(batch, model, optimizer, device)

            train_loss.append(loss)
            global_step += 1
            if args.debug:
                break

        if epoch >= val_epoch:
            print(f"Current Best Val loss: {best_val}, Best Ep: {best_ep}\n")
            with torch.no_grad():
                val = evaluate_metric(args, validation_df, genes, model, device)

            print(f"Val: {val}")
            if val < best_val:
                print(f"{epoch} New best l2: {val} under {best_val}")
                best_val = val
                best_ep = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_val': best_val,
                }, f'{args.model_name}.pt')

    return best_val
