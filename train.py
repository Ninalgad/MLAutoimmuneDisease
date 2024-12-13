import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json
import torchvision
from torch.optim import Adam
from tqdm.notebook import tqdm

from encoder import EncoderNet, ConjoinedNet
from dataset import create_dataloader
from eval import evaluate_metric, evaluate_metric_conj
from sae import PartiallySupervisedSAENet
import utils


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
    def __init__(self, from_logits=False):
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


def segmentation_loss_func(outputs, tar, from_logits=False):
    return DiceBCELoss(from_logits=from_logits)(outputs, tar) + \
           3 * FocalLoss(from_logits=from_logits)(outputs, tar)


def regression_loss_func_l2(outputs, tar):
    return nn.MSELoss()(outputs, tar)


def masked_loss_func_l2(outputs, tar, mask):
    loss_ = (outputs - tar) ** 2
    loss_ = (mask * loss_).sum() / mask.sum()
    return loss_


def regression_loss_func_l1(outputs, tar):
    return nn.L1Loss()(outputs, tar)


def train_encoder(args, train_df, validation_df):
    with open(os.path.join(args.dir_dataset, args.gene_list), 'r') as f:
        genes = json.load(f)['genes']

    train_loader = create_dataloader(
        train_df.patches_path.values, train_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True
    )

    val_loader = create_dataloader(
        validation_df.patches_path.values, validation_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker,
    )

    def train_step_encoder(batch, model, optimizer, device):
        batch = {k: v.to(device).float() for (k, v) in batch.items() if k not in {'coords', 'mask'}}
        optimizer.zero_grad()

        output = model(batch['img'])

        loss_ = regression_loss_func_l2(output['reg'], batch['adata'])

        loss_.backward()
        optimizer.step()

        return loss_.detach().cpu().numpy()

    print('training: EncoderNet efficientnet_v2_s L2Loss')
    model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1').features
    model = EncoderNet(model, 1280)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    return training_loop(args, f'{args.model_name}.pt', train_step_encoder,
                         train_loader, val_loader,
                         model, optimizer, device,
                         evaluate_metric)


def train_conjoined_encoder(args, train_df, validation_df):
    with open(os.path.join(args.dir_dataset, args.gene_list), 'r') as f:
        genes = json.load(f)['genes']

    train_loader = create_dataloader(
        train_df.patches_path.values, train_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True
    )

    val_loader = create_dataloader(
        validation_df.patches_path.values, validation_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker,
    )

    def train_step_encoder(batch, model, optimizer, device):
        batch = {k: v.to(device).float() for (k, v) in batch.items() if k != 'coords'}
        optimizer.zero_grad()

        output = model(batch['img'], batch['mask'])

        loss_ = regression_loss_func_l2(output['reg'], batch['adata'])

        loss_.backward()
        optimizer.step()

        return loss_.detach().cpu().numpy()

    print('training: EncoderNet efficientnet_v2_s L2Loss')
    model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1').features
    model = ConjoinedNet(model, 1280)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    return training_loop(args, f'{args.model_name}.pt', train_step_encoder,
                         train_loader, val_loader,
                         model, optimizer, device,
                         evaluate_metric_conj)


def train_sae(args, train_df, validation_df):
    with open(os.path.join(args.dir_dataset, args.gene_list), 'r') as f:
        genes = json.load(f)['genes']

    train_loader = create_dataloader(
        train_df.patches_path.values, train_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True
    )

    val_loader = create_dataloader(
        validation_df.patches_path.values, validation_df.expr_path.values, genes,
        args.normalize, img_transform=None, size_subset=args.size_subset,
        batch_size=args.batch_size, num_workers=args.num_worker,
    )

    def train_step_encoder(batch, model, optimizer, device):
        batch = {k: v.to(device).float() for (k, v) in batch.items() if k != 'coords'}
        optimizer.zero_grad()

        output = model(batch['img'], batch['mask'])

        sparsity_loss = utils.heaviside_step(output['features'] - 0.01)
        sparsity_loss = sparsity_loss.mean() ** 0.5

        loss_ = regression_loss_func_l2(output['reg'], batch['adata'])
        loss_ += regression_loss_func_l2(output['reconstructed'], output['activations'])
        loss_ += 1e-2 * sparsity_loss

        loss_.backward()
        optimizer.step()

        return loss_.detach().cpu().numpy()

    print('training SAE ConjoinedNet efficientnet_v2_s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.efficientnet_v2_s().features
    model = ConjoinedNet(model, 1280)
    model.to(device)

    model.load_state_dict(torch.load(args.weights_root, device)['model_state_dict'])

    model = PartiallySupervisedSAENet(
        model, activation_output_dim=1280, reg_output_dim=460, n_features=32 * 1280)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    return training_loop(args, f'{args.model_name}-sae.pt', train_step_encoder,
                         train_loader, val_loader,
                         model, optimizer, device,
                         evaluate_metric_conj)


def training_loop(args, weights_save_path, train_step,
                  train_loader, val_loader,
                  model, optimizer, device,
                  evaluation_func,
                  monitor="overall_l2"):
    best_res = {monitor: float('inf')}
    global_step = 0

    steps_per_epoch = len(train_loader)

    num_epochs, val_epoch = args.num_epochs, args.val_epoch
    if args.debug:
        num_epochs, val_epoch = 1, 0

    for epoch in range(num_epochs):
        print(f'Iter: {global_step}, Ep: {epoch}, Current Best: {best_res[monitor]}')
        model.train()
        train_loss = []

        if args.progbar:
            iterator = tqdm(enumerate(train_loader), total=steps_per_epoch)
        else:
            iterator = enumerate(train_loader)

        for i, batch in iterator:
            loss = train_step(batch, model, optimizer, device)

            train_loss.append(loss)
            global_step += 1
            if args.debug:
                break

        if epoch >= val_epoch:
            print(f"Current Best Val loss: {best_res[monitor]}")
            with torch.no_grad():
                res = evaluation_func(args, val_loader, model, device)

            print(f"Val Score: {res}")
            if res[monitor] < best_res[monitor]:
                print(f"{epoch} New best: {res[monitor]} under {best_res[monitor]}")
                best_res = res
                best_res['model_state_dict'] = model.state_dict()
                best_res['global_step'] = global_step
                best_res['epoch'] = epoch
                torch.save(best_res, weights_save_path)
                del best_res['model_state_dict']

    return best_res
