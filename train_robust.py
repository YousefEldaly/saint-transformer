import torch
from torch import nn
from models import SAINT, SAINT_vision

from data_openml import data_prep_openml, task_dset_ids, DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask, add_noise

import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action='store_true')
parser.add_argument('--task', required=True, type=str, choices=['binary', 'multiclass', 'regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP', 'Noemb', 'pos_singleMLP'])
parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=6, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--attention_dropout', default=0.1, type=float)
parser.add_argument('--ff_dropout', default=0.1, type=float)
parser.add_argument('--attentiontype', default='colrow', type=str, choices=['col', 'colrow', 'row', 'justmlp', 'attn', 'attnmlp'])
parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW', 'Adam', 'SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'linear'])
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='testrun', type=str)
parser.add_argument('--set_seed', default=1, type=int)
parser.add_argument('--dset_seed', default=1, type=int)
parser.add_argument('--active_log', action='store_true')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_epochs', default=50, type=int)
parser.add_argument('--pt_tasks', default=['contrastive', 'denoising'], type=str, nargs='*', choices=['contrastive', 'contrastive_sim', 'denoising'])
parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup', 'cutmix'])
parser.add_argument('--pt_aug_lam', default=0.1, type=float)
parser.add_argument('--mixup_lam', default=0.3, type=float)
parser.add_argument('--train_noise_type', default=None, type=str, choices=['missing', 'cutmix'])
parser.add_argument('--train_noise_level', default=0, type=float)
parser.add_argument('--ssl_samples', default=None, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str, choices=['diff', 'same', 'nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)
parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common', 'sep'])
parser.add_argument('--dataset_path', type=str, default='./custom_data', help='Path to custom dataset directory')

opt = parser.parse_args()

# Set up save path
modelsave_path = os.path.join(os.getcwd(), opt.savemodelroot, opt.task, str(opt.dset_id), opt.run_name)
os.makedirs(modelsave_path, exist_ok=True)

# Task label
opt.dtask = 'reg' if opt.task == 'regression' else 'clf'

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")
torch.manual_seed(opt.set_seed)

# Download / load dataset
print('Downloading and processing the dataset, it might take some time.')
if opt.dset_id == -1:
    data = data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2], dataset_path=opt.dataset_path)
else:
    data = data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2])

cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data
if opt.dset_id == -1:
    con_idxs = list(range(X_train['data'].shape[1]))
    cat_idxs = []
    cat_dims = np.array([])

continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

# Adjust hyperparameters for large feature sets
_, nfeat = X_train['data'].shape
if nfeat > 100:
    opt.embedding_size = min(4, opt.embedding_size)
    opt.batchsize = min(64, opt.batchsize)
if opt.attentiontype != 'col':
    opt.transformer_depth = 1
    opt.attention_heads = 4
    opt.attention_dropout = 0.8
    opt.embedding_size = 16
    if opt.optimizer == 'SGD':
        opt.ff_dropout = 0.4
        opt.lr = 0.01
    else:
        opt.ff_dropout = 0.8

# Dataset + loaders
train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

# Model
if opt.dset_id == -1:
    cat_dims = np.array([1])
else:
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

y_dim = 1 if opt.task == 'regression' else len(np.unique(y_train['data'][:, 0]))
model = SAINT(categories=tuple(cat_dims), num_continuous=len(con_idxs), dim=opt.embedding_size,
              dim_out=1, depth=opt.transformer_depth, heads=opt.attention_heads, attn_dropout=opt.attention_dropout,
              ff_dropout=opt.ff_dropout, mlp_hidden_mults=(4, 2), cont_embeddings=opt.cont_embeddings,
              attentiontype=opt.attentiontype, final_mlp_style=opt.final_mlp_style, y_dim=y_dim)
model.to(device)

# Criterion
if y_dim == 2 and opt.task == 'binary':
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and opt.task == 'multiclass':
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise 'Unsupported task'

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=opt.lr)

# Pretraining (if applicable)
if opt.pretrain:
    from pretraining import SAINT_pretrain
    model = SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device)

# Fine-tuning loop with checkpointing and early stopping
save_freq = 5
patience = 5
epochs_no_improve = 0
best_valid_metric = float('inf') if opt.task == 'regression' else 0
log_path = os.path.join(modelsave_path, 'finetune_log.txt')

with open(log_path, 'w') as f_log:
    f_log.write("Fine-tuning begins!\n")

print("Training begins now.")
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for data in trainloader:
        optimizer.zero_grad()
        x_categ, x_cont, y_gts, cat_mask, con_mask = (d.to(device) for d in data)
        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, opt.vision_dset)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)

        loss = criterion(y_outs, y_gts if opt.task == 'regression' else y_gts.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)

    # Save checkpoints
    if (epoch + 1) % save_freq == 0:
        torch.save(model.state_dict(), f"{modelsave_path}/checkpoint_epoch{epoch+1}.pth")

    # Evaluation
    model.eval()
    with torch.no_grad():
        if opt.task == 'regression':
            valid_metric = mean_sq_error(model, validloader, device, opt.vision_dset)
        else:
            acc, auroc = classification_scores(model, validloader, device, opt.task, opt.vision_dset)
            valid_metric = auroc if opt.task == 'binary' else acc

        if (opt.task == 'regression' and valid_metric < best_valid_metric) or \
           (opt.task != 'regression' and valid_metric > best_valid_metric):
            best_valid_metric = valid_metric
            torch.save(model.state_dict(), f"{modelsave_path}/bestmodel.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    # Log
    with open(log_path, 'a') as f_log:
        f_log.write(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Valid Metric={valid_metric:.4f}\n")

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Valid Metric={valid_metric:.4f}")

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        with open(log_path, 'a') as f_log:
            f_log.write(f"Early stopping at epoch {epoch+1}\n")
        break

print("Training complete.")
