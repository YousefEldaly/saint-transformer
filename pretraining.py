import torch
from torch import nn
from data_openml import DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np

def SAINT_pretrain(model, cat_idxs, X_train, y_train, continuous_mean_std, opt, device, 
                   save_freq=10, patience=10):
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    pt_aug_dict = {'noise_type': opt.pt_aug, 'lambda': opt.pt_aug_lam}
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()

    best_loss = float('inf')
    epochs_no_improve = 0
    log_path = os.path.join(os.getcwd(), 'pretrain_log.txt')

    with open(log_path, 'w') as f_log:
        f_log.write("Pretraining begins!\n")
    print("Pretraining begins!")

    for epoch in range(opt.pretrain_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, _, cat_mask, con_mask = [d.to(device) for d in data[:5]]

            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ, x_cont, noise_params=pt_aug_dict)
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask, model, vision_dset)
            else:
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)

            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2, lam=opt.mixup_lam)

            loss = 0

            if 'contrastive' in opt.pt_tasks:
                aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug
