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
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                logits_per_aug1 = aug_features_1 @ aug_features_2.t() / opt.nce_temp
                logits_per_aug2 = aug_features_2 @ aug_features_1.t() / opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss = opt.lam0 * (loss_1 + loss_2) / 2

            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs, dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                n_cat = x_categ.shape[-1]
                for j in range(1, n_cat):
                    l1 += criterion1(cat_outs[j], x_categ[:, j])
                loss += opt.lam2 * l1 + opt.lam3 * l2

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        log_line = f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}\n"
        print(log_line.strip())
        with open(log_path, 'a') as f_log:
            f_log.write(log_line)

        # Save checkpoint every save_freq epochs
        if (epoch + 1) % save_freq == 0:
            ckpt_path = f"pretrain_ckpt_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # Save best model
        if avg_loss < best_loss - 1e-4:  # consider significant improvement
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "pretrain_best_model.pt")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            with open(log_path, 'a') as f_log:
                f_log.write(f"Early stopping at epoch {epoch+1}\n")
            break

    print("END OF PRETRAINING!")
    with open(log_path, 'a') as f_log:
        f_log.write("END OF PRETRAINING!\n")

    return model
        # if opt.active_log:
        #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
        #     })
