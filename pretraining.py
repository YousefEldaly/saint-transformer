import torch
from torch import nn
import copy
import time

from baselines.data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
from torch.utils.data import DataLoader
import torch.optim as optim
from augmentations import embed_data_mask
from augmentations import add_noise

import os
import numpy as np

def SAINT_pretrain(model,cat_idxs,X_train,y_train,continuous_mean_std,opt,device):
    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,opt.dtask, continuous_mean_std)
    trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)
    vision_dset = opt.vision_dset
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    # Create log file
    log_file = open(f"pretraining_log_{time.strftime('%Y%m%d-%H%M%S')}.txt", "w")
    log_file.write("Epoch\tLoss\n")
    
    # Early stopping initialization
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 10
    patience_counter = 0
    early_stop = False
    
    print("Pretraining begins!")
    for epoch in range(opt.pretrain_epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
            
            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in opt.pt_aug:
                from augmentations import add_noise
                x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
            else:
                _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            
            if 'mixup' in opt.pt_aug:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)
            loss = 0
            if 'contrastive' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                if opt.pt_projhead_style == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif opt.pt_projhead_style == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion1(logits_per_aug1, targets)
                loss_2 = criterion1(logits_per_aug2, targets)
                loss   = opt.lam0*(loss_1 + loss_2)/2
            elif 'contrastive_sim' in opt.pt_tasks:
                aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()
                loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
            if 'denoising' in opt.pt_tasks:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs =  torch.cat(con_outs,dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                l1 = 0
                n_cat = x_categ.shape[-1]
                for j in range(1,n_cat):
                    l1+= criterion1(cat_outs[j],x_categ[:,j])
                loss += opt.lam2*l1 + opt.lam3*l2    
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch: {epoch}, Running Loss: {epoch_loss}')
        
        # Log to file
        log_file.write(f"{epoch}\t{epoch_loss}\n")
        log_file.flush()  # Ensure immediate write
        
        # Check for best model and early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'best_model_epoch_{epoch}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stop = True
                # Save final checkpoint before early stop
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, f'last_model_epoch_{epoch}.pth')
                
        # Save latest model every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, 'latest_model.pth')

    # End of training
    log_file.close()
    print('END OF PRETRAINING!')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model
        # if opt.active_log:
        #     wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
        #     })
