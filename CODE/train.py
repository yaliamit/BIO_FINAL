import torch
#import pandas as pd
from torch.utils.data import DataLoader
from data import  WindowData
from tqdm import tqdm
from torch import optim
import torch.nn as nn
from utils import save_model, dice_loss, compute_weights, load_model, multiclass_dice_loss, evaluate, evaluate_t
from model.unet import Unet, UnetRes
# extra packages to help with output and time 
import sys 
import time 

# from model.layers import TrippleConv
import argparse
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

    
 
class curr_loss(torch.nn.Module):
    def __init__(self,zero_weight=.2):
        super(curr_loss, self).__init__()
        self.zero_weight=zero_weight
       
        
    def forward(self,x,y):

        acc0,acc1=0.,0.
        
        if x.shape[1]==2 and self.zero_weight>0:
            
            yp=(y>0).type(torch.float)
            loss=torch.mean(yp*(y-x[:,1,:,:])*(y-x[:,1,:,:]))
            e=x[:,0,:,:][:,None,:,:]
            p0=torch.log(1-e)
            p1=torch.log(e)
            l1=-torch.mean(p1*yp+p0*(1-yp)*self.zero_weight)
            loss=+l1
            acc0=torch.sum(torch.logical_and(yp==0,e<.5).type(torch.float))/torch.sum(yp==0)
            acc1=torch.sum(torch.logical_and(yp==1,e>.5).type(torch.float))/torch.sum(yp==1)
        else:
            loss=torch.mean((y-x[:,0,:,:])*(y-x[:,0,:,:]))
        
        
        return loss, (acc0,acc1)


def trainer(device, trainset, validset, model, save_name, epochs, seed=0, weights=None, continuous=1, dice_steps=0,
    batch_size=32, lr=1e-5, weight_decay=1e-8, zero_weight=.2, ignore_index=-100, log=1,
    loss_number = 100, margin = 1, lr_step=100, gamma=.1 ):
    # torch.manual_seed(seed)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=False, drop_last=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size =lr_step, gamma = gamma)
    if continuous==1:
            criterion = nn.MSELoss()  
    elif continuous==2:
            criterion=curr_loss(zero_weight)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        if weights != None:
            criterion.weight = weights.to(device)
    if log:
        ext=''
        if continuous==2:
            ext='_leak_'
        fid = open('./Output/log'+ext+str(device).split(':')[1]+'.txt', 'a')
        losses, dices, iters, accs = [], [], [], []
    #print('lenloader',len(train_loader))
    
    for epoch in range(epochs):
        print('epoch',epoch)
        sys.stdout.flush()
        curr_time = time.time()
        model.train()
        valid_loss, train_loss, train_acc, valid_acc = 0, 0, 0, 0
        trainset.select_window()
        
        if continuous==2:
            train_acc=torch.zeros(2)
            valid_acc=torch.zeros(2)
        ss=0
        
       
        for x, y in train_loader:
  
            x, y=trainset.transform_window(x,y)
            
            x = x.type(torch.float) 
            if continuous: 
                y=  y.type(torch.float)
            else:
                y = y.type(torch.long)
            tp=time.time()

            pred = model(x)
           
            if continuous==1:
                loss = criterion(pred, y)
                if epoch < dice_steps:
                    loss+= dice_loss(pred, y)
            elif continuous==2:
                loss, acc=criterion(pred,y)
                train_acc+=torch.tensor(acc)
            else:
                if weights != None:
                    weights = compute_weights(y, num_classes=len(weights))
                    weights /= weights.sum()
                    criterion.weight = weights
                loss = criterion(pred, y.squeeze(dim=1))
                acc= evaluate_t(y,pred)
                train_acc+=acc

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            
        scheduler.step()

        if epoch % 10==0:
            model.eval()
            acc, total, dice = 0, 0, 0

            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device, dtype=torch.float)
                    if continuous: 
                        y = y.to(device, dtype=torch.float)
                    else:
                        y = y.to(device, dtype=torch.long)
                    pred = model(x)
                    if continuous==1:  
                        valid_loss+=criterion(pred, y)
                        if epoch < dice_steps:
                            valid_loss+= dice_loss(pred, y)
                    elif continuous==2:
                        vl,acc=criterion(pred, y)
                        valid_loss+=vl
                        valid_acc+=torch.tensor(acc)
                    else:
                        valid_loss += criterion(pred,y.squeeze(dim=1))#.cpu().numpy()
                        acc= evaluate_t(y,pred)
                        valid_acc+=acc
                    
               
            fid.write(f'epoch: {epoch+1}, time elapsed: {round(time.time()- curr_time)} seconds \n')
            fid.write(f'epoch {epoch+1}, train loss: {train_loss.item()/len(train_loader):,.4f}, validation loss:  {valid_loss.item()/len(valid_loader):,.4f},learning rate: {get_lr(optimizer):,.7f} \n')
            if continuous==0:
                fid.write(f' train acc: {train_acc.detach().cpu().numpy()/len(train_loader):,.4f}, valid acc: {valid_acc.detach().cpu().numpy()/len(valid_loader):,.4f} \n')             
            if continuous==2 and zero_weight>0:
                ta=train_acc.detach().cpu().numpy()/len(train_loader)
                va=valid_acc.detach().cpu().numpy()/len(valid_loader)
                fid.write(f' train acc: {ta[0]:,.4f},{ta[1]:,.4f} valid acc: {va[0]:,.4f},{va[1]:.4f} \n')
            fid.flush()
            
        if epoch % 50 == 0 and epoch > 0:
            save_model(save_name + '_' + str(epoch) + '.pkl', model)
            