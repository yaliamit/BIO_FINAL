import numpy as np
#import pandas as pd
import os
import shutil
import pickle
import re
import torch
import cv2
import copy
from skimage import exposure
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import matplotlib as mpl
from matplotlib.patches import Patch



def get_file_numbers(dir):

    ll=os.listdir(dir)
    ff=[]
    for f in ll:
        s=re.findall(r'\d+',f) 
        if len(s)>0:
           
            ff+=[int(s[0])]

    return(ff)
    
def get_file_by_num(dir,i):
    ll=os.listdir(dir)
    ff=[]
    for f in ll:
        s=re.findall(r'\d+',f)
        if len(s)>0:
            iia=int(s[0])
        else:
            continue
        if iia==i:
            ff+=[f]

    return(ff)


colors=[[1,0,0],[1,1,0],[0,1,0],[0,0,1]]
def color_image(imt):
    llt=5
    #print(np.unique(imt))
    bimt=np.zeros((imt.shape[0],imt.shape[1],3))
    for i in range(1,llt):
        j=i-1
        #print(i,j,colors[j])
        bimt[imt==i]=colors[j]
    return bimt

def dice_loss(input, target, epsilon=1e-6):
    sum_dim = (-1, -2)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = (input).sum(dim=sum_dim) + (target).sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return 1 - dice.mean()

def multiclass_dice_loss(input, target, epsilon=1e-6):
    sum_dim = (-1, -2)
    input[target==0] = 0
    target_one_hot = F.one_hot(target, 5)
    input_one_hot = F.one_hot(input, 5)
    dices = []
    for i in range(1,5):
        inter = 2 * (input_one_hot[:,:,:,i] * target_one_hot[:,:,:,i]).sum(dim=sum_dim)
        sets_sum = (input_one_hot[:,:,:,i]).sum(dim=sum_dim) + (target_one_hot[:,:,:,i]).sum(dim=sum_dim)
        # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + epsilon) / (sets_sum + epsilon)
        dices.append(1-dice.mean())
    return torch.mean(torch.stack(dices)).item()

def dice_loss_cont(input, target):
    sum_dim = (-1, -2)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = (input**2).sum(dim=sum_dim) + (target**2).sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + 1e-6) / (sets_sum + 1e-6)
    return 1 - dice.mean()


def stretch_image(img,low=2):
  p2, p98 = np.percentile(img, (low, 100-low))
  img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
  return(img_rescale)

def save_model(path, model):
    with open(path, 'wb') as file:  
        pickle.dump(model, file)

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def compute_weights(y, num_classes=5):
    counts = torch.bincount(y.flatten(), minlength=num_classes)
    probs=counts[1:]/torch.sum(counts[1:])
    weights = 1.0 / (counts.float() + 1e-9)
    return weights

def augment_background_thick(img,bdy,mrg):
    
    new = np.zeros_like(img)
    #new[img==1] = 255
    #new[img==4] = 255
    new[img==2] = 255
    new[img==3] = 255
    new_img = copy.deepcopy(img)
    edges = cv2.Canny(new.astype(np.uint8), 100, 200)
   
    xs, ys = np.where(edges == 255)
    
    mask_tot=np.zeros_like(img, dtype=bool)
    for i in range(mrg):
       
        mask = np.zeros_like(img, dtype=bool)  # Temporary mask to mark new updates
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_xs = np.clip(xs + dx, 0, img.shape[0] - 1)
            shifted_ys = np.clip(ys + dy, 0, img.shape[1] - 1)
            valid_updates = (img[shifted_xs, shifted_ys] == 0) & (~mask[shifted_xs, shifted_ys]) #& (~mask_tot[shifted_xs, shifted_ys])
           
            mask[shifted_xs[valid_updates], shifted_ys[valid_updates]] = True
        if i>bdy:
            new_img[mask] = 4 
        mask_tot[mask]=True
       
        xs, ys = np.where(mask)
        
    return new_img


def augment_background(img):
    new = np.zeros_like(img)
    new[img==1] = 255
    new[img==4] = 255
    # new[img==3] = 255
    new_img = copy.deepcopy(img)
    edges = cv2.Canny(new.astype(np.uint8), 100, 200)
    xs, ys = np.where(edges == 255)
    for _ in range(40):
        mask = np.zeros_like(img, dtype=bool)  # Temporary mask to mark new updates
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_xs = np.clip(xs + dx, 0, img.shape[0] - 1)
            shifted_ys = np.clip(ys + dy, 0, img.shape[1] - 1)
            valid_updates = (img[shifted_xs, shifted_ys] == 0) & (~mask[shifted_xs, shifted_ys])
            mask[shifted_xs[valid_updates], shifted_ys[valid_updates]] = True
        new_img[mask] = 4
        xs, ys = np.where(mask)
    return new_img



def evaluate_t(y, pred):

    pred_t=torch.argmax(pred,dim=1)
    pred_t=pred_t.flatten()
    y_t=y.flatten()
    acc=torch.sum(pred_t[y_t>0]==y_t[y_t>0]).type(torch.float)
    acc=acc/torch.sum(y_t>0)
    return acc

def evaluate(y, pred, reduced=False):
    if reduced:
            confusion_matrix = np.zeros((4,4))
    else:
        confusion_matrix = np.zeros((5,5))
    count = 0
    acc = 0
    class_counts = [0]*5
    for y_true, y_pred in zip(y.flatten(), pred.flatten()):
        if y_true > 0:
            confusion_matrix[y_true, y_pred] += 1
            count += 1
            class_counts[y_true] += 1
            acc += int(y_true == y_pred)
    return confusion_matrix, acc, count, class_counts

class apply_trans(object):
    def __init__(self, window_size=(1,200,200), trans_type='aff', device='cuda', mode='bilinear', padding_mode='border'):
        self.window_size = window_size # (c,h,w)
        self.trans_type = trans_type
        self.device = device
        self.mode = mode
        self.padding_mode = padding_mode
        if self.trans_type == 'shift':
            self.u_dim = 2
            self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1).to(device)
        elif self.trans_type == 'aff':
            self.u_dim = 6
            self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1).to(device)
        elif not trans_type:
            raise NameError('trans_type cannot be recognized')

    def __call__(self, x, u):
        if self.trans_type is not None:
            #id = self.idty.expand((x.shape[0],) + self.idty.size()).to(self.dv)
            # Apply linear only to dedicated transformation part of sampled vector.
            x = x.to(self.device)
            u = u.to(self.device)
            c, h, w = self.window_size
            if self.trans_type=='shift':
                theta=torch.zeros(u.shape[0],2,3).to(self.dv) + self.idty.unsqueeze(0)
                theta[:,:,2]=u
                grid = F.affine_grid(theta, x.view(-1, c, h, w).size())
            elif self.trans_type == 'aff':
                theta = u.view(-1, 2, 3)
                grid = F.affine_grid(theta, x.view(-1, c, h, w).size())
            x = F.grid_sample(x.view(-1, c, h, w), grid, padding_mode=self.padding_mode, mode = self.mode)
            
        return x
    
