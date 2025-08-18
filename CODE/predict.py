from torch.utils.data import DataLoader
import torch
import os
import shutil
import argparse
from PIL import Image
from utils import load_model
import numpy as np
import matplotlib.pyplot as plt

def predict_file(device,target,model_name,file_num,x_prefix,y_prefix, zero_thresh=0, im=None, pad_size=0,  name1=None):
    model = load_model('./saved_models/' + model_name + '.pkl').to(device)
    
    model.eval()
    #print(model_name)
    if im is None:
        dirpath = './data/permeability/' + target

        if 'junc' == x_prefix and 'leakiness' in y_prefix:
            dirpath = './data/leak_outline/'
        elif name1 is not None and 'junction' in x_prefix:
            if 'pred' in x_prefix:
                dirpath = './data/pred/'+name1+'/'+target
        elif 'junction' in x_prefix and 'outline' in y_prefix:
                dirpath = './data/permeability/'+target
        elif 'junc' == x_prefix and 'outline' in y_prefix:
                dirpath = './data/leak_outline/'

        filename=x_prefix+str(file_num)+'.tif'
        filepath = os.path.join(dirpath, filename)
 
        x = plt.imread(filepath)
        if 'im' in x_prefix and len(x.shape)>2:
            x=x[:,:,0]
    else:
        x=im
    x = torch.from_numpy(x[None,None,:,:]).to(device, dtype=torch.float)
    x=torch.nn.functional.pad(x,(pad_size,pad_size,pad_size,pad_size),"constant", 0)
    img = model(x)
    img=img.detach()

    if pad_size>0:
        img = img[:,:,pad_size:-pad_size,pad_size:-pad_size]
    if y_prefix == 'outline':
        img = img.cpu().numpy().argmax(axis=1).astype(np.uint8).squeeze()
    elif y_prefix=='junction':
        img =(img.cpu().numpy().squeeze()*255).astype(np.uint8)
    else:
        img=img.squeeze()
        img[1][img[0]<=zero_thresh]=0
        img=img[1].cpu().numpy()

    return(img)

def predict(device, name, name1, x_prefix='actin', y_prefix='junction', pad_size=0, zero_thresh=0, replace_pj = 0, rewrite=True):
#affine_coef=1,  trans_type='mix', kernel_size=5, rewrite=True, 
            #reduced=True, n_layers = 4, n_window = 200, window_size = 200, margin = 0):
    temp =  ''
    model = load_model('./saved_models/' + name + '.pkl').to(device)
    model.eval()
    for target in ['train', 'valid', 'test']:
        dirpath = './data/' + target
        if name1 is not None and 'pred' in x_prefix:
            dirpath = './data/pred/'+name1+'/'+target
        if 'leakiness' in y_prefix:
            dirpath = './data/permeability/' + target
        
        filenames = os.listdir(dirpath)
        
        with torch.no_grad():
            for filen in filenames: 
                ss = filen.split('_')
                pref=''
                if len(ss)>1:
                    pref=ss[0]+'_'
                    if len(ss)>2:
                        filename='_'.join(ss[1:])
                    else:
                        filename=ss[1]
                else:
                    filename=filen
                #print(filename)
                if filename.startswith(x_prefix):
                    if target == 'test' and name1 is None:
                        file_num = int(filename.split('.')[0].split('n')[1])
                    filepath = os.path.join(dirpath, pref+filename)
                    x = plt.imread(filepath)/255
                    x = torch.from_numpy(x[None,None,:,:]).to(device, dtype=torch.float)
                    x=torch.nn.functional.pad(x,(pad_size,pad_size,pad_size,pad_size),"constant", 0)
                    img = model(x)
                   
                    if pad_size>0:
                        img = img[:,:,pad_size:-pad_size,pad_size:-pad_size]
                    #print(img.shape)
                    if y_prefix == 'outline':
                        img = img.cpu().numpy().argmax(axis=1).astype(np.uint8).squeeze()
                    elif y_prefix=='junction':
                        img =(img.cpu().numpy().squeeze()*255).astype(np.uint8)
                    
                       
                    img = Image.fromarray(img)
                    if replace_pj: 
                        if x_prefix.startswith("actin") and y_prefix.startswith('junction'): 
                            img.save('./data/'+target+ pref+'/pred_'+ y_prefix + filename[len(x_prefix):]) 
                    elif rewrite: 
                            newpath='./data/pred/'+name+'/'+target+'/'
                            if not os.path.exists(newpath):
                                    os.makedirs(newpath)
                            img.save(newpath+pref +'pred_'+  y_prefix + temp + filename[len(x_prefix):])
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-d', '--device', type=str, help='index of the device', default=3)
    parser.add_argument('-x', '--x_prefix', type=str, help='input category', default='actin')
    parser.add_argument('-y', '--y_prefix', type=str, help='output category', default='junction')
    parser.add_argument('-n', '--rewrite', type=int, help='rewrite folder', default=0)
    parser.add_argument('-t', '--zero_thresh', type=float, help='threshold for background', default=0.)
    parser.add_argument('-p','--pred_junction_replace', type = int, help = 'replace tr/te/val pics for pred_junctions', default = 0)
    parser.add_argument('-m', '--pad_size', type = int, help ='padding', default = 0)
    parser.add_argument('-na', '--name', type=str, help='name of model to predict with',default=None)
    parser.add_argument('-naa', '--name1', type=str, help='name of model to predict with',default=None)

    args = parser.parse_args()
    device = 'cuda:' + str(args.device)
    predict(device, args.name, args.name1,  x_prefix=args.x_prefix, y_prefix=args.y_prefix, pad_size=args.pad_size, zero_thresh=args.zero_thresh, replace_pj = args.pred_junction_replace,rewrite=args.rewrite)
     