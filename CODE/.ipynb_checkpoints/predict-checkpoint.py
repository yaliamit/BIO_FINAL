from torch.utils.data import DataLoader
import torch
import os
import shutil
import argparse
from PIL import Image
from utils import load_model
import numpy as np
import matplotlib.pyplot as plt

def predict_file(device,target,model_name, file_num,x_prefix,y_prefix, zero_thresh=0, model_junction_name=None, im=None, pad_size=0,  name1=None, data_path='data/'):
    model = load_model('./saved_models/' + model_name + '.pkl').to(device)
    
    model.eval()
   
    # Read in the image to be processed.
    if im is None:
        dirpath = data_path + target
        # Option to read in predicted junctions already stored on disk.
        if name1 is not None and 'junction' in x_prefix:
            if 'pred' in x_prefix:
                dirpath = data_path+'pred/'+name1+'/'+target
        # model_junction is always applied to 'actin'
        if model_junction_name is None:
            filename=x_prefix+str(file_num)+'.tif'
        else:
            filename='actin'+str(file_num)+'.tif'
        filepath = os.path.join(dirpath, filename)
        x = plt.imread(filepath)    
    else:
        x=im
    xc=x.copy()
    x = torch.from_numpy(xc[None,None,:,:]).to(device, dtype=torch.float)
    x=torch.nn.functional.pad(x,(pad_size,pad_size,pad_size,pad_size),"constant", 0)
   
    img_junction=None
    if model_junction_name is not None:
        model_junction = load_model('./saved_models/' + model_junction_name + '.pkl').to(device)
        img_junction=model_junction(x)
        x=img_junction.clone()
        img_junction=img_junction.detach()
        if pad_size>0:
            img_junction = img_junction[:,:,pad_size:-pad_size,pad_size:-pad_size]
            img_junction =(img_junction.cpu().numpy().squeeze()*255).astype(np.uint8)
        
        
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
    return(img,img_junction)

def predict(device, model_name, pred_folder_name, x_prefix='actin', y_prefix='junction', pad_size=0, zero_thresh=0,datapath='data/'):
#affine_coef=1,  trans_type='mix', kernel_size=5, rewrite=True, 
            #reduced=True, n_layers = 4, n_window = 200, window_size = 200, margin = 0):
    temp =  ''
    
    model = load_model('./saved_models/' + model_name + '.pkl').to(device)
    model.eval()
    for target in ['train', 'valid', 'test']:
        dirpath = datapath+target
        if pred_folder_name is not None and 'pred' in x_prefix:
            dirpath = datapath+'pred/'+pred_folder_name+'/'+target
        filenames = os.listdir(dirpath)
        
        with torch.no_grad():
            for filen in filenames: 
                ss = filen.split('_')
                pref=''

                if 'DF' in filen or 'UF' in filen:
                    if len(ss)>1:
                        pref=ss[0]+'_'
                        if len(ss)>2:
                            filename='_'.join(ss[1:])
                        else:
                            filename=ss[1]
                else:
                    filename=filen

                if filename.startswith(x_prefix):
                    if target == 'test' and pred_folder_name is None:
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
                    
                    if y_prefix=='leakiness':
                        img=img.squeeze()
                        img[1][img[0]<=zero_thresh]=0
                        img=img[1].cpu().numpy()
                    img = Image.fromarray(img)

                    thr=''
                    newpath=datapath+'pred/'+model_name
                    if y_prefix=='leakiness':
                        thr=str(zero_thresh)
                        newpath=newpath+'_thr_'+thr+'/'+target+'/'
                    else:
                        newpath=newpath+'/'+target+'/'
                    if not os.path.exists(newpath):
                                    os.makedirs(newpath)
                    img.save(newpath+pref +'pred_'+  y_prefix + temp + filename[len(x_prefix):])
                    
    if 'leakiness' in args.y_prefix:
        newpath=datapath+'pred/'+model_name+'_thr_'+thr
        file_name='./Output/log_leak_'+str(device).split(':')[1]+'.txt'
        os.rename(file_name,newpath+pref+'/log.txt')        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-d', '--device', type=str, help='index of the device', default=3)
    parser.add_argument('-x', '--x_prefix', type=str, help='input category', default='actin')
    parser.add_argument('-y', '--y_prefix', type=str, help='output category', default='junction')
    parser.add_argument('-t', '--zero_thresh', type=float, help='threshold for background', default=0.)
    parser.add_argument('-m', '--pad_size', type = int, help ='padding', default = 0)
    parser.add_argument('-na', '--model_name', type=str, help='name of model to predict with',default=None)
    parser.add_argument('-naa', '--pred_folder_name', type=str, help='name of model to predict with',default=None)
    parser.add_argument('-dp', '--data_path', type=str, help='name of data path',default='data/')


    args = parser.parse_args()
    device = 'cuda:' + str(args.device)
    predict(device, args.model_name, args.pred_folder_name,  x_prefix=args.x_prefix, y_prefix=args.y_prefix, pad_size=args.pad_size, zero_thresh=args.zero_thresh, datapath=args.data_path)
     
