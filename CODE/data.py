import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import apply_trans, augment_background_thick
import re

class WindowData(Dataset):
    def  __init__(self, split='train', x_prefix='actin', y_prefix='junction', data_path='./data', continuous=1, reduced=0,
                seed=0, img_size=(1509,1053), window_size=(200,200), small_window_size=150, n_window=200, n_channels=1,
                trans_type='aff', device='cuda', r00=0.1, r11=0.1, r01=1, r10=1, n_selected_img=20, extra_dir=0, bdy=5, mrg=25, leak_thresh=.1, flip=True):
        np.random.seed(seed)
        # torch.manual_seed(seed)
        self.extra_dir=extra_dir
        self.n_channels=n_channels
        self.flip=flip
        splita=split
        # testtrain - Running a test on the training images.
        if 'test' in split and 'train' in split:
            splita='train'
            split='test'
        # Predict outline from predicted junctions already computed in the pred_directory.
        if 'pred' in x_prefix:
            self.directory = os.path.join('./data', splita)
            self.pred_directory = os.path.join(data_path,splita)
        else:
            self.directory = os.path.join(data_path, splita)
        print(self.directory)
        self.n_selected_img = n_selected_img
        # Size of extracted windows, and size of smaller subwindows used after deformations.
        self.window_size, self.n_window, self.small_window_size = window_size, n_window, small_window_size

        # Size of the original images.
        self.img_size = img_size
        self.r00, self.r11, self.r10, self.r01 = r00, r11, r10, r01

        # Transformation type - just rotation, just affine, rotation + affine
        self.trans_type = trans_type
        
        # Transformation of input image
        self.transform_x = apply_trans((n_channels,)+window_size, 'aff', device)
        # Transformation of output image.
        self.transform_y = apply_trans((1,)+window_size, 'aff', device)
        self.perm=y_prefix=='leakiness'
        self.import_image(x_prefix, y_prefix, continuous, reduced, split,bdy,mrg,leak_thresh)
        #print('self.x_img',self.x_img.shape)
        # In test mode you read in the whole big image and run the convolutional model on it.
        if split == 'test' or split=='valid' or split=='traintest':
            self.x_window = torch.from_numpy(self.x_img)
            self.y_window = torch.from_numpy(self.y_img[:,None,:,:])
        # Extract subwindows 
        else:
            self.select_window()
           

    def import_image(self, x_prefix, y_prefix, continuous, reduced, split,bdy,mrg,leak_thresh):
        if 'pred' in x_prefix:
            filenames = os.listdir(self.pred_directory)
        else:
            filenames = os.listdir(self.directory)
        ii=[]
        fx=[]
        for ff in filenames:
          if 'DF' in ff or 'UF' in ff:
            ss=ff.split('_')
            if len(ss)>2:
                f='_'.join(ss[1:])
            else:
                f=ss[1]
            if f.startswith(x_prefix):
                fx+=[ss[0]+'_'+f]
                ii+=[int(f[len(x_prefix):].split('.')[0])]
          else:
            if ff.startswith(x_prefix):
                fx+=[ff]
                ii+=[int(ff[len(x_prefix):].split('.')[0])]
        fx=np.array(fx)
        jj=np.array(np.int64(np.argsort(ii)))
        fx=fx[jj]
        x_img, y_img = [], []
        self.n_img = 0
        for filen in fx: 
            ss=filen.split('_')
            pref=''
            if len(ss)>1:
                pref=ss[0]+'_'
                if len(ss)>2:
                    filename='_'.join(ss[1:])
                else:
                    filename=ss[1]
            else:
                filename=filen

            if filename.startswith(x_prefix):
                dir=self.directory
                if 'pred' in x_prefix:
                    dir=self.pred_directory
                filepath = os.path.join(dir, pref+filename)
                img = plt.imread(filepath)/255
                ## INPUTS
                dx1,dx2,dy1,dy2=0,0,0,0
                # Adjust big image sizes to be all the same.
                if img.shape[0]<self.img_size[0]:
                            dx1=(self.img_size[0]-img.shape[0])//2
                            dx2=self.img_size[0]-dx1-img.shape[0]
                if img.shape[1]<self.img_size[1]:
                            dy1=(self.img_size[1]-img.shape[1])//2
                            dy2=self.img_size[1]-dy1-img.shape[1]
                img = img.reshape(1,img.shape[0],img.shape[1])
                if y_prefix == 'leakiness':
                    self.img_size=img.shape[1:3]
                img=np.pad(img,((0,0),(dx1,dx2),(dy1,dy2)),mode='constant',constant_values=(0))  
                x_img.append(img)

                ## Corresponding OUTPUTS
                filepath = os.path.join(self.directory, pref+y_prefix+filename[len(x_prefix):])
              
                img = plt.imread(filepath)
    
                if continuous:
                    if y_prefix != 'leakiness':
                        img = img/255.
                    else:
                        img = img/255.
                        if leak_thresh>=0:
                            img[img<leak_thresh]=0
             
                        else:
                            img=2.5+np.clip(np.log(img),-3.,0.)
                else:
                    if split == 'train' and y_prefix=='outline':
                        img=augment_background_thick(img,bdy, mrg)
                #if split=='test':
                img=np.pad(img,((dx1, dx2),(dy1,dy2)),mode='constant',constant_values=(0))
                y_img.append(img)
                self.n_img += 1
        self.x_img = np.array(x_img)
        self.y_img = np.array(y_img)
        # If using only 4 labels, merge labels 1 and 2.
        if (not continuous) & reduced:
                self.y_img[(self.y_img==2) | (self.y_img==1)] = 1
                self.y_img[self.y_img==3] = 2
                self.y_img[self.y_img==4] = 3

    # Randomly select subwindows of size window_size and in some cases remove empty windows.
    def select_window(self):
        x_window, y_window = [], []
        selected = np.random.choice(self.n_img, min(self.n_img, self.n_selected_img), replace=False)
        
        for i in selected:
            x, y = self.x_img[i], self.y_img[i]
            h = np.random.randint(self.img_size[0]-self.window_size[0], size=self.n_window)
            w = np.random.randint(self.img_size[1]-self.window_size[1], size=self.n_window)
            for j in range(self.n_window):
                y_temp = y[h[j]:h[j]+self.window_size[0],w[j]:w[j]+self.window_size[1]].reshape((1,)+self.window_size)
                if y_temp.max() > 0 or self.perm:
                    x_window.append(x[:,h[j]:h[j]+self.window_size[0],
                                    w[j]:w[j]+self.window_size[1]])              
                    y_window.append(y_temp)
        x_window, y_window = np.array(x_window, dtype=np.float32), np.array(y_window, dtype=np.float32)
        self.x_window, self.y_window = torch.from_numpy(x_window), torch.from_numpy(y_window)

            
    def __len__(self):
        return self.x_window.size()[0]

    def __getitem__(self, idx):
        return self.x_window[idx], self.y_window[idx]

    def transform_window(self,x_window,y_window):

        with torch.no_grad():
            hs = self.small_window_size//2
            hw1 = self.window_size[0]//2
            hw2 = self.window_size[1]//2
            
            if self.flip:
                B=(torch.rand(self.x_window.shape[0])<.5)
                xfl=torch.flip(self.x_window,[3])
                yfl=torch.flip(self.y_window,[3])
                self.x_window[B]=xfl[B]
                self.y_window[B]=yfl[B]
            if self.trans_type == 'rotate' or self.trans_type == 'mix':
                u = torch.zeros((x_window.size()[0],6))
                
                angle_radians = torch.from_numpy(np.random.uniform(-np.pi, np.pi,u.shape[0]))
                u[:,0]= torch.cos(angle_radians)
                u[:,4]= torch.cos(angle_radians)
                u[:,1]= -torch.sin(angle_radians)
                u[:,3]= torch.sin(angle_radians)
                xt = self.transform_x(x_window, u)
                yt = self.transform_y(y_window, u)

            if self.trans_type == 'aff' or self.trans_type == 'mix':
                u = torch.rand((x_window.size()[0],6))
                u[:,2]=0 # x translation
                u[:,5]=0 # y translation
                u[:,0]=(u[:,0]-.5)*self.r00 + 1 # rotation r00 
                u[:,4]=(u[:,4]-.5)*self.r11 + 1 # rotation r11
                u[:,1]=(u[:,1]-.5)*self.r01 # rotation r01
                u[:,3]=(u[:,3]-.5)*self.r10 # rotation r10
            
                if self.trans_type == 'aff':
                    xt = self.transform_x(x_window, u)
                    yt = self.transform_y(y_window, u)
                else:
                    xt = self.transform_x(xt, u)
                    yt = self.transform_y(yt, u)
            xt = xt[:,:, hw1-hs:hw1+hs, hw2-hs:hw2+hs]#.cpu()
            yt = yt[:,:, hw1-hs:hw1+hs, hw2-hs:hw2+hs]#.cpu()

            return xt, yt