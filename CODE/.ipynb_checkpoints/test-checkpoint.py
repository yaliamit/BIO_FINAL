from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn as nn
from skimage import io
#from data import WindowData_actin_junction, WindowData_junction_outline, WindowData_actin_junction_outline, WindowData_pred_junction_outline
from predict import predict_file
from utils.utilstest import load_model, dice_loss, plot_img, plot_outline, confustion, evaluate, get_file_numbers, get_file_by_num
import os
import numpy as np
import re



def process_file(device,ima,model_name_a,model_name_o,target='test/'):
        
           
             imj_p=predict_file(device,None,model_name_a,None,x_prefix='actin',
                         y_prefix='junction', zero_thresh=0, im=ima,  name1=None)
             imo_p=predict_file(device,None,model_name_o,None,x_prefix='pred_junction',
                         y_prefix='outline', zero_thresh=0, im=imj_p,  name1=None)
        

             return imo_p


def test_new(device, model_name_a, model_name_o,target='test/',reduced=0):
    
    fid = open('./Output/log'+str(device).split(':')[1]+'.txt', 'a')
    dir='data/'+target
    ii=get_file_numbers(dir)
    ii=np.sort(ii)
    ii=np.unique(ii)
    if reduced:
        confusion_matrix=np.zeros((4,4))
    else:
        confusion_matrix = np.zeros((5,5))
        class_counts = [0]*5
    acc, count = 0, 0
    for i,j in enumerate(ii):
        print(i)
        ff=get_file_by_num('./data/'+target,j)
        for f in ff:
            if 'actin' in f:
                ima = io.imread(os.path.join('./data/',target,f))
            elif 'outline' in f:
                imo =io.imread(os.path.join('./data/',target, f))
        y=imo.astype(int)
        
        pred=process_file(device,ima,model_name_a,model_name_o,target)

        temp1, temp2, temp3, temp4 = evaluate(y, pred,reduced)
        confusion_matrix += temp1
        acc += temp2
        count += temp3
        class_counts = np.add(class_counts, temp4)
    confusion_matrix=confusion_matrix[1:,1:]
    fid.write(str(confusion_matrix[0:(4-reduced),0:(4-reduced)]/confusion_matrix[0:(4-reduced),0:(4-reduced)].sum(axis=1, keepdims=True))+'\n')
    fid.write(str(acc/count)+'\n')
    if not reduced:
            cc=np.zeros((3,3))
            cc[0,0]=confusion_matrix[0:2,0:2].sum()
            cc[0,1]=confusion_matrix[0,2]+confusion_matrix[1,2]
            cc[0,2]=confusion_matrix[0,3]+confusion_matrix[1,3]
            cc[1,0]=confusion_matrix[2,0]+confusion_matrix[2,1]
            cc[2,0]=confusion_matrix[3,0]+confusion_matrix[3,1]
            cc[1:3,1:3]=confusion_matrix[2:4,2:4]
            fid.write('Reduced:\n')
            fid.write(str(cc/cc.sum(axis=1,keepdims=True))+'\n')
            fid.write('Reduced acc:'+str(np.sum(np.diag(cc))/np.sum(cc))+'\n')
    aa=str(class_counts[1:5-reduced]/np.sum(class_counts[1:5-reduced]))
    fid.write('class probs' + aa + '\n')

def tester(device, testset, model, continuous=1, reduced=0):
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    model = model.to(device)
    test_loss = 0
    model.eval()
    fid = open('./Output/log'+str(device).split(':')[1]+'.txt', 'a')
    totloss=0
    with torch.no_grad():
        acc, count = 0, 0
        if not continuous:
            if reduced:
                confusion_matrix=np.zeros((4,4))
            else:
                confusion_matrix = np.zeros((5,5))
            class_counts = [0]*5
        numc=10
        for x, y in test_loader:
            x = x.to(device, dtype=torch.float)
            y = y.cpu().numpy()
            if not continuous:
                y=y.astype(int)
            pred = model(x)
            pred=pred.cpu().numpy()
            if not continuous:
                pred=pred.argmax(axis=1).astype(int)
                temp1, temp2, temp3, temp4 = evaluate(y, pred,reduced)
                confusion_matrix += temp1
                acc += temp2
                count += temp3
                class_counts = np.add(class_counts, temp4)

            else:
                loss=np.mean((pred-y)*(pred-y))
                print(loss)
                totloss+=loss
                count+=1
    if not continuous:
        confusion_matrix=confusion_matrix[1:,1:]
        print(str(confusion_matrix[0:(4-reduced),0:(4-reduced)]/confusion_matrix[0:(4-reduced),0:(4-reduced)].sum(axis=1, keepdims=True))+'\n')
        fid.write(str(confusion_matrix[0:(4-reduced),0:(4-reduced)]/confusion_matrix[0:(4-reduced),0:(4-reduced)].sum(axis=1, keepdims=True))+'\n')
    
        fid.write(str(acc/count)+'\n')
        print(confusion_matrix)
        if not reduced:
            cc=np.zeros((3,3))
            cc[0,0]=confusion_matrix[0:2,0:2].sum()
            cc[0,1]=confusion_matrix[0,2]+confusion_matrix[1,2]
            cc[0,2]=confusion_matrix[0,3]+confusion_matrix[1,3]
            cc[1,0]=confusion_matrix[2,0]+confusion_matrix[2,1]
            cc[2,0]=confusion_matrix[3,0]+confusion_matrix[3,1]
            cc[1:3,1:3]=confusion_matrix[2:4,2:4]
            print(cc)
            fid.write('Reduced:\n')
            fid.write(str(cc/cc.sum(axis=1,keepdims=True))+'\n')
            fid.write('Reduced acc:'+str(np.sum(np.diag(cc))/np.sum(cc))+'\n')
        aa=str(class_counts[1:5-reduced]/np.sum(class_counts[1:5-reduced]))
        fid.write('class probs' + aa + '\n')
    else:
        print('Total_loss',totloss/count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-k', '--kernel_size', type=int, help='Kernel size for the operation', default=5)
    parser.add_argument('-p', '--padding', type=int, help='padding size for the operation', default=2)
    parser.add_argument('-a', '--affine', type=float, help='affine coefficient for the operation', default=1)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs for the operation', default=100)
    parser.add_argument('-t', '--transform', type=int, help='whether to transform the image', default=1)
    parser.add_argument('-d', '--device', type=str, help='index of the device', default=0)
    parser.add_argument('-m', '--message', type=str, help='log message', default='')
    parser.add_argument('-r', '--reduced', type=int, help='whether combine label 1 & 2', default=0)
    args = parser.parse_args()
    device = 'cuda:' + str(args.device)
    tester(device, shift_image=args.transform, kernel_size=args.kernel_size, padding=args.padding, message=args.message, affine_coef=args.affine)
