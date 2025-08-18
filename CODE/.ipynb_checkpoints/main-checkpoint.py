import argparse
import torch
import os

from train import trainer
from test import tester, test_new
from data import WindowData
from model.unet import Unet
from utils import load_model, save_model, compute_weights
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('-x', '--x_prefix', type=str, help='input prefix', default='actin')
parser.add_argument('-y', '--y_prefix', type=str, help='output prefix', default='junction')
parser.add_argument('--cuda', type=str, help='cuda type', default='cuda:0')
parser.add_argument('--seed', type=int, help='random seed', default=0)
parser.add_argument('--data_path', type=str, help='the path of data', default='./data')
parser.add_argument('--result_path', type=str, help='the path of result', default='./images/fig4')
parser.add_argument('--trans_type', choices=['aff', 'rotate', 'mix'], type=str, help='type of transformation', default='mix')
parser.add_argument('-a', '--affine_coef', type=float, help='affine coefficient for the operation', default=0.1)
parser.add_argument('-ll', '--leak_thresh', type=float, help='leakiness threshold', default=0.1)
parser.add_argument('-k', '--kernel_size', type=int, help='Kernel size for the operation', default=5)
parser.add_argument('-p', '--padding', type=int, help='padding size for the operation', default=2)
parser.add_argument('--model_path', type=str, help='the path of data', default='./saved_models')
parser.add_argument('--plot', type=int, help='plot', default=0)
parser.add_argument('--reduced', type=int, help='reduced labels', default=0)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs for the operation', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
parser.add_argument('--wd', type=float, help='weight decay', default=1e-8)

parser.add_argument('--ga', type=float, help='decrease in learning rate', default=.1)
parser.add_argument('--lr_step', type=int, help='learning rate step change', default=100)
parser.add_argument('--batch_size', type=int, help='batch size', default=32)
parser.add_argument('--zero_weight', type=float, help='add weighted bernoulli loss to mse loss', default=.2)
parser.add_argument('--log', type=int, help='log', default=1)
parser.add_argument('--dice_steps', type=int, help='log', default=0)
parser.add_argument('--weights', type=int, help='weights', default=1)
parser.add_argument('--save', type=int, help='save model', default=0)
parser.add_argument('--SA', type=int, help='Self Attention', default=0)
# Arguments added for future testing 
parser.add_argument('-l', '--n_layers', choices=[2, 3, 4], type = int, help = "number of down/up layers", default = 4)
parser.add_argument('-nw', '--n_window', type = int, help = 'Number of windows per image', default = 200) 
parser.add_argument('-ns', '--n_selected', type = int, help = 'Number of images per epoch', default = 20) 
parser.add_argument('-ws', '--window_size', type = int, help = 'Window Size', default = 200)
parser.add_argument('-s', '--split', type=int, help='what index to change the loss', default=100)
parser.add_argument('-m', '--margin', type = int, help ='margin on padding', default = 0)
parser.add_argument('-mr', '--mrg', type = int, help ='margin on adding background', default = 25)
parser.add_argument('-bd', '--bdy', type = int, help ='boundary on adding background', default = 5)

parser.add_argument('-o', '--out_margin', type = int, help ='margin for classification', default = 0)
parser.add_argument('-co', '--cont', type = int, help ='continue training', default = 0)
parser.add_argument('-fp', '--flip', type = int, help ='flip images', default = 0)
parser.add_argument('-ex', '--extra_dir', type = int, help ='etra input of actin for classification', default = 0)
parser.add_argument('-pe', '--perm', type = int, help ='permeability data', default = 0)
parser.add_argument('-trte', '--traintest', type = int, help ='test on train', default = 0)

parser.add_argument('-na', '--model_name', type=str, help='name of model to predict with',default=None)
parser.add_argument('-no', '--model_name_o', type=str, help='name of model to predict with',default=None)
parser.add_argument('-di', '--direct_test', type=str, help='name of model to predict with',default=0)


main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true')
main_command.add_argument('--test', action='store_false', dest='train')

if __name__ == '__main__':
    args = parser.parse_args()
    device = args.cuda
    #
    if args.y_prefix=='leakiness':
        name = args.x_prefix + '_' + args.y_prefix + '_' + args.trans_type + '_' + str(args.affine_coef) + "_ws_"+str(args.window_size)+"_zero_weight_"+str(args.zero_weight)+"_leak_thresh_"+str(args.leak_thresh) + "_fl_"+str(args.flip)
    else:
        name = args.x_prefix + '_' + args.y_prefix + '_' + args.trans_type + '_' + str(args.affine_coef) + '_kernel_' + str(args.kernel_size)+ "_nlayers_" + str(args.n_layers) +"_ds_"+str(args.dice_steps)+"_lrstep_"+str(args.lr_step)+"_ws_"+str(args.window_size)+"_fl_"+str(args.flip)
    
    if args.y_prefix == 'outline':
        name=name+"_bdy_"+str(args.bdy)+"_mrg_"+str(args.mrg)
    if args.y_prefix == 'leakiness':
        name=name+"_a_"+str(args.affine_coef)
    if 'abs' in args.data_path:
        name=name+'_abs'
    print('name',name)
    margin=args.margin
    n_channels=1
    if args.extra_dir:
            n_channels=2
    
    # Outline yields a categorical output, all others yield a continuous output.
    if args.y_prefix != 'outline':# and args.y_prefix != 'leakiness':
        continuous = 1 # Junction output
        n_output = 1
        if args.y_prefix=='leakiness':
            continuous=2 # Leakiness output is 2 dimensional, probability of none, and leakiness level otherwise.
        ignore_index = -100
    else:
        continuous = 0
        # Either 5 categories, or boundary and thick boundary merged as one category.
        n_output = 4 if args.reduced else 5
        ignore_index = 0
        margin=args.out_margin
        
    img_size=(1509,1053)
    if args.y_prefix=='leakiness':
        img_size=(1584, 1120)
    
    if args.train:
        torch.manual_seed(args.seed)

        # Setup the training set loader and the validation set loader. 
        trainset = WindowData(split='train',img_size=img_size, x_prefix=args.x_prefix, y_prefix=args.y_prefix, data_path=args.data_path,  n_selected_img=args.n_selected,
            continuous=continuous, trans_type=args.trans_type, device=device, r01=args.affine_coef, r10=args.affine_coef, seed=args.seed, 
            reduced=args.reduced, n_window= args.n_window, n_channels=n_channels, window_size= (args.window_size, args.window_size), 
            small_window_size= int(args.window_size*0.75),  bdy=args.bdy, mrg=args.mrg, leak_thresh=args.leak_thresh)
        validset = WindowData(split='valid', img_size=img_size,x_prefix=args.x_prefix, y_prefix=args.y_prefix, data_path=args.data_path, device=device, n_selected_img=args.n_selected,
            continuous=continuous, seed=args.seed, reduced=args.reduced, n_window = args.n_window, n_channels=n_channels,
            window_size= (args.window_size, args.window_size), small_window_size= int(args.window_size*0.75), extra_dir=args.extra_dir, leak_thresh=args.leak_thresh)
        
        # Continue training an existing model.
        if args.cont:
            model = load_model(os.path.join(args.model_path, args.model_name+'.pkl'))
        else:
            LI=0
            if (continuous==2):
                # Determines the structure of the unet. 
                # If using a leak_thresh you need two outputs: the probability of some leakiness
                # and the level of leakiness.
                if (args.leak_thresh>0):
                    LI=2
                    n_output=2
                else:
                    LI=1
                    n_output=1           
            model = Unet(n_channels = n_channels, n_classes = n_output, kernel_size= args.kernel_size, padding = args.padding, n_layers = args.n_layers, LI=LI )
        pc=0

        for pp in model.parameters():
            pc+=np.prod(pp.shape)
        print('Number of parameters:' + str(pc))
        if args.weights and not continuous:
            weights = compute_weights(torch.from_numpy(trainset.y_img), num_classes=n_output)
            weights /= weights.sum()
        else:
            weights = None
        os.makedirs(args.model_path, exist_ok=True)
        save_name=os.path.join(args.model_path, name + '_' + str(args.cuda).split(':')[1])
        if args.log:
            print(args.cuda)
            ext=''
            if 'leakiness' in args.y_prefix:
                ext='_leak_'
            os.makedirs("Output",exist_ok=True)
            f = open('./Output/log'+ext+str(args.cuda).split(':')[1]+'.txt', 'a')
            f.write('ARGS:'+str(args))
            f.write('\n')
            f.close()
        print(args)

        # Perform the training.
        trainer(args.cuda, trainset, validset, model, save_name, args.epochs, weights=weights, continuous=continuous, 
            dice_steps=args.dice_steps, batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd, zero_weight=args.zero_weight, 
            ignore_index=ignore_index, log=args.log, seed=args.seed, loss_number = args.split, margin= margin, lr_step=args.lr_step, gamma=args.ga)
        
        if args.save:
            save_model(save_name + '.pkl', model)
        ext=''
        if 'leakiness' in args.y_prefix:
                ext='_leak_'
        f = open('./Output/log'+ext+str(args.cuda).split(':')[1]+'.txt', 'a')
        f.write(name+'_'+str(args.cuda).split(':')[1] +'\n')
        f.close()
    # Testing an existing outline model relative to user marked outlines.
    else:
      if args.direct_test:
        target='test/'
        if args.traintest:
            target='train/'
        test_new(args.cuda,args.model_name, args.model_name_o,target='test/',reduced=0)
      else:
        split='test'
        if args.traintest:
            split='traintest'
        f = open('./Output/log'+str(args.cuda).split(':')[1]+'.txt', 'a')
        f.write(f'Testing:{split}\n')
        testset = WindowData(split=split, img_size=img_size,x_prefix=args.x_prefix, y_prefix=args.y_prefix, data_path=args.data_path, 
            continuous=continuous, reduced=args.reduced, n_window = args.n_window, n_channels=n_channels, extra_dir=args.extra_dir)
        model = load_model(os.path.join(args.model_path, args.model_name+'.pkl'))
        #print(model)
        pc=0
        for pp in model.parameters():
            pc+=np.prod(pp.shape)
        print('Number of parameters:' + str(pc))
        f = open('./Output/log'+str(args.cuda).split(':')[1]+'.txt', 'a')
        f.write(f'Tested Model: {args.model_name}\n')
        f.close()
        tester(args.cuda, testset, model, continuous=continuous, reduced=args.reduced)



