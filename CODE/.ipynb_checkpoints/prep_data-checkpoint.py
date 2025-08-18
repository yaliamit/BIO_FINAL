import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.utilstest import apply_trans, remove_data, stretch_image #, augment_background, extract_index,augment_background_thick
import argparse
predir='/home/amit/ga/markyl/BIO/'
parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('-t', '--threshold', type=int, help='threshold', default=0)

def construct_actin_junction(n_test=1, n_valid=1, seed=0):
    
    datadirs = ['ML_Actin_Junctions/20220906_Singlez_actin_junctions/20220906_Singlez_actin_junctions_DF/',
            'ML_Actin_Junctions/20220906_Singlez_actin_junctions/20220906_Singlez_actin_junctions_UF/',
            'ML_Actin_Junctions/20220307_SingleZ_actin_junctions/20220307_SingleZ_actin_junctions_DF/',
            'ML_Actin_Junctions/20220307_SingleZ_actin_junctions/20220307_SingleZ_actin_junctions_UF/']

    actin_prefix = ['w1actin', 'w1actin', 'w1561', 'w1561'] # @param {'type':'string'}
    junction_prefix = ['w1junction', 'w1junction', 'w1491', 'w1491'] # @param {'type':'string'}

    np.random.seed(seed)
    remove_data()
    n_dir = len(datadirs)
    count_train, count_valid, count_test = 0, 0, 0
    for i in range(n_dir):
        directory = predir+datadirs[i]
        filenames = os.listdir(directory)
        actin_names = sorted([filename for filename in filenames if filename.startswith(actin_prefix[i])], key=extract_index)
        junction_names = sorted([filename for filename in filenames if filename.startswith(junction_prefix[i])], key=extract_index)
        n_img = len(actin_names)
        selected = np.random.choice(n_img, n_img, replace=False)
        for j in range(n_img):
            actin_name = actin_names[selected[j]]
            junction_name = junction_prefix[i] + actin_name[len(actin_prefix[i]):]
            if j < n_valid:
                filepath = os.path.join(directory, actin_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/valid/actin'+str(count_valid)+'.tif')
                filepath = os.path.join(directory, junction_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/valid/junction'+str(count_valid)+'.tif')
                count_valid += 1
            elif j < n_test + n_valid:
                filepath = os.path.join(directory, actin_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/test/actin'+str(count_test)+'.tif')
                filepath = os.path.join(directory, junction_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/test/junction'+str(count_test)+'.tif')
                count_test += 1
            else:
                filepath = os.path.join(directory, actin_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/train/actin'+str(count_train)+'.tif')
                filepath = os.path.join(directory, junction_name)
                img = stretch_image(plt.imread(filepath)[:1509,:1053])
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save('./data/train/junction'+str(count_train)+'.tif')
                count_train += 1

def construct_actin_outline(n_test=1, n_valid=1, seed=0, augment=0):
    np.random.seed(seed)
    remove_data()
    datadir = predir+'./ML_Actin_Junctions/Junctions_Outlines_Annotated'
    dates = os.listdir(datadir) # 0227, 0307, 0906
    count_train, count_valid, count_test = 0, 0, 0
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    for date in dates:
        subdirs = os.listdir(os.path.join(datadir, date)) # 0227, 0307, 0906
        if '.DS_Store' in subdirs:
            subdirs.remove('.DS_Store')
        for subdir in subdirs:
            cells = os.listdir(os.path.join(datadir, date, subdir)) # different images
            if '.DS_Store' in cells:
                cells.remove('.DS_Store')
            n_cell = len(cells)
            selected = np.random.choice(n_cell, n_cell, replace=False)
            for i in range(n_cell):
                files = os.listdir(os.path.join(datadir, date, subdir, cells[selected[i]])) # actin/junction/anotated
                if i < n_valid:
                    split = 'valid'
                    count_valid += 1
                    count = count_valid
                elif i < n_valid + n_test:
                    split = 'test'
                    count_test += 1
                    count = count_test
                    print(count, subdir)
                else:
                    split = 'train'
                    count_train += 1
                    count = count_train
                for file in files:
                    path = os.path.join(datadir, date, subdir, cells[selected[i]], file)
                    if file.startswith('outlineCenter'):
                        img = plt.imread(path).astype(np.uint8)
                        if augment == 1:
                            img = augment_background(img)
                        img = Image.fromarray(img[:1509,:1053])
                        if np.max(img) == 0:
                            print(path)
                        # img.save(os.path.join('./data/', split, 'outline'+str(count)+'.tif'))
                        if len(np.unique(img)) > 5:
                            print(np.unique(img), file)
                    elif '1642' in file or '2491' in file or '2642' in file:
                        try:
                            img = stretch_image(plt.imread(path))
                            img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                            img = Image.fromarray(img[:1509,:1053])
                            # img.save(os.path.join('./data/', split, 'junction'+str(count)+'.tif'))
                        except:
                            print(path)
                            continue
                        
                    elif '3561' in file or '3491' in file or '1561' in file:
                        try: 
                            img = stretch_image(plt.imread(path))
                            img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                            img = Image.fromarray(img[:1509,:1053])
                            # img.save(os.path.join('./data/', split, 'actin'+str(count)+'.tif'))
                        except:
                            print(path)
                            continue                           

def construct_dfuf(seed=0, count=27):
    np.random.seed(seed)
    datadir = predir+'./ML_Actin_Junctions/Junctions_Outlines_Annotated-selected'
    dates = os.listdir(datadir) # 0227, 0307, 0906
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    for date in dates:
        cell_type = date[-2:]
        cells = os.listdir(os.path.join(datadir, date)) # different images
        if '.DS_Store' in cells:
            cells.remove('.DS_Store')
        n_cell = len(cells)
        selected = np.random.choice(n_cell, n_cell, replace=False)
        for i in range(n_cell):
            files = os.listdir(os.path.join(datadir, date, cells[selected[i]])) # actin/junction/anotated
            count += 1
            for file in files:
                path = os.path.join(datadir, date, cells[selected[i]], file)
                # print(file)
                if file.startswith('outlineCenter'):
                    img = plt.imread(path).astype(np.uint8)
                    img = Image.fromarray(img[:1509,:1053])
                    if np.max(img) == 0:
                        print(0, path)
                    img.save(os.path.join('./data/test', 'outline'+str(count)+'.tif'))
                    if len(np.unique(img)) > 5:
                        print('more than 4 labels', np.unique(img), file)
                elif 'VECad' in file:
                    img = stretch_image(plt.imread(path))
                    img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                    img = Image.fromarray(img[:1509,:1053])
                    img.save(os.path.join('./data/test', 'junction'+str(count)+'.tif'))
                    
                elif 'Actin' in file:
                    img = stretch_image(plt.imread(path))
                    img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                    img = Image.fromarray(img[:1509,:1053])
                    img.save(os.path.join('./data/test', 'actin'+str(count)+'.tif'))                                        



def construct_permeability(n_test=1, n_valid=1, seed=0, threshold=0, stretch=0):
    np.random.seed(seed)
    remove_data(path='data/permeability')
    datadir = predir+'ML_Actin_Junctions/Permeability'
    dates = os.listdir(datadir) # 0227, 0307, 0906
    count_train, count_valid, count_test = 0, 0, 0
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    print(datadir)
    print(os.listdir(datadir))
    leakiness_max  = 1570
    actin_max = 55140
    junction_max = 4685
    

    for date in dates:
        files = os.listdir(os.path.join(datadir, date)) # 0227, 0307, 0906
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        cell_idx = np.unique(list(map(lambda x: int(x[3] + (x[4] if '0'<= x[4] <= '9' else '')), files)))
        n_cells = len(cell_idx)
        selected = np.random.choice(cell_idx, n_cells, replace=False)
        for i in range(n_cells):
            if i < n_valid:
                split = 'valid'
                count_valid += 1
                count = count_valid
            elif i < n_valid + n_test:
                split = 'test'
                count_test += 1
                count = count_test
            else:
                split = 'train'
                count_train += 1
                count = count_train
            file = files[0][:3] + str(selected[i]) + '_w1491zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
            print(img.max())
            img[img>leakiness_max] = leakiness_max
            img = ((img)/(leakiness_max)*255)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join('./data/permeability', split, 'leakiness'+str(count)+'.tif'))

            # Actin
            file = files[0][:3] + str(selected[i]) + '_w2642zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
            # print(img.max())
            if stretch:
                img = stretch_image(plt.imread(path))
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
            else:
                img[img>actin_max] = actin_max
                img = ((img)/(actin_max)*255)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join('./data/permeability', split, 'actin'+str(count)+'.tif'))

            # Junction
            file = files[0][:3] + str(selected[i]) + '_w3561zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
            if stretch:
                img = stretch_image(plt.imread(path))
                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
            else:
                img[img>junction_max] = actin_max
                img = ((img)/(junction_max)*255)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join('./data/permeability', split, 'junction'+str(count)+'.tif'))  

def get_stats(n_test=1, n_valid=1, seed=0, threshold=0):
    np.random.seed(seed)
    remove_data(path='data/permeability')
    datadir = predir+'ML_Actin_Junctions/Permeability'
    dates = os.listdir(datadir) # 0227, 0307, 0906
    count_train, count_valid, count_test = 0, 0, 0
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    print(datadir)
    print(os.listdir(datadir))
    print(dates)
    IMG_P=[]
    IMG_A=[]
    IMG_J=[]
    for date in dates:
        print(date)
        pp=os.path.join(datadir, date)
        print(pp)
        files = os.listdir(pp) # 0227, 0307, 0906
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        print('files',len(files))
        
        cell_idx = np.unique(list(map(lambda x: int(x[3] + (x[4] if '0'<= x[4] <= '9' else '')), files)))
        n_cells = len(cell_idx)
        print('n_cells',n_cells)
        selected = np.random.choice(cell_idx, n_cells, replace=False)
        for i in range(n_cells):
            if i < n_valid:
                split = 'valid'
                count_valid += 1
                count = count_valid
            elif i < n_valid + n_test:
                split = 'test'
                count_test += 1
                count = count_test
            else:
                split = 'train'
                count_train += 1
                count = count_train
            file = files[0][:3] + str(selected[i]) + '_w1491zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
            
            IMG_P+=[img.flatten()]

            file = files[0][:3] + str(selected[i]) + '_w2642zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
            
            IMG_A+=[img.flatten()]
            
            file = files[0][:3] + str(selected[i]) + '_w3561zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = plt.imread(path)
           
            IMG_J+=[img.flatten()]

    return IMG_P, IMG_A, IMG_J

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-s', '--stretch', type=int, help='normalize actin and junction images', default=0)
    args = parser.parse_args()
    print('stretch',args.stretch)
    construct_permeability(n_test=5, n_valid=5, stretch=args.stretch)
