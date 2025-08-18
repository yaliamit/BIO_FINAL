import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import re
import shutil
from utils import augment_background, stretch_image



speclist=['20221111_DF','20240304_DF','20240423_UF','20240425_UF']

def is_spec(date):

    for s in speclist:
        if s in date:
            return True
    return False

def construct_actin_outline(n_test=1, n_valid=1, seed=0, augment=0,save=False):
    np.random.seed(seed)
    #remove_data()
    datadir = '/home/amit/ga/BIO/Junctions_Outlines_Annotated'
    dates = os.listdir(datadir) # 0227, 0307, 0906
    count, count_train, count_valid, count_test =  0, 0, 0, 0
    DFF=[]
    UFF=[]
    if os.path.isdir('data/temp/'):
        shutil.rmtree('data/temp/')
    os.makedirs('data',exist_ok=True)
    os.mkdir('data/temp/')
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    for date in dates:
        isspec=is_spec(date)
        #print(date,isspec)
        subdirs = os.listdir(os.path.join(datadir, date)) # 0227, 0307, 0906
        if '.DS_Store' in subdirs:
            subdirs.remove('.DS_Store')
        for subdir in subdirs:
          if 'Permeability' not in subdir:
            
            cells = os.listdir(os.path.join(datadir, date, subdir)) # different images
            if '.DS_Store' in cells:
                cells.remove('.DS_Store')
            n_cell = len(cells)
            #selected = np.random.choice(n_cell, n_cell, replace=False)
            selected=range(n_cell)
            for i in range(n_cell):
                
                pth=os.path.join(datadir, date, subdir, cells[selected[i]])
                files = os.listdir(pth) # actin/junction/anotated
                split=''
                count+=1
                if 'DF' in pth:
                    DFF+=[count]
                    pref='DF_'
                elif 'UF' in pth:
                    UFF+=[count]
                    pref='UF_'
                for file in files:
                    
                    path = os.path.join(datadir, date, subdir, cells[selected[i]], file)
                    if file.startswith('outlineCenter'):
                        #print('out',file)
                        if save:
                            img = plt.imread(path).astype(np.uint8)
                            if augment == 1:
                                img = augment_background(img)
                            
                            img = Image.fromarray(img[:1509,:1053])
                            
                            if np.max(img) == 0:
                                print('0 max',path)
                                os.remove(path)
                            
                            img.save(os.path.join('data/temp/', split, pref+'outline'+str(count)+'.tif'))
                            # if len(np.unique(img)) > 5:
                            #     print(np.unique(img), file)
                    elif (isspec and 'VECad' in file) or ((not isspec) and
                    ('1642' in file or '2491' in file or '2642' in file)):
                        #print('junc',file)
                        try:
                           if save:
                                img = stretch_image(plt.imread(path))
                                img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                                img = Image.fromarray(img[:1509,:1053])
                           
                                img.save(os.path.join('data/temp/', split, pref+'junction'+str(count)+'.tif'))
                        except:
                            print('oops',path)
                            continue
                        
                    elif (isspec and 'Actin' in file) or ((not isspec) and
                    ('3561' in file or '3491' in file or '1561' in file)):
                                   
                        try:
                          if save:
                            print('Actin',file)
                            img = stretch_image(plt.imread(path))
                            img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
                            img = Image.fromarray(img[:1509,:1053])
                            img.save(os.path.join('data/temp/', split, pref+'actin'+str(count)+'.tif'))
                        except:
                            print('oops',path)
                            continue  
    print('count',count)
    return(DFF,UFF)

def allocate_to_train_test_valid():

    shutil.rmtree('data/train', ignore_errors=True)
    shutil.rmtree('data/test', ignore_errors=True)
    shutil.rmtree('data/valid', ignore_errors=True)

    os.makedirs('data/train/',exist_ok=True)
    os.makedirs('data/test/',exist_ok=True)
    os.makedirs('data/valid/',exist_ok=True)
    aa=os.listdir('data/temp/')
    
    
    n_cell=237
    selected = np.random.choice(n_cell, n_cell, replace=False)
    
    aatr=[]
    aava=[]
    aate=[]
    goodi=[]
    for i in range(n_cell):
    
        tt=[]
        for f in aa:
            s=re.findall(r'\d+',f)
            if len(s)>0:
                ii=int(s[0])
            if i+1==ii:
                tt+=[f]
        if len(tt)==3:
            goodi+=[i]
    print('goodi',len(goodi))
    np.random.shuffle(goodi)

    for j,i in enumerate(goodi):
        tt=[]
        for f in aa:
            s=re.findall(r'\d+',f)
            if len(s)>0:
                ii=int(s[0])
            if i+1==ii:
                tt+=[f]
        if len(tt)==3:
            if j<128:
                for t in tt:
                    shutil.move('data/temp/'+t,'./data/train/'+t)
                aatr+=tt
            elif j<188:
                for t in tt:
                    shutil.move('data/temp/'+t,'./data/test/'+t)
                aate+=tt
            else:
                for t in tt:
                    shutil.move('data/temp/'+t,'./data/valid/'+t)
                aava+=tt
        else:
            print('tt',tt)
    
    print(len(aatr)/3,len(aate)/3,len(aava)/3) 



def construct_permeability(n_test=1, n_valid=1, seed=0, augment=0):
    np.random.seed(seed)


    shutil.rmtree('data/permeability', ignore_errors=True)

    os.makedirs('data/permeability',exist_ok=True)
    os.makedirs('data/permeability/train',exist_ok=True)
    os.makedirs('data/permeability/test',exist_ok=True)
    os.makedirs('data/permeability/valid',exist_ok=True)
    datadir = '/home/amit/ga/BIO/Junctions_Outlines_Annotated/Permeability'
    leakiness_max = 1570
    dates = os.listdir(datadir) # 0227, 0307, 0906
    count_train, count_valid, count_test = 0, 0, 0
    if '.DS_Store' in dates:
        dates.remove('.DS_Store')
    for date in dates:
        ctype = 'UF_' if 'UF' in date else 'DF_'
        
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
            #img = stretch_image(plt.imread(path))
            #img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
           
            img = plt.imread(path)
            img[img>leakiness_max] = leakiness_max
            img = ((img)/(leakiness_max)*255)
            #img = Image.fromarray(img)
            img = Image.fromarray(img.astype(np.uint8))
            print('MIN',np.min(img))
            img.save(os.path.join('data/permeability', split,'leakiness'+str(count)+'.tif'))
            file = files[0][:3] + str(selected[i]) + '_w2642zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = stretch_image(plt.imread(path))
            img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join('data/permeability', split,'actin'+str(count)+'.tif'))
            file = files[0][:3] + str(selected[i]) + '_w3561zyla' + ('-1' if date.endswith('DF') else '')  + '.tif'
            path = os.path.join(datadir, date, file)
            img = stretch_image(plt.imread(path))
            img = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join('data/permeability', split,'junction'+str(count)+'.tif'))  
    print(count_train, count_test, count_valid)
