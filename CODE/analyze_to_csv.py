import os
import numpy as np
import pandas as pd
import cv2
from skimage import io, morphology, measure
from scipy.ndimage import binary_fill_holes
from predict import predict, predict_file
import argparse
import re
from utils import get_file_by_num, get_file_numbers


def process_files(device,j,model_name_a,model_name_o,target='test/',gt=False,datapath='data/'):
        ff=get_file_by_num(datapath+target,j)
        #print(j,ff)
        celldata=[]
        if 'DF' in ff[0]:
            celltype='DF'
        else:
            celltype='UF'
        for f in ff:
            if 'actin' in f:
                ima = io.imread(os.path.join(datapath,target,f))
            elif 'junction' in f:
                imj = io.imread(os.path.join(datapath,target, f))
            elif 'outline' in f:
                imo =io.imread(os.path.join(datapath,target, f))
        imj_p=imo_p=None
        if not gt:
            imj_p=predict_file(device,None,model_name_a,None,x_prefix='actin',
                         y_prefix='junction', zero_thresh=0, im=ima,  name1=None)
            imo_p=predict_file(device,None,model_name_o,None,x_prefix='pred_junction',
                         y_prefix='outline', zero_thresh=0, im=imj_p,  name1=None)
        

        return ima, imj, imo, imj_p, imo_p, celltype
          

def analyze_cell(j,o,ima, imj, celltype, reduced=0):

        celldata=[]
        if reduced:
            o[o==2]=1
            o[o==3]=2
            o[o==4]=3
            mask = np.isin(o, np.arange(1,3)).astype(np.uint8)
        else:
            mask = np.isin(o, np.arange(1,4)).astype(np.uint8)
        junctions = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1] 
        junctions = morphology.binary_closing(junctions, morphology.disk(5))
        junctions = morphology.binary_erosion(junctions, morphology.disk(5))
        junctions[:,0:20]=True
        junctions[0:20,:]=True
        junctions[:,-20:]=True
        junctions[-20:,:]=True
        cells = binary_fill_holes(junctions)
        cells = cells & np.invert(junctions)
        labeled_cells = measure.label(cells)
        props = measure.regionprops(labeled_cells)
        k=0
        for prop in props:
            data = []
            area = prop.area
            #print('area',area)
            if area <= 5000 or area>=300000:
                continue
            
            k+=1
            data.append(area) # area
            data.append(prop.major_axis_length / prop.minor_axis_length) # major_minor_ratio

            temp = np.zeros_like(o)
            temp[prop.coords[:, 0], prop.coords[:, 1]] = 1
            tempd = morphology.binary_dilation(temp, morphology.disk(5)).astype(np.uint8)

            propsA = measure.regionprops(tempd, intensity_image=ima)
            propsV = measure.regionprops(tempd - temp, intensity_image=imj)

            data.append(propsA[0].mean_intensity if propsA else 0) # mean_intensity_a
            data.append(propsV[0].mean_intensity if propsV else 0) # mean_intensity_v
            data.append(0) # mean_intensity_f
            
            propsJ = measure.regionprops(tempd, intensity_image=o)
            pixel_values = propsJ[0].intensity_image if propsJ else np.array([])

            n1 = np.sum(pixel_values == 1)
            n2 = np.sum(pixel_values == 2)
            n3 = np.sum(pixel_values == 3)
            if reduced:
                tot = n1 + n2
            else:
                tot = n1 + n2 + n3
            #print(n1,n2,n3)
            data.append(n1 / tot if tot else 0) # fraction_1
            data.append(n2 / tot if tot else 0) # fraction_2
            data.append(n3 / tot if tot else 0) # fraction_3
            data.append(prop.centroid[0])
            data.append(prop.centroid[1])
            data.append(celltype)
            data.append(j)
            data.append(k)
            celldata.append(data)
       
        return(celldata)





def match_points_from_ims(j,o,ot,ima, imj, celltype, reduced=0):

    
  
    cdp=analyze_cell(j,o,ima, imj, celltype, reduced)
    cdt=analyze_cell(j,ot,ima, imj,celltype, reduced)
    if len(cdt)==0:
        return None, None, None, None, None, None, None
    cdt=np.array(cdt)
    centt=np.float32(cdt[:,8:10])
    centp=None
    if len(cdp)>0:
        cdp=np.array(cdp)
        centp=np.float32(cdp[:,8:10])
    
    
    ctp=[]
    used_is=[]
    centt_m=[]
    JJ=-np.ones(len(centt),dtype=np.int32)
    if centp is not None:
        for j,ct in enumerate(centt):
              match=True
              sqs=np.sqrt(np.sum((centp-ct)*(centp-ct),axis=1))
              #print(ct,sqs)
              ii=np.argmin(sqs)
              while ii in used_is and ii is not None:
                  st=list(sqs)
                  del st[ii]
                  sqs=np.array(st)
                  #print(sqs)
                  if len(sqs>0):
                      ii=np.argmin(sqs)
                  else:
                      print("can't match",ct)
                      match=False
                      ii=None
              if match and sqs[ii]<150:
                  used_is+=[ii]
                  ctp+=[centp[ii]]
                  centt_m+=[ct]
                  JJ[j]=np.int32(ii)
        centp=np.int32(centp)
        ctp=np.array(ctp)
        centt_m=np.array(centt_m)
    return np.array(np.int32(centt)), np.int32(centt_m), np.array(np.int32(ctp)), centp, JJ, cdp, cdt



def match_points(cdt,cdp):

    
    if len(cdt)==0:
        return None, None, None, None
    #print(cdt.shape, cdp.shape)
    centp=np.float32(cdp[:,8:10])
    centt=np.float32(cdt[:,8:10])
    ctp=[]
    used_is=[]
    centt_m=[]
    JJ=-np.ones(len(centt),dtype=np.int32)
    for j,ct in enumerate(centt):
          match=True
          sqs=np.sqrt(np.sum((centp-ct)*(centp-ct),axis=1))
          #print(ct,sqs)
          ii=np.argmin(sqs)
          while ii in used_is and ii is not None:
              st=list(sqs)
              del st[ii]
              sqs=np.array(st)
              #print(sqs)
              if len(sqs>0):
                  ii=np.argmin(sqs)
              else:
                  print("can't match",ct)
                  match=False
                  ii=None
          if match and sqs[ii]<150:
              used_is+=[ii]
              ctp+=[centp[ii]]
              centt_m+=[ct]
              JJ[j]=np.int32(ii)
    centp=np.int32(centp)
    ctp=np.array(ctp)
    centt_m=np.array(centt_m)
    return np.array(np.int32(centt)), np.int32(centt_m), np.array(np.int32(ctp)), centp, JJ






def analyze_p(device,model_name_a, model_name_o,target='test/',reduced=0,gt=False,dfp=None,datapath='data/'):
    celldata = []
 
    ii=get_file_numbers(datapath+target)
    blank=np.zeros(13,dtype=np.int32)
    ii=np.sort(ii)
    ii=np.unique(ii)
    
    for i,j in enumerate(ii):
        ima, imj, imo, imj_p, imo_p, celltype=process_files(device,j,model_name_a,model_name_o,gt=gt)
       
        
        if gt:
            o=imo
        else:
            o=imo_p
        data=analyze_cell(j,o,ima, imj, celltype, reduced=reduced) 
        
        celld=np.atleast_2d(np.array(data))
        if celld.shape[1]==0:
            continue
        if dfp is not None:
            dfpl=dfp[pd.to_numeric(dfp.image_idx)==j]
            ct,ctm,ctp,centp,JJ=match_points(celld,np.array(dfpl))  
            
            for k,l in enumerate(JJ):
                if l>=0:       
                    data[k].extend(list(dfpl.iloc[l]))
                    #print('k',k,len(data[k]))
                else:
                    bl=blank.copy()
                    bl[11]=j
                    data[k].extend(list(bl))
            
            celld=np.array(data)
        celldata.append(celld)  
    
    celldata=np.concatenate(celldata,axis=0)
  
    df = pd.DataFrame(celldata)
    if dfp is None:
        df.columns = ['area', 'major_minor_ratio', 'mean_intensity_a', 'mean_intensity_v', 'mean_intensity_f', 'fraction_1', 'fraction_2', 'fraction_3','centroid_x','centroid_y','celltype','image_idx','cell_idx']
    else:
      df.columns=['area', 'major_minor_ratio', 'mean_intensity_a', 'mean_intensity_v', 'mean_intensity_f', 'fraction_1', 'fraction_2', 'fraction_3','centroid_x','centroid_y','celltype','image_idx','cell_idx','area_p', 'major_minor_ratio_p', 'mean_intensity_a_p', 'mean_intensity_v_p','mean_intensity_f_p', 'fraction_1_p', 'fraction_2_p', 'fraction_3_p','centroid_x_p','centroid_y_p','celltype_p','image_idx_p','cell_idx_p']
    
    return df
