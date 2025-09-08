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


def process_files(device,j,model_name_a,model_name_o,model_name_leak=None, target='test/',gt=False,datapath='data/'):
        leak_thresh=.2
        ff=get_file_by_num(datapath+target,j)
        print(j,ff)
        celldata=[]
        ima=imj=imo=iml=None
        celltype='Unknown'
        if 'DF' in ff[0]:
            celltype='DF'
        elif 'UF' in ff[0]:
            celltype='UF'
        for f in ff:
            if 'actin' in f:
                ima = io.imread(os.path.join(datapath,target,f))
            elif 'junction' in f:
                imj = io.imread(os.path.join(datapath,target, f))
            elif 'outline' in f:
                imo =io.imread(os.path.join(datapath,target, f))
            elif 'leakiness' in f:
                iml =io.imread(os.path.join(datapath,target, f))
                iml = iml/255.
                if leak_thresh>=0:
                    iml[iml<leak_thresh]=0
        imj_p=imo_p=iml_p=None

        imj_p,_=predict_file(device,None,model_name_a,None,x_prefix='actin',
                         y_prefix='junction', zero_thresh=0, im=ima,  name1=None)
        if 'pred' in model_name_o:
            imo_p,_=predict_file(device,None,model_name_o,None,
                               x_prefix='pred_junction',
                         y_prefix='outline', zero_thresh=0, im=imj_p,  name1=None)
        else:
            imo_p,_=predict_file(device,None,model_name_o,None,
                               x_prefix='junction',
                         y_prefix='outline', zero_thresh=0, im=imj_p,  name1=None)

        if model_name_leak is not None:
                if 'pred_junction' in model_name_leak:
                    iml_p,_=predict_file(device,None,model_name_leak,None,    
                            x_prefix='pred_junction',y_prefix='leakiness',zero_thresh=0.8,
                            im=imj_p,  name1=None, data_path=datapath)
                elif 'junction' in model_name_leak:
   
                    iml_p,_=predict_file(device,'test/',model_name_leak,j,
                                       x_prefix='junction',y_prefix='leakiness',
                                       zero_thresh=0.8,data_path=datapath)
                elif 'actin' in model_name_leak:
                    iml_p,_=predict_file(device,'test/',model_name_a,j,x_prefix='actin',
                         y_prefix='leakiness', zero_thresh=0.8, name1=None, data_path=datapath)

        return ima, imj, imo, iml,  imj_p, imo_p, iml_p, celltype
          

def analyze_cell(j,o,ima, imj, iml, celltype, reduced=0):

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
            # Fill region with 1's
            temp[prop.coords[:, 0], prop.coords[:, 1]] = 1
            # Dilate region
            tempd = morphology.binary_dilation(temp, morphology.disk(5)).astype(np.uint8)
           
            
            
            propsA = measure.regionprops(tempd, intensity_image=ima)
            propsV = measure.regionprops(tempd - temp, intensity_image=imj)

            data.append(propsA[0].mean_intensity if propsA else 0.000) # mean_intensity_a
            data.append(propsV[0].mean_intensity if propsV else 0.000) # mean_intensity_v
            data.append(0.000) # mean_intensity_f

            # Measure proportions on boundaries, because o is only boundaries.
            propsJ = measure.regionprops(tempd, intensity_image=o)
            propsL=None
            if iml is not None:
                propsL = measure.regionprops(tempd, intensity_image=iml)
            pixel_values = propsJ[0].intensity_image if propsJ else np.array([])
            
          
            n1 = np.sum(pixel_values == 1)
            n2 = np.sum(pixel_values == 2)
            
            rl1=rl2=0.000
            if reduced:
                tot = n1 + n2
                if propsL is not None:
                    if n1>0:
                        rl1=np.mean(propsL[0].intensity_image[pixel_values==1])
                    if n2>0:
                        rl2=np.mean(propsL[0].intensity_image[pixel_values==2])

            else:
                n3 = np.sum(pixel_values == 3)
                tot = n1 + n2 + n3
            #print(n1,n2,n3)
            data.append(n1 / tot if tot else 0.000) # fraction_1
            data.append(n2 / tot if tot else 0.000) # fraction_2
            if not reduced:
                data.append(n3 / tot if tot else 0.000) # fraction_3
            else:
                data.append(0.000)
            data.append(rl1)
            data.append(rl2)
            data.append(prop.centroid[0])
            data.append(prop.centroid[1])
            data.append(celltype)
            data.append(j)
            data.append(k)
            celldata.append(data)
       
        return(celldata)





def match_points_from_ims(j,o,ot,ima, imj, celltype, reduced=0):

    
  
    cdp=analyze_cell(j,o,ima, imj, None, celltype, reduced)
    cdt=analyze_cell(j,ot,ima, imj,None, celltype, reduced)
    if len(cdt)==0:
        return None, None, None, None, None, None, None
    cdt=np.array(cdt)
    
    centt=np.float32(cdt[:,10:12])
    centp=None
    if len(cdp)>0:
        cdp=np.array(cdp)
        centp=np.float32(cdp[:,10:12])
  
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
    
    centp=np.float32(cdp[:,10:12])
    centt=np.float32(cdt[:,10:12])
    
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






def analyze_p(device,model_name_a, model_name_o,model_name_l=None, target='test/',reduced=0,gt=False,dfp=None,thr=.1,thr_p=.8,datapath='data/'):
    celldata = []
 
    ii=get_file_numbers(datapath+target)
    blank=np.zeros(15,dtype=np.int32)
    ii=np.sort(ii)
    ii=np.unique(ii)
    print(ii)
    for i,j in enumerate(ii):
        ima, imj, imo, iml, imj_p,imo_p, iml_p, celltype \
        =process_files(device,j,model_name_a,model_name_o,model_name_leak=model_name_l, gt=gt,datapath=datapath)
        
        if iml is not None:
            iml=(iml>0).astype(np.float32)
            iml_p=(iml_p>0).astype(np.float32)
        
        if gt and imo is not None:
            
            o=imo
        else:
            o=imo_p
            
        if gt:
            imleak=iml
        else:
            imleak=iml_p
 
        data=analyze_cell(j,o,ima, imj, imleak, celltype, reduced=reduced) 
        print('data',len(data))
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

            celld=np.atleast_2d(np.array(data))
            

        celldata.append(celld)  
    
    celldata=np.concatenate(celldata,axis=0)
    print(celldata.shape)
    df = pd.DataFrame(celldata)
    if dfp is None:
        df.columns = ['area', 'major_minor_ratio', 'mean_intensity_a', 'mean_intensity_v', 'mean_intensity_f', 'fraction_bdy_p', 'fraction_broken_p', 'fraction_3','leak_on_bdy','leak_on_broken','centroid_x','centroid_y','celltype','image_idx','cell_idx']
    else:
      df.columns=['area', 'major_minor_ratio', 'mean_intensity_a', 'mean_intensity_v', 'mean_intensity_f', 'fraction_bdy', 'fraction_broken', 'fraction_3','leak_on_bdy','leak_on_broke','centroid_x','centroid_y','celltype','image_idx','cell_idx','area_p', 'major_minor_ratio_p', 'mean_intensity_a_p', 'mean_intensity_v_p','mean_intensity_f_p', 'fraction_bdy_p', 'fraction_broken_p', 'fraction_3_p','leak_on_bdy_p','leak_on_broke_p','centroid_x_p','centroid_y_p','celltype_p','image_idx_p','cell_idx_p']
    
    convert_dict={}
    for i,d in enumerate(df.columns):
        if i != 12 and i!=27:
             convert_dict[d]=float
        else:
             convert_dict[d]=str

    df=df.astype(convert_dict)
    tmp = df.select_dtypes(include=[np.number])
    df.loc[:, tmp.columns] = np.round(tmp,decimals=2)
    
    
    return df, iml, iml_p
