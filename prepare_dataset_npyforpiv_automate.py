import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image
import copy
import SimpleITK as sitk

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

print("config read")
#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
# MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
# IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
# CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
# CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
# META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

MASK_DIR = "./data_pivnpy_4x4/Mask/"
META_DIR = "./data_pivnpy_4x4/Meta/"
#DICOM_DIR = "./datapiv/Mask/"
#MASK_DIR
IMAGE_DIR="./data_pivnpy_4x4/Image/"
CLEAN_DIR_IMAGE="./data_pivnpy_4x4/Clean/Image/"
CLEAN_DIR_MASK="./data_pivnpy_4x4/Clean/Mask/"
#META_DIR

# print(MASK_DIR)
# print(META_DIR)
# print(DICOM_DIR) 
# print(MASK_DIR) 
# print(IMAGE_DIR)
# print(CLEAN_DIR_IMAGE) 
# print(CLEAN_DIR_MASK) 
# print(META_DIR) 

# if not os.path.exists(IMAGE_DIR):
#     os.makedirs(IMAGE_DIR)
# if not os.path.exists(MASK_DIR):
#     os.makedirs(MASK_DIR)
# if not os.path.exists(CLEAN_DIR_IMAGE):
#     os.makedirs(CLEAN_DIR_IMAGE)
# if not os.path.exists(CLEAN_DIR_MASK):
#     os.makedirs(CLEAN_DIR_MASK)
# if not os.path.exists(META_DIR):
#     os.makedirs(META_DIR)  


#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#print(mask_threshold)

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
#print(confidence_level)
padding = parser.getint('pylidc','padding_size')
#print(padding)

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])


    def create_npy_from_mhd(self):
        prefix = [str(x).zfill(3) for x in range(1000)]
         # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)       

        S_MASK_DIR = "./data_pivmhd_4x4/Mask/"
        S_META_DIR = "./data_pivmhd_4x4/Meta/"
        #DICOM_DIR = "./datapiv/Mask/"
        #MASK_DIR
        S_IMAGE_DIR="./data_pivmhd_4x4/Image/"
        S_CLEAN_DIR_IMAGE="./data_pivmhd_4x4/Clean/Image/"
        S_CLEAN_DIR_MASK="./data_pivmhd_4x4/Clean/Mask/"
        print("GODFREY")
        #self.IDRI_list=self.IDRI_list[:32]

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            print(pid)
      #      vol = scan.to_volume()
       #     print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = Path(S_IMAGE_DIR) / pid
            patient_mask_dir = Path(S_MASK_DIR) / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)


            # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
            # This current for loop iterates over total number of nodules in a single patient
    #            mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
     #           lung_np_array = vol[cbbox]

           # We calculate the malignancy information
            
           # nodule_slice=0
            slice=0
            if len(nodules_annotation) > 0:
                print("UNCLEAN")
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    # nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                    # mask_name = "{}/{}_MA{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                    #print(nodule_name)
                    # print(mask_name)
                    linex = "{}/{}_NI{}".format(pid,pid[-4:],prefix[nodule_idx])
                    liney = "{}/{}_MA{}".format(pid,pid[-4:],prefix[nodule_idx])
                    #print(linex)
                    #print(liney)
                    #linex=nodule_name
                    #liney=mask_name
                    linesplitx=linex.split("/")
                    linedirx=linesplitx[0]
                    linefilx=linesplitx[1]   
                    lungimgnamex=S_IMAGE_DIR+"/"+linex+".mhd"
                    lungimgx=sitk.ReadImage(lungimgnamex) 
                    lungimgx= sitk.GetArrayFromImage(lungimgx)
                    linesplity=liney.split("/")
                    linediry=linesplity[0]
                    linefily=linesplity[1]
                    #print(linediry)
                    #print(linefily)
                    lungimgnamey=S_MASK_DIR+"/"+liney+".mhd"
                    lungimgy=sitk.ReadImage(lungimgnamey) 
                    lungimgy= sitk.GetArrayFromImage(lungimgy)                 
                    (ix,jx,kx)=lungimgx.shape
                    for index in range(ix):
                        lungimgslicex=lungimgx[index,:,:]
                        #print(lungimgslicex.shape)
                        if not os.path.exists(self.img_path+"/"+linedirx):
                          os.makedirs(self.img_path+"/"+linedirx) 
                        lungimgnmx=self.img_path+linex;
                        nodule_namex = "{}_slice{}".format(lungimgnmx,prefix[index])
                        #nodule_namex = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])   
                      #slice_namex = "_slice{}".format(prefix[index])
                        # print("nodulename")
                        # print(nodule_namex)
                        np.save(Path(nodule_namex),lungimgslicex)
                        lungimgslicey=lungimgy[index+1,:,:]
                        #print(lungimgslicey.shape)
                        if not os.path.exists(self.mask_path+"/"+linediry): #TOCHANGE
                              os.makedirs(self.mask_path+"/"+linediry)     #TOCHANGE
                        lungimgnmy=self.mask_path+liney;  #TOCHANGE
                        nodule_namey = "{}_slice{}".format(lungimgnmy,prefix[index])
                        #nodule_namey = "{}/{}_MA{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        #slice_namey = "_slice{}".format(prefix[index-1])
                        # print("maskname")
                        # print(nodule_namey)
                        np.save(Path(nodule_namey),lungimgslicey)                    
                        meta_list = [pid[-4:],nodule_idx,prefix[index],nodule_namex,nodule_namey,malignancy,cancer_label,False]
                        self.save_meta(meta_list)              
            else: 
               # nodule_name = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
               # mask_name = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
               # print(nodule_name)
               # print(mask_name) 
               print("CLEAN")
               # linex = "{}/{}_CN{}".format(pid,pid[-4:],prefix[nodule_idx])
               # liney = "{}/{}_CM{}".format(pid,pid[-4:],prefix[nodule_idx])
               linex = "{}/{}_CN".format(pid,pid[-4:])
               liney = "{}/{}_CM".format(pid,pid[-4:])               
               print(linex)
               print(liney)
               #linex=nodule_name
               #liney=mask_name
               linesplitx=linex.split("/")
               linedirx=linesplitx[0]
               linefilx=linesplitx[1]   
               lungimgnamex=S_CLEAN_DIR_IMAGE+"/"+linex+".mhd"
               lungimgx=sitk.ReadImage(lungimgnamex) 
               lungimgx= sitk.GetArrayFromImage(lungimgx)
               linesplity=liney.split("/")
               linediry=linesplity[0]
               linefily=linesplity[1]
               print(linediry)
               print(linefily)
               lungimgnamey=S_CLEAN_DIR_MASK+"/"+liney+".mhd"
               lungimgy=sitk.ReadImage(lungimgnamey) 
               lungimgy= sitk.GetArrayFromImage(lungimgy)                 
               (ix,jx,kx)=lungimgx.shape
               for index in range(ix):
                   lungimgslicex=lungimgx[index,:,:]
                   print(lungimgslicex.shape)
                   if not os.path.exists(self.clean_path_img+"/"+linedirx):
                       os.makedirs(self.clean_path_img+"/"+linedirx) 
                   lungimgnmx=self.clean_path_img+linex;
                   nonnodule_namex = "{}_slice{}".format(lungimgnmx,prefix[index])
                   #nodule_namex = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
                   #slice_namex = "_slice{}".format(prefix[index])
                   print("clean nodulename")
                   print(nonnodule_namex)
                   np.save(Path(nonnodule_namex),lungimgslicex)
                      
                   lungimgslicey=lungimgy[index+1,:,:]
                   print(lungimgslicey.shape)
                   if not os.path.exists(self.clean_path_mask+"/"+linediry): #TOCHANGE
                       os.makedirs(self.clean_path_mask+"/"+linediry)     #TOCHANGE
                   lungimgnmy=self.clean_path_mask+liney;  #TOCHANGE
                   nonnodule_namey = "{}_slice{}".format(lungimgnmy,prefix[index])
                   #nodule_namey = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
                   #slice_namey = "_slice{}".format(prefix[index-1])
                   print("clean maskname")
                   print(nonnodule_namey)
                   np.save(Path(nonnodule_namey),lungimgslicey)                    
               #    meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_namex,nodule_namey,malignancy,cancer_label,False]
                   meta_list = [pid[-4:],slice,prefix[index],nonnodule_namex,nonnodule_namey,0,False,True]
                   self.save_meta(meta_list)                     
        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)                    
                    
                    
                    
                    
                    
                    
                    
        #          with open(S_META_DIR+"/"+'nodulemhd.csv') as f1, open(S_META_DIR+"/"+'nodulemaskmhd.csv') as f2:
        #       for x, y in zip(f1, f2)
        #           linex=x.strip()
        #           liney=y.strip()
        #           print(f'line {count}: {line}')
        #           linesplitx=linex.split("/")
        #           linedirx=linesplitx[0]
        #           linefilx=linesplitx[1]
        #           print(linedirx)
        #           print(linefilx)
        #           lungimgnamex=S_IMAGE_DIR+"/"+linex+".mhd"
        #           lungimgx=sitk.ReadImage(lungimgnamex) 
        #           lungimgx= sitk.GetArrayFromImage(lungimgx)
        #           linesplity=liney.split("/")
        #           linediry=linesplity[0]
        #           linefily=linesplity[1]
        #           print(linediry)
        #           print(linefily)
        #           lungimgnamey=S_MASK_DIR+"/"+liney+".mhd"
        #           lungimgy=sitk.ReadImage(lungimgnamey) 
        #           lungimgy= sitk.GetArrayFromImage(lungimgy)
        #           print(type(lungimgx))
        #           print(lungimgx.shape)
        #           (ix,jx,kx)=lungimgx.shape
            
        #             for index in range(ix):
        #                 lungimgslicex=lungimgx[index,:,:]
        #                   print(lungimgslicex.shape)
        #                   # This itereates through the slices of a single nodule
        #                   # Naming of each file: NI= Nodule Image, MA= Mask Original
        #                   #nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
        #                   if not os.path.exists(self.img_path+"/"+linedirx):
        #                    os.makedirs(self.img_path+"/"+linedirx) 
                    
        #                   lungimgnmx=self.img_path+linex;
        #                   print(lungimgnmx)
        #                   nodule_namex = "{}_slice{}".format(lungimgnmx,prefix[index])
        #                   slice_namex = "_slice{}".format(prefix[index])
        #                   print("nodulename")
        #                   print(nodule_namex)
        #                   np.save(Path(nodule_namex),lungimgslicex)
        #                   # np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
        #                       #if index == 0:
                      
                      
        #                lungimgslicey=lungimgy[index+1,:,:]
        #                   print(lungimgslicey.shape)
        #                # This itereates through the slices of a single nodule
        #                # Naming of each file: NI= Nodule Image, MA= Mask Original
        #                #nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
        #                if not os.path.exists(self.mask_path+"/"+linediry): #TOCHANGE
        #                    os.makedirs(self.mask_path+"/"+linediry)     #TOCHANGE
        #                   lungimgnmy=self.mask_path+line;  #TOCHANGE
                    
        #                nodule_namey = "{}_slice{}".format(lungimgnmy,prefix[index-1])
        #                slice_namey = "_slice{}".format(prefix[index-1])
        #                print("nodulename")
        #                print(nodule_namey)
        #                np.save(Path(nodule_namey),lungimgslicey)
                
        #                 meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_namex,nodule_namey,malignancy,cancer_label,False]

        #                 self.save_meta(meta_list)
                
                
                
                



        # #For the CLEAN folder
        
        # with open(S_META_DIR+"/"+'nonnodulemhd.csv') as f:
        #     lines = f.readlines()
        #     count = 0
        #     for line in lines:
        #         count += 1
        #         line = line.strip()

        #         print(f'line {count}: {line}')
                
        #         linesplit=line.split("/")
        #         linedir=linesplit[0]
        #         linefil=linesplit[1]
        #         print(linedir)
        #         print(linefil)
                
        #         lungimgname=S_CLEAN_DIR_IMAGE+"/"+line+".mhd" #TOCHANGE
        #         lungimg=sitk.ReadImage(lungimgname) 
        #         lungimg= sitk.GetArrayFromImage(lungimg)
        #         print(type(lungimg))
        #         print(lungimg.shape)
        #         (i,j,k)=lungimg.shape
        #         #print(i)
        #         for index in range(i):
        #             lungimgslice=lungimg[index,:,:]
        #             print(lungimgslice.shape)
        #             # This itereates through the slices of a single nodule
        #             # Naming of each file: NI= Nodule Image, MA= Mask Original
        #             #nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
        #             if not os.path.exists(self.clean_path_img+"/"+linedir): #TOCHANGE
        #                 os.makedirs(self.clean_path_img+"/"+linedir)   #TOCHANGE
        #             lungimgnm=self.clean_path_img+line; #TOCHANGE
        #             print(lungimgnm)
        #             nodule_name = "{}_slice{}".format(lungimgnm,prefix[index])
        #             slice_name = "_slice{}".format(prefix[index])
        #             print("nodulename")
        #             print(nodule_name)
        #             np.save(Path(nodule_name),lungimgslice)
        #             # np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
                    
        # with open(S_META_DIR+"/"+'nonnodulemaskmhd.csv') as f:
        #     lines = f.readlines()
        #     count = 0
        #     for line in lines:
        #         count += 1
        #         line = line.strip()
        #         print(f'line {count}: {line}')

        #         linesplit=line.split("/")
        #         linedir=linesplit[0]
        #         linefil=linesplit[1]
        #         print(linedir)
        #         print(linefil)
                
        #         lungimgname=S_CLEAN_DIR_MASK+"/"+line+".mhd" #TOCHANGE
        #         #print(lungimgname)
        #         lungimg=sitk.ReadImage(lungimgname) 
        #         lungimg= sitk.GetArrayFromImage(lungimg)
        #         print(type(lungimg))
        #         print(lungimg.shape)
        #         (i,j,k)=lungimg.shape
        #         #print(i)
        #         for index in range(i):
        #             if index == 0:
        #                 continue
        #             lungimgslice=lungimg[index,:,:]
        #             print(lungimgslice.shape)
        #             # This itereates through the slices of a single nodule
        #             # Naming of each file: NI= Nodule Image, MA= Mask Original
        #             #nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
        #             if not os.path.exists(self.clean_path_mask+"/"+linedir): #TOCHANGE
        #                 os.makedirs(self.clean_path_mask+"/"+linedir)     #TOCHANGE
        #             lungimgnm=self.clean_path_mask+line;  #TOCHANGE
                 
        #             nodule_name = "{}_slice{}".format(lungimgnm,prefix[index-1])
        #             slice_name = "_slice{}".format(prefix[index-1])
        #             print("nodulename")
        #             print(nodule_name)
        #             np.save(Path(nodule_name),lungimgslice)
                    
        #         # maskimgname=str(patient_mask_dir)+"/"+mask_name+".mhd"
        #         # sitk.WriteImage(lungimg, lungimgname)  
        #         # sitk.WriteImage(maskimg, maskimgname)    

    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)
        self.IDRI_list=self.IDRI_list[408:]
        #print(self.IDRI_list)

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            print(pid)
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            print(pid)
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)

                    extraframeprev=np.reshape(vol[:,:,cbbox[2].start-1], (512,512,1))
                    extraframeafter=np.reshape(vol[:,:,cbbox[2].stop+1], (512,512,1))
                    lung_np_array = vol[cbbox]

                    extramaskframeprev=np.full((512,512,1), 0)
                    extramaskframeafter=np.full((512,512,1), 0)
                    mask_np_array = copy.copy(mask)

                    lung_np_array = np.concatenate((extraframeprev, lung_np_array, extraframeafter), axis=2)
                    mask_np_array = np.concatenate((extramaskframeprev, mask_np_array, extramaskframeafter), axis=2)
                    #print(lung_np_array.shape)
                    #print(mask_np_array.shape)

                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    
                    segmentedlungarray = []  
                    #segmentedlungarray=None
                    for nodule_slice in range(mask_np_array.shape[2]):
                        #print(nodule_slice)
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        # if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                        #     print("continued")
                        #     continue
                        # Segment Lung part only
                        lung_segmented_np_array = segment_lung(lung_np_array[:,:,nodule_slice])
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
                        lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        lung_segmented_np_array=np.reshape(lung_segmented_np_array, (512,512,1))
                       # segmentedlungarray=np.append(segmentedlungarray,lung_segmented_np_array,2)
                        segmentedlungarray.append(lung_segmented_np_array)
                        
                    lung_segmented_np_array = np.asarray(segmentedlungarray)
                    lung_segmented_np_array=np.squeeze(lung_segmented_np_array)
                    lung_segmented_np_array=np.moveaxis(lung_segmented_np_array, 0, -1)
                    nodule_slice=0
                    # This itereates through the slices of a single nodule
                    # Naming of each file: NI= Nodule Image, MA= Mask Original
                   # nodule_name = "{}/{}_NI{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                   # mask_name = "{}/{}_MA{}_slice{}".format(pid,pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                    nodule_name = "{}/{}_NI{}".format(pid,pid[-4:],prefix[nodule_idx])
                    mask_name = "{}/{}_MA{}".format(pid,pid[-4:],prefix[nodule_idx])                    
                    meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_name,mask_name,malignancy,cancer_label,False]

                    self.save_meta(meta_list)
                    nodule_name = nodule_name.split("/")
                    nodule_name=nodule_name[1]
                    print(nodule_name)
                    mask_name = mask_name.split("/")
                    mask_name = mask_name[1]                          
                    
                    
                    lung_segmented_np_array=np.moveaxis(lung_segmented_np_array, -1, 0)
                    mask_np_array=np.moveaxis(mask_np_array, -1, 0)
                    
                    lungimg = sitk.GetImageFromArray(lung_segmented_np_array)
                    maskimg = sitk.GetImageFromArray(mask_np_array)
                    lungimgname=str(patient_image_dir)+"/"+nodule_name+".mhd"
                    maskimgname=str(patient_mask_dir)+"/"+mask_name+".mhd"
                    sitk.WriteImage(lungimg, lungimgname)  
                    sitk.WriteImage(maskimg, maskimgname)  
 
                    # np.save(patient_image_dir / nodule_name,lung_segmented_np_array)
                    # np.save(patient_mask_dir / mask_name,mask[:,:,nodule_slice])
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                segmentedlungarray = [] 
                maskarray = []
                for slice in range(vol.shape[2]):
                    if slice >50:
                        break
                    #print(slice)
                    lung_segmented_np_array = segment_lung(vol[:,:,slice])
                    lung_segmented_np_array[lung_segmented_np_array==-0] =0
                    lung_mask = np.zeros_like(lung_segmented_np_array)
                    
                    lung_segmented_np_array=np.reshape(lung_segmented_np_array, (512,512,1))
                    lung_mask=np.reshape(lung_mask, (512,512,1))
                    segmentedlungarray.append(lung_segmented_np_array)
                    maskarray.append(lung_mask)


                lung_segmented_np_array = np.asarray(segmentedlungarray)
                lung_segmented_np_array=np.squeeze(lung_segmented_np_array)
                lung_segmented_np_array=np.moveaxis(lung_segmented_np_array, 0, -1)
                slice=0

                mask_np_array = np.asarray(maskarray)
                mask_np_array=np.squeeze(mask_np_array)
                mask_np_array=np.moveaxis(mask_np_array, 0, -1)

                #CN= CleanNodule, CM = CleanMask
                # nodule_name = "{}/{}_CN001_slice{}".format(pid,pid[-4:],prefix[slice])
                # mask_name = "{}/{}_CM001_slice{}".format(pid,pid[-4:],prefix[slice])
                nodule_name = "{}/{}_CN".format(pid,pid[-4:])
                mask_name = "{}/{}_CM".format(pid,pid[-4:])                
                meta_list = [pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True]
                # print(nodule_name)
                # print(mask_name)
                # print(meta_list)    
                # print(patient_clean_dir_image)
                # print(patient_clean_dir_mask)
                self.save_meta(meta_list)
                # print(lung_segmented_np_array.shape)
                # print(patient_clean_dir_image)
                # print(nodule_name)
                nodule_name = nodule_name.split("/")
                nodule_name=nodule_name[1]
                mask_name = mask_name.split("/")
                mask_name = mask_name[1]                    
                # print(nodule_name)
                # print(mask_name)
                # print(patient_clean_dir_image / nodule_name)
                # np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
                # np.save(patient_clean_dir_mask / mask_name, lung_mask)

                lung_segmented_np_array=np.moveaxis(lung_segmented_np_array, -1, 0)
                mask_np_array=np.moveaxis(mask_np_array, -1, 0)

                lungimg = sitk.GetImageFromArray(lung_segmented_np_array)
                maskimg = sitk.GetImageFromArray(mask_np_array)
                lungimgname=str(patient_clean_dir_image)+"/"+nodule_name+".mhd"
                maskimgname=str(patient_clean_dir_mask)+"/"+mask_name+".mhd"
                sitk.WriteImage(lungimg, lungimgname)  
                sitk.WriteImage(maskimg, maskimgname)  

        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)



if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    print(DICOM_DIR)
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()
  #   print("LIDC")
  #   print(LIDC_IDRI_list)
  #   LIDC_IDRI_list=LIDC_IDRI_list[803:]    
    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
  #  test.prepare_dataset()
    test.create_npy_from_mhd()
