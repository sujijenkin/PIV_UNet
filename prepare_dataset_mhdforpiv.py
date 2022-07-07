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

MASK_DIR = "./datapiv/Mask/"
META_DIR = "./datapiv/Meta/"
#DICOM_DIR = "./datapiv/Mask/"
#MASK_DIR
IMAGE_DIR="./datapiv/Image/"
CLEAN_DIR_IMAGE="./datapiv/Clean/Image/"
CLEAN_DIR_MASK="./datapiv/Clean/Mask/"
#META_DIR

print(MASK_DIR)
print(META_DIR)
print(DICOM_DIR) 
print(MASK_DIR) 
print(IMAGE_DIR)
print(CLEAN_DIR_IMAGE) 
print(CLEAN_DIR_MASK) 
print(META_DIR) 

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

print(mask_threshold)

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
print(confidence_level)
padding = parser.getint('pylidc','padding_size')
print(padding)

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
        #self.IDRI_list=self.IDRI_list[408:]
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
    test.prepare_dataset()
