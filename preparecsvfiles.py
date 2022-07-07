#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:26:55 2022

@author: sujiwosa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    meta = pd.read_csv('./data1/Meta/meta_info.csv')
    df = pd.read_csv('./data1/Meta/meta_info.csv', usecols = ['original_image','mask_image','is_clean'])
    rslt_df = df[df['is_clean'] == False]
    rslt_df.to_csv('nodule_test.csv',columns=['original_image'],index=False, header=False)
    rslt_df.to_csv('nodulemask_test.csv',columns=['mask_image'],index=False, header=False)

   # df = pd.read_csv('./data1/Meta/meta_info.csv', usecols = ['original_image','mask_image','is_clean'])
    rslt_df = df[df['is_clean'] == True]
    rslt_df.to_csv('nonnodule_test.csv',columns=['original_image'],index=False, header=False)
    rslt_df.to_csv('nonnodulemask_test.csv',columns=['mask_image'],index=False, header=False)    