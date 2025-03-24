# For splitting the data into train and test before training the model.
# Import necessary libraries.
import os
import numpy as np
import shutil
import pandas as pd
from tqdm import tqdm
import datetime
import sys

# Source root directory path (dataset path).
root_dir = "/home/binu/data/sample_data_10_class/Augmented_datas/"
# Destination directory name.
new_root = '/AllDatasets/'

# Datetime for logging purpose.
now = datetime.datetime.now()
formatted_date = now.strftime('%B-%d-%Y_%H:%M')

# Table to analyse train-test split image stats.
df=pd.DataFrame(columns=['Class','Total_Images','Train_Images','Test_Images'])

# Display the classes (subfolders) in the dataset.
classes = os.listdir(root_dir)
print('Total Classes :',len(classes),'\nClasses = ',classes)

# For avoiding the splitting of data again; as a csv file is created once the data have been splitted. 
if any(file.endswith('.csv') for file in classes):
	print("Already splitted")
	sys.exit()

# Creating directory for 'train' and 'test' images.
for cls in classes:
    os.makedirs(root_dir + new_root + 'train/' + cls, exist_ok=True)
    os.makedirs(root_dir +new_root + 'test/' + cls, exist_ok=True)

# Looping through each classes.
for cls in tqdm(classes,desc='Splitting the data...'):
    src= root_dir +'/'+ cls

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)

    # 75% for training and 25% for testing.
    test_FileNames,train_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.25)])
    test_FileNames = [src+'/' + name for name in test_FileNames]
    train_FileNames = [src+'/' + name for name in train_FileNames]
    
    # updating the stat table.
    pos=len(df)		
    df.loc[pos]=[cls,len(allFileNames),len(train_FileNames),len(test_FileNames)]
   
    # copying the files to respective train or test folder.
    for name in train_FileNames:
        shutil.copy(name, root_dir + new_root+'train/'+cls )
    for name in test_FileNames:
         shutil.copy(name,root_dir + new_root+'test/'+cls)

# Saving the table as csv file for later use.
df.to_csv(f'{root_dir}/info_{formatted_date}.csv',index=False)
print(df)

