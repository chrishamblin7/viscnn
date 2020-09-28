#functions for formating image_data folders

import os
from subprocess import call


def add_label_from_folder_2_name(folder_path):
    files = os.listdir(folder_path)
    folder = folder_path.split('/')[-1]
    for file in files:
        new_file = folder+'_'+file
        call('mv %s %s'%(os.path.join(folder_path,file),os.path.join(folder_path,new_file)),shell=True)

def add_label_to_files_in_all_category_folders(root_path):
    subfolders = os.listdir(root_path)
    categories = []
    for subfolder in subfolders:
        if os.path.isdir(os.path.join(root_path,subfolder)):
            categories.append(subfolder)
    for category in categories:
        add_label_from_folder_2_name(os.path.join(root_path,category))

def numbers_2_spelling(folder_path):
    target_dict = {'0_':'zero_','1_':'one_','2_':'two_','3_':'one_','4_':'one_',
                    '5_':'five_','6_':'six_','7_':'seven_','8_':'eight_','9_':'nine_'}
    files = os.listdir(folder_path)
    for file in files:
        for key in target_dict:
            if key in file:
                new_file = file.replace(key,target_dict[key])
                call('mv %s %s'%(os.path.join(folder_path,file),os.path.join(folder_path,new_file)),shell=True)
                break
