import random
import os
import shutil

def random_copyfile(img_path, train_path, test_path, num):
    folder_name_list = list(os.path.join(name) for name in os.listdir(img_path))
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    for folder_name in folder_name_list:
    	folderPath = os.path.join(img_path, folder_name)
        name_list = list(os.path.join(folderPath, name) for name in os.listdir(folderPath))
        name_list = random.shuffle(name_list)

        trainFolderPath = os.path.join(train_path, folder_name)
        os.makedirs(trainFolderPath)
        for oldname in name_list[:num]:
            shutil.copy(oldname, oldname.replace(folderPath, trainFolderPath))
        
        testFolderPath = os.path.join(test_path, folder_name)
        os.makedirs(testFolderPath)
        for oldname in name_list[num:]:
            shutil.copy(oldname, oldname.replace(folderPath, testFolderPath))

srcPath = 'HOSPITAL_2_560/train'
dstPath = 'HOSPITAL_2_560/test'
random_copyfile(srcPath, dstPath, 280)
