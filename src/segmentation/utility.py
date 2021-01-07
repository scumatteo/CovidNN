import numpy as np
import shutil
import os
from glob import glob
from sklearn.model_selection import train_test_split



#funzione che ritorna i nomi delle immagini, splittando il dataset in 80% training e 20% validation
def get_train_valid_names():
    img_paths = glob('/content/XRAYChestSegmentation/originals/*.png')
    names = [x.split('/')[-1] for x in img_paths]
    print('Numero totale di immagini del dataset XRAYChestSegmentation e rispettive maschere: ', len(names))

    X = names
    train_X,valid_X = train_test_split(X,test_size=0.2,random_state=42)
    return train_X,valid_X

#funzione di utility per copiare un insieme di immagini da una sorgente a una destinazione
def copy_images(in_paths,out_paths):
    for src,dst in zip(in_paths,out_paths):
        shutil.copy(src,dst)

#funzione per copiare tutte le immagini di training e validation set nelle cartelle create
#in modo da poter utilizzare la flow_from_directory di Keras per l'addestramento 
def copy_sets(train,valid,base_path):
    img_in_paths = [''.join(['/content/XRAYChestSegmentation/originals/',name]) for name in train]
    img_out_paths = [''.join([base_path,'train/images/input/',name]) for name in train]
    copy_images(img_in_paths,img_out_paths)

    mask_in_paths = [''.join(['/content/XRAYChestSegmentation/masks/',name]) for name in train]
    mask_out_paths = [''.join([base_path,'train/masks/input/',name]) for name in train]
    copy_images(mask_in_paths,mask_out_paths)

    img_in_paths = [''.join(['/content/XRAYChestSegmentation/originals/',name]) for name in valid]
    img_out_paths = [''.join([base_path,'validation/images/input/',name]) for name in valid]
    copy_images(img_in_paths,img_out_paths)

    mask_in_paths = [''.join(['/content/XRAYChestSegmentation/masks/',name]) for name in valid]
    mask_out_paths = [''.join([base_path,'validation/masks/input/',name]) for name in valid]
    copy_images(mask_in_paths,mask_out_paths)

#funzione di utility che stampa il numero di file contenuti nelle sottocartelle di base_path
def get_stats(base_path):
    for root, directories,filenames in os.walk(base_path):
        if len(filenames)>0:
            print (root,":",len(filenames))