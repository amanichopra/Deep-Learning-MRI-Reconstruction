import numpy as np
import glob
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import argparse

def load_miccai(path, num_files_to_load=float('inf'), n_zero_ratio = 0.1, max_num_volumes=float('inf'), max_num_slices=float('inf')): # 
    data_list = glob.glob(path + '/*.nii.gz')[:num_files_to_load]
    data = []
    for file in data_list:
        img = nib.load(file).get_fdata()
        
        num_vols = min(max_num_volumes, img.shape[3])
        for vol in range(num_vols):
            num_slcs = min(max_num_slices, img.shape[2])
           
            for slc in range(num_slcs):
                if np.count_nonzero(img[:, :, slc, vol]) / np.prod(img[:, :, slc, vol].shape) >= n_zero_ratio:
                    data.append(img[:, :, slc, vol].reshape(img.shape[0], img.shape[1], 1))            
    return np.array(data)

def unsample_data(data, mask, accelaration_factor=5):
    uns_data = []
    for i in range(data.shape[0]):
        fourier = np.fft.fft2(data[i,:,:])
        cen_fourier  = np.fft.fftshift(fourier)
        subsam_fourier = np.multiply(cen_fourier,mask) # undersampling in k-space
        uncen_fourier = np.fft.ifftshift(subsam_fourier)
        zro_image = np.fft.ifft2(uncen_fourier) # zero-filled reconstruction
        uns_data.append(zro_image)        
    return np.array(uns_data)

def unsample_data_add_noise(data, mask, noise_ratio, accelaration_factor=5):
    fft_data = []
    uns_data = []
    for i in range(data.shape[0]):
        fourier = np.fft.fft2(data[i,:,:])
        cen_fourier  = np.fft.fftshift(fourier)
        fft_data.append(cen_fourier)
    fft_data = np.array(fft_data)
    fft_std = np.std(fft_data)
    
    nstd = (noise_ratio*fft_std)/np.sqrt(2)
    insh = (fft_data.shape[1],fft_data.shape[2])
    
    for i in range(fft_data.shape[0]):    
        fft_noise=fft_data[i,:,:]+np.random.normal(0,nstd,insh)+1j*np.random.normal(0,nstd,insh) #adding noise
        subsam_fourier = np.multiply(fft_noise,mask) # undersampling in k-space
        uncen_fourier = np.fft.ifftshift(subsam_fourier)
        zro_image = np.fft.ifft2(uncen_fourier) # zero-filled reconstruction
        uns_data.append(zro_image) 
        
    return np.array(uns_data)

def get_logger(path):
    # Create a custom logger
    logger = logging.getLogger()

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'{path}/load_preprocess.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)
    
    return logger

def configCLIArgparser():
    parser = argparse.ArgumentParser(description='Load and preprocess data.')
    parser.add_argument('--micaii_path', type=str, help='Path to MICAII dataset.', default='./data/raw/canine-legs/training-images')
    parser.add_argument('--mask_path', type=str, help='Path to mask used for undersampling.', default='./masks/mask_1dg_a5.pickle')
    parser.add_argument('--save_path', type=str, help='Path to save raw and undersampled imgages.', default='./data')
    parser.add_argument('--num_files', type=int, help='Number of files to load.', default=1)
    parser.add_argument('--max_num_slices', type=int, help='Maximum number of slices to load for volume.', default=25)
    parser.add_argument('--max_num_volumes', type=int, help='Path to save raw and undersampled imgages.', default=1)
    parser.add_argument('--save', type=str, help='Boolean indicating whether to save preprocessed imgages.', default='True')
                        
    args = parser.parse_args()
    args.save = eval(args.save)                    
    return args

def main(args, logger):
    logger.info('Loading MICAII dataset...')
    imgs = load_miccai(args.micaii_path, num_files_to_load=args.num_files, max_num_slices=args.max_num_slices, max_num_volumes=args.max_num_volumes)
    
    logger.info('Loading mask...')
    with open(args.mask_path, 'rb') as f:
        mask = pickle.load(f)   
    
    X1, X2 = train_test_split(imgs[:, :, :, 0], shuffle=False)
    X2, X3, X4, X5 = np.array_split(X2, 4)
    
    logger.info('Constructing undersampled dataset...')
    uns_imgs_1 = unsample_data(X1, mask)
    uns_imgs_2 = unsample_data_add_noise(X2, mask, 0.1) #10% noise-overlapping
    uns_imgs_3 = unsample_data_add_noise(X3, mask, 0.2) #20% noise-overlapping
    uns_imgs_4 = unsample_data_add_noise(X4, mask, 0.1) #10% noise-nonoverlapping
    uns_imgs_5 = unsample_data_add_noise(X5, mask, 0.2) #20% noise-nonoverlapping

    uns_imgs_final = np.concatenate([uns_imgs_1, uns_imgs_2, uns_imgs_3, uns_imgs_4, uns_imgs_5])
    uns_imgs_final = uns_imgs_final.reshape((uns_imgs_final.shape[0], uns_imgs_final.shape[1], uns_imgs_final.shape[2], 1))           
    if args.save:  
        logger.info('Saving images...')
        with open(f'{args.save_path}/training_usamp.pickle', 'wb') as f:
            pickle.dump(uns_imgs_final, f, protocol=4)
        
        with open(f'{args.save_path}/training.pickle', 'wb') as f:
            pickle.dump(imgs, f, protocol=4)
    logger.info('Complete!')
                        
if __name__ == '__main__':
    logger = get_logger('./logs')
    logger.info('Parsing CLI args...')
    args = configCLIArgparser()
    main(args, logger)                   
                        