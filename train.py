import numpy as np
import glob
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Flatten, Add
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal

import logging
import argparse
import time

# loss for discriminator
def accw(y_true, y_pred):
    y_pred=K.clip(y_pred, -1, 1)
    return K.mean(K.equal(y_true, K.round(y_pred)))

# loss for generator to find structured similarity between images
def mssim(y_true, y_pred):
    costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    return costs

# loss for generator
def wloss(y_true,y_predict):
    return -K.mean(y_true*y_predict)

# discriminator model
def discriminator(inp_shape = (256,256,1), trainable = True):
    gamma_init = RandomNormal(1., 0.02)
    
    inp = Input(shape = (256,256,1))
    
    l0 = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp) #b_init is set to none, maybe they are not using bias here, but I am.
    l0 = LeakyReLU(alpha=0.2)(l0)
    
    l1 = Conv2D(64*2, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l0)
    l1 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l1)
    l1 = LeakyReLU(alpha=0.2)(l1)
    
    l2 = Conv2D(64*4, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l1)
    l2 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l2)
    l2 = LeakyReLU(alpha=0.2)(l2)
    
    l3 = Conv2D(64*8, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l2)
    l3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l3)
    l3 = LeakyReLU(alpha=0.2)(l3)
    
    l4 = Conv2D(64*16, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l3)
    l4 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l4)
    l4 = LeakyReLU(alpha=0.2)(l4)
    
    l5 = Conv2D(64*32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l4)
    l5 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l5)
    l5 = LeakyReLU(alpha=0.2)(l5)
    
    l6 = Conv2D(64*16, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l5)
    l6 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l6)
    l6 = LeakyReLU(alpha=0.2)(l6)
    
    l7 = Conv2D(64*8, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l6)
    l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
    l7 = LeakyReLU(alpha=0.2)(l7)
    #x
    
    l8 = Conv2D(64*2, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
    l8 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l8)
    l8 = LeakyReLU(alpha=0.2)(l8)
    
    l9 = Conv2D(64*2, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l8)
    l9 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l9)
    l9 = LeakyReLU(alpha=0.2)(l9)
    
    l10 = Conv2D(64*8, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l9)
    l10 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l10)
    l10 = LeakyReLU(alpha=0.2)(l10)
    #y
    l11 = Add()([l7,l10])
    l11 = LeakyReLU(alpha = 0.2)(l11)
    
    out=Conv2D(filters=1,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l11)
    model = Model(inputs = inp, outputs = out)
    return model

# RRDB (residual in residual dense block at bottom of unet to increase depth of network)
def resden(x,fil,gr,beta,gamma_init,trainable):   
    # concatentations create residual connections between consecutive blocks
    x1=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x1=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x1)
    x1=LeakyReLU(alpha=0.2)(x1)
    
    x1=Concatenate(axis=-1)([x,x1])
    
    x2=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x1)
    x2=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x2)
    x2=LeakyReLU(alpha=0.2)(x2)

    x2=Concatenate(axis=-1)([x1,x2])
        
    x3=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x2)
    x3=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3=LeakyReLU(alpha=0.2)(x3)

    x3=Concatenate(axis=-1)([x2,x3])
    
    x4=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x3)
    x4=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x4)
    x4=LeakyReLU(alpha=0.2)(x4)

    x4=Concatenate(axis=-1)([x3,x4])
    
    x5=Conv2D(filters=fil,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x4)
    x5=Lambda(lambda x:x*beta)(x5)
    xout=Add()([x5,x])
    
    return xout

def resresden(x,fil,gr,betad,betar,gamma_init,trainable):
    x1=resden(x,fil,gr,betad,gamma_init,trainable)
    x2=resden(x1,fil,gr,betad,gamma_init,trainable)
    x3=resden(x2,fil,gr,betad,gamma_init,trainable)
    x3=Lambda(lambda x:x*betar)(x3)
    xout=Add()([x3,x])
    
    return xout

# generator model
def generator(inp_shape, trainable = True): 
    gamma_init = RandomNormal(1., 0.02)

    fd=512
    gr=32
    nb=12
    betad=0.2
    betar=0.2
    
    ## contracting phase in unet 
    inp_real_imag = Input(inp_shape)
    lay_128dn = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_real_imag)

    lay_128dn = LeakyReLU(alpha = 0.2)(lay_128dn)

    lay_64dn = Conv2D(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128dn)
    lay_64dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64dn)
    lay_64dn = LeakyReLU(alpha = 0.2)(lay_64dn)

    lay_32dn = Conv2D(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64dn)
    lay_32dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32dn)
    lay_32dn = LeakyReLU(alpha=0.2)(lay_32dn)

    lay_16dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32dn)
    lay_16dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16dn)
    lay_16dn = LeakyReLU(alpha=0.2)(lay_16dn)  #16x16 

    lay_8dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16dn)
    lay_8dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_8dn)
    lay_8dn = LeakyReLU(alpha=0.2)(lay_8dn) #8x8


    xc1=Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8dn) #8x8
    xrrd=xc1
    
    ## bottle neck with RRDbs
    # add residual connections between pairs of blocks
    for m in range(nb):
        xrrd=resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

    xc2=Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(xrrd)
    lay_8upc=Add()([xc1,xc2])

    ## expansion phase in unet
    lay_16up = Conv2DTranspose(1024, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8upc)
    lay_16up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16up)
    lay_16up = Activation('relu')(lay_16up) #16x16

    lay_16upc = Concatenate(axis = -1)([lay_16up,lay_16dn])

    lay_32up = Conv2DTranspose(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16upc) 
    lay_32up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32up)
    lay_32up = Activation('relu')(lay_32up) #32x32

    lay_32upc = Concatenate(axis = -1)([lay_32up,lay_32dn])

    lay_64up = Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32upc)
    lay_64up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64up)
    lay_64up = Activation('relu')(lay_64up) #64x64

    lay_64upc = Concatenate(axis = -1)([lay_64up,lay_64dn])

    lay_128up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64upc)
    lay_128up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_128up)
    lay_128up = Activation('relu')(lay_128up) #128x128

    lay_128upc = Concatenate(axis = -1)([lay_128up,lay_128dn])

    lay_256up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128upc)
    lay_256up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_256up)
    lay_256up = Activation('relu')(lay_256up) #256x256

    out =  Conv2D(1, (1,1), strides = (1,1), activation = 'tanh', padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_256up)

    model = Model(inputs = inp_real_imag, outputs = out)

    return model

def define_gan_model(gen_model, dis_model, inp_shape):
    dis_model.trainable = False
    inp = Input(shape = inp_shape)
    out_g = gen_model(inp)
    out_dis = dis_model(out_g)
    out_g1 = out_g
    model = Model(inputs = inp, outputs = [out_dis, out_g, out_g1])
    #model.summary()
    return model

def get_logger(path):
<<<<<<< HEAD
    # Ensure path/to/train.log doesn't exist
    if os.path.isfile(f'{path}/train.log'):
        os.remove(f'{path}/train.log')
    
=======
>>>>>>> 0925e39aea6b229541e0c635a12cdb78336111cd
    # Create a custom logger
    logger = logging.getLogger()

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'{path}/train.log')
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
    parser = argparse.ArgumentParser(description='Traing GAN.')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs to train.', default=300)
    parser.add_argument('--n_batch', type=int, help='Number of batches used in training. Ensure this number is > than the number of samples!', default=10)
    parser.add_argument('--n_critic', type=int, help='Number of times to train discriminator/epoch.', default=3)
    parser.add_argument('--clip_val', type=float, help='Value to use for gradient clipping.', default=0.05)
    parser.add_argument('--in_shape_gen', type=str, help='Input shape for generator. Specify as a list of commas without spaces (ex. "256,256,2").', default='256,256,2')
    parser.add_argument('--in_shape_dis', type=str, help='Input shape for discriminator. Specify as a list of commas without spaces (ex. "256,256,1").', default='256,256,1')
    parser.add_argument('--log_device_placement', type=str, help='Boolean indicating whether to specify what device is being used for tensorflow operations.', default='False')
    parser.add_argument('--data_path', type=str, help='Path to training data.', default='./data/training.pickle')
    parser.add_argument('--usam_path', type=str, help='Path to undersampled training data.', default='./data/training_usamp.pickle')
    parser.add_argument('--t_size', type=int, help='Number of images to train with.', default=64)
    parser.add_argument('--epoch_end_im_save_path', type=str, help='Path to save images generated at end of each epoch.', default='./plots/training')
    parser.add_argument('--mod_save_path', type=str, help='Path to save best model based on metric to compare models.', default='./models')
    parser.add_argument('--mod_compare_metric', type=str, choices=['disc_loss', 'gen_loss', 'mae', 'gen_sim_loss', 'gan_loss'], help='Metric to use to compare models for saving.', default='gen_sim_loss')

    args = parser.parse_args()
    args.in_shape_gen = tuple([int(dim) for dim in args.in_shape_gen.split(',')])
    args.in_shape_dis = tuple([int(dim) for dim in args.in_shape_dis.split(',')])
    args.log_device_placement = eval(args.log_device_placement)
    return args
        
def train(g_par, d_par, gan_model, dataset_real, u_sampled_data,  n_epochs, n_batch, n_critic, clip_val, n_patch, logger, im_save_path, mod_save_path, mod_compare_metric):
    bat_per_epo = int(dataset_real.shape[0]/n_batch)
    if bat_per_epo == 0:
        class InvalidNBatchArgumentException(Exception):
            pass
        raise InvalidNBatchArgumentException('--n_batch argument is invalid! Ensure this number is greater than the number of samples.')
    
    d_loss = None
    g_loss = [None, None, None, None]
    mod_compare_metric_mapping = {'disc_loss': 'd_loss', 'gen_loss': 'g_loss[1]', 'mae': 'g_loss[2]', 'gen_sim_loss': 'g_loss[3]', 'gan_loss': 'g_loss[0]'}
    
    half_batch = int(n_batch/2)
    plt.imsave(f'{im_save_path}/real.png', dataset_real[0][:, :, 0], cmap='gray')    
    plt.imsave(f'{im_save_path}/undersampled.png', u_sampled_data[0][:, :, 0], cmap='gray')   
    best_g_loss = float('inf')
    best_mod_weights = {'gen': None, 'disc': None, 'gan': None}
		
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            start = time.time() # start time
            
            # training the discriminator
            for k in range(n_critic):
                ix = np.random.randint(0, dataset_real.shape[0], half_batch)
            
                X_real = dataset_real[ix]
                y_real = np.ones((half_batch,n_patch,n_patch,1))
            
                ix_1 =  np.random.randint(0, u_sampled_data.shape[0], half_batch)
                X_fake  = g_par.predict(u_sampled_data[ix_1])
                y_fake = -np.ones((half_batch,n_patch,n_patch,1))
            
                X, y = np.vstack((X_real, X_fake)), np.vstack((y_real,y_fake))
                d_loss, accuracy = d_par.train_on_batch(X,y)
            
                for l in d_par.layers:
                    weights=l.get_weights()
                    weights=[np.clip(w, -clip_val,clip_val) for w in weights]
                    l.set_weights(weights)

            # training the generator
            ix = np.random.randint(0, dataset_real.shape[0], n_batch)
            X_r = dataset_real[ix]
            X_gen_inp = u_sampled_data[ix]
            y_gan = np.ones((n_batch,n_patch,n_patch,1))
            
            g_loss = gan_model.train_on_batch([X_gen_inp], [y_gan, X_r, X_r])
            end = time.time() # end time
            logger.info(f'Epoch: {i+1}, Time: {(end-start)}, Batch: {j + 1}/{bat_per_epo}, Disc Loss: {d_loss}, Accuracy: {accuracy},  Gen Loss: {g_loss[1]},  MAE: {g_loss[2]},  Gen Sim Loss: {g_loss[3]}, GAN Loss: {g_loss[0]}')
            if eval(mod_compare_metric_mapping[mod_compare_metric]) < best_g_loss: # save best model based on metric
                best_g_loss = eval(mod_compare_metric_mapping[mod_compare_metric])
                logger.info(f'Model has best performance so far based on {mod_compare_metric}...')
                best_mod_weights['gen'] = g_par.get_weights()
                best_mod_weights['disc'] = d_par.get_weights()
                best_mod_weights['gan'] = gan_model.get_weights()
                
            
        logger.info(f'Saving example of image generated at end of epoch {i+1}...') 
        fake = g_par.predict(u_sampled_data[0][None, :, :, :])
        plt.imsave(f'{im_save_path}/fake_{i+1}.png', fake[0, :, :, 0], cmap='gray')
    
    logger.info('Saving best model after training...')
    g_par.set_weights(best_mod_weights['gen'])
    d_par.set_weights(best_mod_weights['disc'])
    gan_model.set_weights(best_mod_weights['gan'])
    
    g_par.save(f'{mod_save_path}/G')
    d_par.save(f'{mod_save_path}/D')
    gan_model.save(f'{mod_save_path}/GAN')

def build_and_train(args, logger):
    if args.log_device_placement: tf.debugging.set_log_device_placement(True)
    
    logger.info('Building discriminator model...')
    d_model = discriminator(inp_shape = args.in_shape_dis, trainable = True)
    #d_model.summary()
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    d_model.compile(loss = wloss, optimizer = opt, metrics = [accw])

    logger.info('Building generator model...')
    g_model = generator(inp_shape = args.in_shape_gen , trainable = True)
    
    logger.info('Building GAN model...')
    gan_model = define_gan_model(g_model, d_model, args.in_shape_gen)
    opt = Adam(lr = 0.0001, beta_1 = 0.5)
    gan_model.compile(loss = [wloss, 'mae', mssim], optimizer = opt, loss_weights = [0.01, 20.0, 1.0]) #loss weights for generator training
    n_patch=d_model.output_shape[1]
    
    logger.info('Loading real dataset...')
    dataset_real = pickle.load(open(args.data_path,'rb'))[:args.t_size] # Ground truth
    dataset_real = dataset_real.reshape(-1, 256, 256, 1)
    logger.info(f'Real dataset loaded with shape: {dataset_real.shape}...')
    logger.info('Loading undersampled dataset...')
    u_sampled_data = pickle.load(open(args.usam_path,'rb'))[:args.t_size]
    u_sampled_data = u_sampled_data.reshape(-1, 256, 256, 1) # # Zero-filled reconstructions
    logger.info(f'Unsampled dataset loaded with shape: {u_sampled_data.shape}...')
    u_sampled_data_2c = np.concatenate((u_sampled_data.real, u_sampled_data.imag), axis = -1)
    
    logger.info('Starting training...')
    train(g_model, d_model, gan_model, dataset_real, u_sampled_data_2c, args.n_epochs, args.n_batch, args.n_critic, args.clip_val, n_patch, logger, args.epoch_end_im_save_path, args.mod_save_path, args.mod_compare_metric)
    logger.info('Training complete!')

if __name__ == '__main__':
    logger = get_logger('./logs')
    logger.info('Parsing CLI args...')
    args = configCLIArgparser()
    logger.info(f"CLI args parsed: {vars(args)}")
    build_and_train(args, logger)
    
