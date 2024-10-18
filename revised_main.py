#!/usr/bin/env python
from __future__ import print_function, division
import h5py
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import matplotlib.pyplot as plt
from os.path import join
from scipy.io import loadmat
from utils import compressed_sensing as cs
from utils.metric import complex_psnr
from cascadenet.network.model import build_d2_c2
from cascadenet.util.helpers import from_lasagne_format, to_lasagne_format


def prep_input(im, acc=4):
    """Undersample the batch, then reformat them into what the network accepts."""
    mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = to_lasagne_format(im)
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)
    return im_und_l, k_und_l, mask_l, im_gnd_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
                        default=['4.0'], help='Acceleration factor for k-space sampling')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')
    args = parser.parse_args()

    acc = float(args.acceleration_factor[0])
    Nx, Ny = 640, 322  # Image dimensions (change if needed)
    save_fig = args.savefig

    # Load pre-trained model weights
    model_name = 'd2_c2'
    project_root = '.'  # Adjust if your project structure is different
    save_dir = join(project_root, 'models/%s' % model_name)

    # Specify network input shape
    input_shape = (1, 2, Nx, Ny)  # single image, batch size = 1
    net_config, net = build_d2_c2(input_shape)

    # Load pre-trained model parameters
    with np.load('./models/pretrained/d2_c2_epoch_x.npz') as f:  # specify the pre-trained model file
        param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)

    # Load your MRI data
    Un_Sample_data = '/content/Deep-MRI-Reconstruction/file_brain_AXT1POST_207_2070829_undersampled.h5'
    # load original data
    un_sample = h5py.File(Un_Sample_data) # Read HDF5 data (MRI aquired data)
    un_kspace = un_sample['kspace'][:]



    
    # Select one coil and one slice
    coil_index = 1  # Change this if you want to use a different coil
    slice_index = 8  # Change this if you want to use a different slice
    test_image = un_kspace[slice_index, coil_index]  # Shape will be (640, 320)

    # Preprocess the image
    im_und, k_und, mask, im_gnd = prep_input(test_image, acc=acc)

    # Compile the test function
    input_var = net_config['input'].input_var
    mask_var = net_config['mask'].input_var
    kspace_var = net_config['kspace_input'].input_var
    target_var = T.tensor4('targets')

    val_fn = theano.function([input_var, mask_var, kspace_var, target_var],
                             [lasagne.layers.get_output(net)],
                             on_unused_input='ignore')

    # Run the model on your image
    pred = val_fn(im_und, mask, k_und, im_gnd)[0]

    # Convert predictions back to original format
    pred_image = from_lasagne_format(pred)
    im_und_image = from_lasagne_format(im_und)

    # Compute PSNR
    psnr_value = complex_psnr(test_image, pred_image, peak='max')
    print("PSNR: {:.6f}".format(psnr_value))

    # Optionally, save the reconstructed images
    if save_fig:
        plt.imsave(join(save_dir, 'reconstructed_image.png'), abs(pred_image[0]), cmap='gray')
        plt.imsave(join(save_dir, 'undersampled_image.png'), abs(im_und_image[0]), cmap='gray')

    print('Inference completed.')
