# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
# from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from datetime import datetime
from tensorflow.keras.layers import (Input, Dense, Reshape, Conv2D, LeakyReLU,
                                     BatchNormalization, LocallyConnected2D,
                                     Activation, ZeroPadding2D, Lambda, Flatten,
                                     Embedding, ELU, Dropout, UpSampling2D, Cropping2D, LayerNormalization,
                                     )

from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import time
import h5py
# MBD import
# from keras.engine import InputSpec, Layer
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.keras.layers import Lambda, ZeroPadding2D, LocallyConnected2D

# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import shuffle
import pandas as pd

from tqdm import tqdm

import deepdish as dd

from hep_ml import reweight
from hep_ml.metrics_utils import ks_2samp_weighted

import uproot3
import uproot
from datetime import datetime
import pytz
import awkward0

###############
#import webpage
particle_options = {
    "Inclusive": "/content/drive/MyDrive/Data/dp_ana_Inclusive.hdf5",
    "GMM_ECAL": "/content/drive/MyDrive/Data/dp_ana_GMM_ECAL.hdf5",
    "GMM_Target": "/content/drive/MyDrive/Data/dp_ana_GMM_Target.hdf5",
    "PN_ECAL": "/content/drive/MyDrive/Data/dp_ana_PN_ECAL.hdf5",
    "PN_Target": "/content/drive/MyDrive/Data/dp_ana_PN_Target.hdf5",

    "Inclusive_4e6": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/Inclusive_ana_4e6.tfrecord",
    "GMM_ECAL_8e5": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/GMM_ECAL_ana_8e5.hdf5",
    "GMM_Target_8e5": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/GMM_Target_ana_8e5.hdf5",
    "PN_ECAL_4e5": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/PN_ECAL_ana_4e5.hdf5",
    "PN_Target_4e5": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/PN_Target_ana_4e5.hdf5",

    "Inclusive_cut_7GeV": "/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/Inclusive_ana_4e6_cut_at_7GeV.hdf5",
    'Inclusive_larger_than_7GeV': '/content/drive/Shareddrives/desmond.z.he1998.HK(CRN.NGO)/TrainingData/Inclusive_ana_4e6_cut_larger_then_7GeV.tfrecord'
}


def print_arg(args):
    for key in vars(args).keys():
        print(key, "is set to be", vars(args)[key])


def read_file(infile, particle):
    # Function for hdf5 file
    # h5file = h5py.File(infile, 'r')
    h5file = dd.io.load(infile)
    ECAL_centre = h5file['ECAL_centre']
    Energy = h5file["Energy"]
    sizes = ECAL_centre.shape
    print("There are {} events with {} x {} layout for {}".format(sizes[0], sizes[1], sizes[2], particle))
    # y = [particle] * ECAL_centre.shape[0]

    return ECAL_centre, Energy, sizes


def parse_tfr_element(element):
    data = {
        'images': tf.io.FixedLenFeature([], tf.string),
        'energy': tf.io.FixedLenFeature([], tf.float32),
    }

    content = tf.io.parse_single_example(element, data)
    raw_image = content['images']
    energy = content['energy']

    feature = tf.io.parse_tensor(raw_image, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[20, 20, 1]) / 1000
    energy = tf.reshape(energy, shape=[1]) / 1000
    return {'images': feature,
            'energy': energy
            }


def get_dataset_small(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LOGAN training and deploying')

    parser.add_argument('--in_file', '-i', action="store", type=str,
                        default="/lustre/collider/hezhengting/TrainningData/PN_ECAL_ana_4e5.hdf5",
                        help='Training dataset')

    parser.add_argument('--disc_lr', action="store", type=float, default=1e-3,
                        help='Discriminator initial learning rate')

    parser.add_argument('--end_learning_rate', action="store", type=float, default=1e-4,
                        help='Discriminator end learning rate')

    parser.add_argument('--disc_opt', action="store", type=str, default="Adam",
                        help='Discriminator optimizer')

    parser.add_argument('--adam_beta_1', action="store", type=float, default=0.5,
                        help='Beta 1 of Adam optimizer')

    parser.add_argument('--adam_beta_2', action="store", type=float, default=0.9,
                        help='Beta 2 of Adam optimizer')

    parser.add_argument('--decay_steps', action="store", type=int, default=1000,
                        help='Decay steps of learning rate')

    parser.add_argument('--decay_power', action="store", type=float, default=2,
                        help='Decay power of learning rate')

    parser.add_argument('--decay_rate', action="store", type=float, default=0.9,
                        help='Decay rate of learning rate')

    parser.add_argument('--gen_lr', action="store", type=float, default=5e-4,
                        help='Generator initial learning rate')

    parser.add_argument('--gen_opt', action="store", type=str, default="Nadam",
                        help='Optimizer of generator')

    parser.add_argument('--energy_cut', action="store", type=float, default=1e-3,
                        help='Energy cut in the unit of MeV')

    parser.add_argument('--generator_extra_step', action="store", type=int, default=1,
                        help='Steps of generator in each train step')

    parser.add_argument('--discriminator_extra_steps', action="store", type=int, default=1,
                        help='Steps of discriminator in each train step')

    parser.add_argument('--batch_size', action="store", type=int, default=600,
                        help='Batch size')

    parser.add_argument('--final_layer_activation', action="store", type=str, default="softplus",
                        help='Type of function on final layer of generator')

    parser.add_argument('--z_alpha', action="store", type=float, default=0.9,
                        help='Z_alpha in latent optimization')

    parser.add_argument('--z_beta', action="store", type=float, default=0.1,
                        help='Z_beta in latent optimization')

    parser.add_argument('--g_network_type', action="store", type=str, default="DownSampling",
                        help='Type of generator network')

    parser.add_argument('--use_latent_optimization', action="store", type=bool, default=True,
                        help='Whether to use latent optimization')

    parser.add_argument('--lambda_E', action="store", type=float, default=1e2,
                        help='lambda_E in loss function of generator')

    parser.add_argument('--E_factor', action="store", type=float, default=0,
                        help='E factor in loss function of generator')

    parser.add_argument('--latent_size', action="store", type=int, default=1024,
                        help='Dimension of noise from latent space')

    args = parser.parse_args()

    in_file = args.in_file

    for key in particle_options.keys():
        if key in in_file:
            particle_type = key

    disc_lr = args.disc_lr
    disc_opt = args.disc_opt  # @param ["Adam", "Nadam"]
    adam_beta_1 = args.adam_beta_1  # @param {type:"number"}
    adam_beta_2 = args.adam_beta_2  # @param {type:"number"}
    decay_steps = args.decay_steps  # @param {type:"slider", min:100, max:2000, step:100}
    decay_power = args.decay_power  # @param ["0.5", "1", "2"] {type:"raw"}
    decay_rate = args.decay_rate  # @param {type:"number"}
    gen_lr = args.gen_lr  # @param {type:"number"}
    gen_opt = args.gen_opt  # @param ["Adam", "Nadam"]
    energy_cut = args.energy_cut  # @param {type:"number"}
    generator_extra_step = args.generator_extra_step  # @param {type:"integer"}
    discriminator_extra_steps = args.discriminator_extra_steps  # @param {type:"integer"}
    batch_size = args.batch_size  # @param {type:"slider", min:100, max:1000, step:100}
    BATCH_SIZE = batch_size
    final_layer_activation = args.final_layer_activation  # @param ["relu", "softplus"]
    z_alpha = args.z_alpha  # @param {type:"number"}
    z_beta = args.z_beta  # @param {type:"number"}
    g_network_type = args.g_network_type  # @param ["UpSampling", "DownSampling"]
    use_latent_optimization = args.use_latent_optimization  # @param ["True", "False"] {type:"raw"}

    end_learning_rate = args.end_learning_rate  # @param {type:"number"}

    lambda_E = args.lambda_E

    E_factor = args.E_factor

    latent_size = args.latent_size

    g_pfx = 'params_generator_epoch_'
    d_pfx = 'params_discriminator_epoch_'
    print_arg(args)
    args_dict = vars(args)
    print(args_dict)

    hdf5_dataset = (in_file[-4:] == 'hdf5')
    tfrecord_dataset = (in_file[-8:] == 'tfrecord')
    if hdf5_dataset:
        ECAL_centre, Energy, sizes = read_file(in_file, particle_type)
        # le = LabelEncoder()
        # y = le.fit_transform(y)
        # print(list(le.classes_))
        # ECAL_centre, Energy, y = shuffle(ECAL_centre, Energy, y, random_state=0)

        train_images = (ECAL_centre.reshape(sizes[0], 20, 20, 1).astype(
            'float32')) / 1000  # The scale of eV should be enough

        # train_images = train_images * (train_images>energy_cut)
        Energy = (Energy.reshape(sizes[0]).astype('float32')) / 1000
        print("The shape of tranning data is", train_images.shape)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices({
                'images': train_images,
                'energy': Energy
            })
                .shuffle(train_images.shape[0], reshuffle_each_iteration=True)
                .batch(batch_size)
            # .prefetch(tf.data.AUTOTUNE)
        )

    if tfrecord_dataset:
        tfrecord_file = in_file
        train_dataset = get_dataset_small(tfrecord_file)
        sizes = [4000000]
        # sizes.append(test_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy())
        train_dataset = train_dataset.shuffle(sizes[0], reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        # train_dataset = train_dataset.cache()
