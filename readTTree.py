#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
description: Convert GEANT4 root files into hdf5 files
author: Zhengting He
"""
import ROOT as root
import numpy as np
import sys
import h5py

def momentum_to_energy(Px, Py, Pz):
    electron_mass = 0.51099895##Unit : Mev
    return np.sqrt(Px**2 + Py**2 + Pz**2 +electron_mass**2)

def read_and_write(infile, outfile):
    f = root.TFile.Open(infile,"READ")
    myTree = f.Get("dp")
    if not myTree:
        print("Failed to get TTree")
        sys.exit(1)
    entries = myTree.GetEntries()
    print ("Number of entries :", entries)

    data_to_write = np.zeros(shape=(entries,20,20))
    energy_to_write = np.zeros(entries)

    for entry in range(entries):
        myTree.GetEntry(entry)
        #image = myTree.ECAL_Hits
        image = myTree.ECAL_E_CellXY
        np_image = np.asarray(image)
        np_image = np_image.reshape(20,20)
        data_to_write[entry] = np_image
        Initial_Px = myTree.Initial_Px
        Initial_Py = myTree.Initial_Py
        Initial_Pz = myTree.Initial_Pz
        energy = momentum_to_energy(Initial_Px, Initial_Py, Initial_Pz)
        energy_to_write[entry] = energy


    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset("ECAL_centre", data=data_to_write)
        hf.create_dataset("Energy", data=energy_to_write)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert root to HDF5 files')

    parser.add_argument('--in-file', '-i', action="store", required=True,
                        help='input ROOT file')

    parser.add_argument('--out-file', '-o', action="store", required=True,
                        help='output hdf5 file')

    args = parser.parse_args()
    read_and_write(args.in_file, args.out_file)