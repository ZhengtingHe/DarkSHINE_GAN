#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
description: Convert GEANT4 root files into hdf5 files
author: Zhengting He
"""

import numpy as np
import h5py
import uproot

def momentum_to_energy(Px, Py, Pz):
    electron_mass = 0.51099895##Unit : Mev
    return np.sqrt(Px**2 + Py**2 + Pz**2 +electron_mass**2)

def read_with_uproot(infile,outfile,cut_value):
    tree_name = 'dp'
    variables = ['ECAL_E_CellXY', 'ECAL_E_total']
    cuts = 'ECAL_E_total > ' + str(cut_value*1000)
    energy = 8000 #MeV !!!, not GeV
    events = uproot.open(infile+':'+tree_name) 
    data = events.arrays(variables,cuts,library="np")
    num_entries = data[variables[0]].shape[0]
    print('The number of entries is ', num_entries)
    if not outfile:
        outfile = infile[:-5] + '.hdf5'
    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset("ECAL_centre", data=data['ECAL_E_CellXY'].reshape(num_entries,20,20))
        hf.create_dataset("Energy", data= energy * np.ones(num_entries))
        hf.create_dataset("ECAL_E_total", data=data['ECAL_E_total'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert root to HDF5 files')

    parser.add_argument('--in-file', '-i', action="store", type=str, required=True,
                        help='input ROOT file')

    parser.add_argument('--out-file', '-o', action="store", type=str, default=None,
                        help='output hdf5 file')

    parser.add_argument('--cut-energy', '-c', action="store", type=float, default=None,
                        help='energy cut value in GeV')


    args = parser.parse_args()
    read_with_uproot(args.in_file, args.out_file, args.cut_energy)

    


