# -*- coding: utf-8 -*-

import os.path
import scipy.io as sio

import mim.config

N_FILES = 12
SEED_FILE = "/home/sapfo/andersb/ekg_sample/U2_03528.mat"
OUTPUTDIR = os.path.join("test_data", "fake_carlsson_ecg")


def filename_generator(root_dir):
    for k in range(N_FILES):
        file = os.path.join(root_dir, OUTPUTDIR, f"fake_ecg_{k:02d}.mat")
        yield file


if __name__ == "__main__":
    ecg_mat = sio.loadmat(SEED_FILE)
    os.mkdir(OUTPUTDIR)
    for i, outfile in enumerate(filename_generator(mim.config.ROOT_PATH)):
        f = i + 1
        ecg_mat['Data']['ECG'][0][0].fill(f)
        ecg_mat['Data']['Beat'][0][0].fill(f)
        sio.savemat(outfile, ecg_mat)
        print(f"Wrote {outfile}")
