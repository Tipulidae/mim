import glob

from massage.ecg import to_hdf5


PATH_TO_EXPECT_ECG_HDF5 = '/mnt/air-crypt/air-crypt-expect/axel/ecg.hdf5'


def create_ecg_database():
    """
    Take all the Expect ECG files in .mat format, process them and store in
    hdf5-format. Metadata is included, which checks various aspects of the
    file, including whether the data has the proper format and so on.
    Note that this takes about an hour to run.
    """
    hbg_glob = (
        '/mnt/air-crypt/air-crypt-expect/andersb/data/Expect-HBG-2019-12-04/'
        'JonasCarlssonEKG/HBG_ALL_ECG_mat/*.mat'
    )
    lu_glob = (
        '/mnt/air-crypt/air-crypt-expect/andersb/data/Expect-Lund-2019-09-23/'
        'JonasCarlssonEKG/Expect_Lund_MatFiles/*.mat'
    )

    expect_ecg_paths = list(glob.iglob(hbg_glob)) + list(glob.iglob(lu_glob))
    expect_ecg_paths = list(sorted(expect_ecg_paths))

    to_hdf5(expect_ecg_paths, PATH_TO_EXPECT_ECG_HDF5)
