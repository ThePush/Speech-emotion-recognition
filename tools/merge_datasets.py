import numpy as np
import os


def main():
    x_ravdess = np.load(
        'np_arrays/RAVDESS/x_RAVDESS.npy')
    y_ravdess = np.load(
        'np_arrays/RAVDESS/y_RAVDESS.npy')

    x_crema = np.load('np_arrays/CREMA/x_CREMA.npy')
    y_crema = np.load('np_arrays/CREMA/y_CREMA.npy')
    
    x_tess = np.load('np_arrays/TESS/x_TESS.npy')
    y_tess = np.load('np_arrays/TESS/y_TESS.npy')

    x_merged = np.concatenate((x_ravdess, x_crema, x_tess), axis=0)
    y_merged = np.concatenate((y_ravdess, y_crema, y_tess), axis=0)

    if not os.path.exists('np_arrays/RAVDESS_CREMA_TESS'):
        os.mkdir('np_arrays/RAVDESS_CREMA_TESS')

    np.save(
        'np_arrays/RAVDESS_CREMA_TESS/x_RAVDESS_CREMA_TESS.npy', x_merged)
    np.save(
        'np_arrays/RAVDESS_CREMA_TESS/y_RAVDESS_CREMA_TESS.npy', y_merged)


if __name__ == '__main__':
    main()
