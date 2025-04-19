import numpy as np
from utils.lpf import lp_filter_zero_phase
from scipy.signal import butter, sosfilt, resample, iirnotch, tf2sos


def preproc(signal, fc, fs, fs_old, order=64, Q_factor=30, freq=50):
    # Design the filter
    normalized_cutoff = fs / fs_old
    sos = butter(order, normalized_cutoff, btype='low', output='sos')

    # Design the notch filter
    w0 = freq / (fs / 2)  # Normalize frequency to Nyquist
    b, a = iirnotch(w0, Q_factor)
    sos_notch = tf2sos(b, a)

    # Prepare to store the downsampled signal
    num_samples_new = int(signal.shape[1] * fs / fs_old)  # New time dimension
    preprocessed_signal = np.zeros((signal.shape[0], num_samples_new))

    # Process each channel independently
    nChannels = signal.shape[0]
    for ch in range(nChannels):
        # Apply low-pass filter
        lpf_signal = signal[ch] - lp_filter_zero_phase(signal[ch], fc/fs_old)
        # Apply the antialiasing filter
        filtered_signal = sosfilt(sos, lpf_signal)
        # Resample
        downsampled_signal = resample(filtered_signal, num_samples_new)
        # Apply the notch filter
        preprocessed_signal[ch] = sosfilt(sos_notch, downsampled_signal)

    return preprocessed_signal
