from matplotlib.mlab import detrend
from scipy import interpolate
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, start_time = None):
        self.capture_window = 300  # number of frames to consider
        self.sliding_window_size = 10  # number of frames to remove when exceeding capture window
        
        self.captures = []  # r, g, b, time
        self.start_time = start_time
        self.estimations = []

    def add_frame(self, r, g, b, time):
        if r is not None and g is not None and b is not None:
            if len(self.captures) >= self.capture_window:
                print("Removing oldest capture")
                self.captures = self.captures[self.sliding_window_size:]  # remove oldest 10 captures
            
            self.captures.append((r, g, b, time - self.start_time))
            print(f"Added capture: R={r}, G={g}, B={b}, Time={time - self.start_time if self.start_time else time}")
        
    def length(self):
        print(f"Length of captures: {len(self.captures) if self.captures else 0}")
        return len(self.captures) if self.captures else 0

    def estimate(self):
        if len(self.captures) < self.capture_window:
            print("Not enough data to estimate BPM.")
            return None

        # unpack and normalize
        red, green, blue, times = np.array(self.captures).T
        
        # implementation of POS algorithm described in the fllowing paper:
        # https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf

        # safety check: prevent division by zero if camera is dark
        red = red / (np.mean(red) + 1e-6)
        green = green / (np.mean(green) + 1e-6)
        blue = blue / (np.mean(blue) + 1e-6)

        # POS algorithm
        s1 = green - blue
        s2 = green + blue - 2 * red
        
        # safety check: avoid division by zero if s2 is flat
        std_s2 = np.std(s2)
        alpha = np.std(s1) / std_s2 if std_s2 > 1e-6 else 0
        
        raw_signal = s1 + alpha * s2

        # S = s1 + alpha * s2
        # fs = len(self.captures) / (self.captures[-1].time - self.captures[0].time)  # sampling frequency
        # print(f"Sampling frequency: {fs:.2f} Hz")

        # interpolation (resample to 30 Hz uniformly)
        # we want to map the jittery camera frames to a perfect 30Hz timeline
        raw_times = np.array([t for t in times])
            
        # target sampling frequency 
        fs = 30.0 
        uniform_times = np.arange(raw_times[0], raw_times[-1], 1/fs)
        
        # interpolate the POS signal onto this grid
        interpolator = interpolate.interp1d(raw_times, raw_signal, kind='cubic', fill_value="extrapolate")
        S = interpolator(uniform_times)
        
        # detrend to remove slow drift (DC offset/light changes)
        # S = detrend(S)
        
        # n_samples = len(S)

        # bandpass filter, from 0.7 Hz to 3 Hz (42-180 BPM)
        fmin_hz, fmax_hz = 0.7, 3  
        nyquist = 0.5 * fs
        b, a = butter(2, [fmin_hz / nyquist, fmax_hz / nyquist], btype='band')
        S_filtered = filtfilt(b, a, S)

        # fast fourier transform
        # use hamming window to reduce fake peaks at edges
        n_samples = len(S_filtered)
        windowed = S_filtered * np.hamming(n_samples)
        Y = rfft(windowed)
        freqs = rfftfreq(n_samples, d=1/fs)
        
        # convert to BPM and magnitude
        freqs_bpm = freqs * 60.0
        mag = np.abs(Y)

        # peak selection in the specific band 
        band_mask = (freqs_bpm >= 42) & (freqs_bpm <= 180)
        
        if not np.any(band_mask):
            return 0.0

        band_mag = mag[band_mask]
        band_freqs = freqs_bpm[band_mask]

        # find peaks in the specific band
        peaks, properties = find_peaks(band_mag, height=0)
        
        if len(peaks) == 0:
            return 0.0

        # pick the strongest peak
        best_peak_idx = np.argmax(band_mag[peaks])
        peak_bpm = band_freqs[peaks[best_peak_idx]]

        # store the estimate for smoothing
        self.estimations.append(peak_bpm)

        # keep only last 7 estimates for smoothing
        if len(self.estimations) > 7:
            self.estimations = self.estimations[-7:]  

        smoothed_bpm = np.mean(self.estimations)  

        # save FFT plot with BPM axis
        plt.figure()
        plt.plot(freqs_bpm, mag)
        plt.xlim(fmin_hz * 60, fmax_hz * 60)
        plt.xlabel("Frequency (BPM)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.savefig(f"fft_pos.png")
        plt.close()

        return smoothed_bpm