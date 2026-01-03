from matplotlib.mlab import detrend
from scipy import interpolate
from sklearn.decomposition import FastICA
import numpy as np
import time
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings

class Capture:
    def __init__(self, r = None, g = None, b = None, time = None):
        self.red = r
        self.green = g
        self.blue = b
        self.time = time


class Estimator:
    def __init__(self, start_time = None):
        self.capture_window = 150  # number of frames to consider
        self.captures = []
        self.start_time = start_time
        self.estimations = []

    def add_frame(self, r, g, b, time):
        if r is not None and g is not None and b is not None:
            if len(self.captures) >= self.capture_window:
                print("Removing oldest capture")
                self.captures = self.captures[20:]  # remove oldest 20 captures
            
            self.captures.append(Capture(r, g, b, time - self.start_time))
            print(f"Added capture: R={r}, G={g}, B={b}, Time={time - self.start_time if self.start_time else time}")
        
    def length(self):
        print(f"Length of captures: {len(self.captures)}")
        return len(self.captures)

    def estimate(self):
        if len(self.captures) < self.capture_window:
            print("Not enough data to estimate BPM.")
            return None

        # 1. Unpack and Normalize (Your code was correct here)
        red, green, blue = np.array([[cap.red, cap.green, cap.blue] for cap in self.captures]).T
        
        # Safety check: Prevent division by zero if camera is dark
        red = red / (np.mean(red) + 1e-6)
        green = green / (np.mean(green) + 1e-6)
        blue = blue / (np.mean(blue) + 1e-6)

        # 2. POS Algorithm (Your math is correct)
        s1 = green - blue
        s2 = green + blue - 2 * red
        
        # Safety check: avoid divide by zero if s2 is flat
        std_s2 = np.std(s2)
        alpha = np.std(s1) / std_s2 if std_s2 > 1e-6 else 0
        
        raw_signal = s1 + alpha * s2
        
        # 3. INTERPOLATION (The Missing Link)
        # We must map the jittery camera frames to a perfect 30Hz timeline
        raw_times = np.array([cap.time for cap in self.captures])
        
        if len(raw_times) < 30: # Need at least ~1 second of data
            return 0.0
            
        # Create a perfect grid: 30 fps from start to end
        fs = 30.0 
        uniform_times = np.arange(raw_times[0], raw_times[-1], 1/fs)
        
        # Interpolate the POS signal onto this grid
        interpolator = interpolate.interp1d(raw_times, raw_signal, kind='cubic', fill_value="extrapolate")
        S = interpolator(uniform_times)
        
        # Detrend to remove slow drift (DC offset/light changes)
        S = detrend(S)
        
        n_samples = len(S)

        # 4. Bandpass Filter 
        fmin_hz, fmax_hz = 0.8, 3  # Expanded slightly to catch edge cases
        nyquist = 0.5 * fs
        b, a = butter(2, [fmin_hz / nyquist, fmax_hz / nyquist], btype='band')
        S_filtered = filtfilt(b, a, S)

        # 5. FFT
        # Use Hamming window to reduce spectral leakage
        windowed = S_filtered * np.hamming(n_samples)
        Y = rfft(windowed)
        freqs = rfftfreq(n_samples, d=1/fs)
        
        # Convert to BPM and Magnitude
        freqs_bpm = freqs * 60.0
        mag = np.abs(Y)

        # 6. Peak Selection Logic
        band_mask = (freqs_bpm >= 45) & (freqs_bpm <= 200)
        
        if not np.any(band_mask):
            return 0.0

        band_mag = mag[band_mask]
        band_freqs = freqs_bpm[band_mask]

        # Find peaks in the specific band
        peaks, properties = find_peaks(band_mag, height=0)
        
        if len(peaks) == 0:
            return 0.0

        # Pick the strongest peak
        # Optimization: "prominences" is often better than just "height" for noisy data
        # but height is fine for a start.
        best_peak_idx = np.argmax(band_mag[peaks])
        peak_bpm = band_freqs[peaks[best_peak_idx]]

        # save FFT plot with BPM axis
        plt.figure()
        plt.plot(freqs_bpm, mag)
        plt.xlim(fmin_hz * 60, fmax_hz * 60)
        plt.xlabel("Frequency (BPM)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.savefig(f"fft_pos.png")
        plt.close()

        return peak_bpm