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
        self.ica_channels = None
        self.ica_converged = None
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
    
    def pos_fft(self):
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

    def ica_fft(self):
        # Run ICA (output shape: n_samples, n_components)
        ica = FastICA(n_components=3, max_iter=100)
        rgb_channels = np.array([[cap.red, cap.green, cap.blue] for cap in self.captures])
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', ConvergenceWarning)
                ica_channels = ica.fit_transform(rgb_channels)

                # check for convergence warnings, sometimes it does not converge and the results are not valid
                conv_warnings = [warn for warn in w if issubclass(warn.category, ConvergenceWarning)]
                if conv_warnings:
                    print("Warning: FastICA did not converge within max_iter. Consider increasing max_iter.")
                    self.ica_converged = False
                else:
                    self.ica_converged = True
        except Exception as e:
            print(f"ICA fitting failed with exception: {e}")
            self.ica_converged = False
            return None

        # Check for NaNs in the ICA output just in case
        if np.isnan(ica_channels).any():
            print("ICA transformation produced NaNs.")
            self.ica_converged = False
            return None
        
        self.ica_channels = ica_channels  # keep samples x components
        n_samples, n_components = self.ica_channels.shape
        print("total channels: ", n_components)

        # timing: use actual frame timestamps
        times = np.array([cap.time for cap in self.captures])
        if len(times) < 2:
            print("Not enough timestamps.")
            return None
        mean_dt = np.mean(np.diff(times))    # seconds between frames
        fs = 1.0 / mean_dt
        print(f"fs = {fs:.2f} Hz (dt = {mean_dt:.4f} s)")

        freqs = fftfreq(n_samples, d=mean_dt)[:n_samples // 2]
        freqs_bpm = freqs * 60.0

        # physiological band in Hz and BPM
        fmin_hz, fmax_hz = 0.8, 3.0         # ~48 - 180 BPM
        bpm_estimates = []

        for i in range(n_components):
            channel = self.ica_channels[:, i].astype(float)
            channel = channel - np.mean(channel)            # remove DC
            b, a = butter(2, [fmin_hz / (0.5 * fs), fmax_hz / (0.5 * fs)], btype='band')
            channel = filtfilt(b, a, channel)
            windowed = channel * np.hamming(n_samples)     # reduce leakage

            Y = fft(windowed)
            mag = 2.0 / n_samples * np.abs(Y[:n_samples // 2])

            # restrict to physiological band
            band_mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
            if not np.any(band_mask):
                print(f"Channel {i}: no frequencies in band.")
                continue

            band_mag = mag[band_mask]
            band_freqs_bpm = freqs_bpm[band_mask]

            peaks, peaks_dict = find_peaks(band_mag, height=(None, None), prominence=(None, None), width=(None, None), plateau_size=(None, None))
            print(f"Channel {i} peaks at BPMs: {band_freqs_bpm[peaks]} with magnitudes: {band_mag[peaks]}")
            print(f"Channel {i} peak prominence: {peaks_dict.get('prominences')}")

            # compute relative power of each peak (relative to total band power)
            total_band_power = np.sum(band_mag)
            if total_band_power <= 0:
                print(f"Channel {i}: non-positive total band power.")
                continue

            peak_rel_power = band_mag[peaks] / total_band_power
            # pick the single peak with maximum relative power for this channel
            max_idx = int(np.argmax(peak_rel_power))
            peak_bpm = float(band_freqs_bpm[peaks[max_idx]])
            rel_power = float(peak_rel_power[max_idx])

            bpm_estimates.append((peak_bpm, rel_power))
            print(f"Channel {i} selected peak: {peak_bpm:.1f} BPM with rel power {rel_power:.3f}")


            # save FFT plot with BPM axis
            plt.figure()
            plt.plot(freqs_bpm, mag)
            plt.xlim(fmin_hz * 60, fmax_hz * 60)
            plt.xlabel("Frequency (BPM)")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.savefig(f"fft_channel_{i}.png")
            plt.close()

        if not bpm_estimates:
            print("No BPM estimates found.")
            return None

        # choose the frequency from the channel whose selected peak has the highest relative power
        best = max(bpm_estimates, key=lambda t: t[1])
        bpm = float(best[0])
        print(f"Estimated BPM (best channel): {bpm:.1f} (rel power {best[1]:.3f})")
        return bpm

    def estimate(self):
        if len(self.captures) < self.capture_window:
            print("Not enough data to estimate BPM.")
            return None

        # bpm = self.ica_fft()
        bpm = self.pos_fft()

        return bpm

    def plot_channels(self):
        if self.ica_channels is None:
            print("No ICA channels to plot.")
            return
        
        import matplotlib.pyplot as plt

        time_axis = [cap.time for cap in self.captures]

        plt.figure(figsize=(12, 8))
        for i in range(self.ica_channels.shape[1]):
            plt.subplot(self.ica_channels.shape[1], 1, i + 1)
            plt.plot(time_axis, self.ica_channels[:, i])
            plt.title(f'ICA Channel {i + 1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig('ica_channels.png')