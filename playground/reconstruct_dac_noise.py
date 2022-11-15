import numpy as np
from matplotlib import pyplot as plt

noise_coords = np.array([(78, 185), (104, 238), (162, 286), (193, 285), (307, 348), (392, 384), (547, 393),
                         (705., 394), (861, 402), (934, 424)])
spike_coords = np.array([(797, 309), (824, 335), (622, 361), (343, 153.)])


def translate_coords_to_data(coords: np.ndarray, interp: bool = True, n_points: int = 10 ** 4, BOTTOM_LEFT=(78, 578)):
    coords[:, 0] -= BOTTOM_LEFT[0]
    coords[:, 1] = BOTTOM_LEFT[1] - coords[:, 1]

    x_unit = 235 - 78
    y_unit = 0.5 * (578 - 157)

    coords[:, 0] /= x_unit
    coords[:, 1] /= y_unit

    if interp:
        log_freqs = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_points)
        log_amps_per_sqrt_hz = np.interp(log_freqs, coords[:, 0], coords[:, 1])
    else:
        log_freqs = coords[:, 0]
        log_amps_per_sqrt_hz = coords[:, 1]

    freqs = 10 ** log_freqs
    amps_per_sqrt_hz = 10 ** log_amps_per_sqrt_hz

    return freqs, amps_per_sqrt_hz


freqs, amps_per_sqrt_hz = translate_coords_to_data(noise_coords)
spike_freqs, spike_amps_per_sqrt_hz = translate_coords_to_data(spike_coords, interp=False)

for spike_freq, spike_amp in zip(spike_freqs, spike_amps_per_sqrt_hz):
    spike_freq_ind = np.argmin(abs(freqs - spike_freq))
    amps_per_sqrt_hz[spike_freq_ind] = spike_amp

plt.plot(freqs * 1e6, amps_per_sqrt_hz)
plt.semilogx()
plt.semilogy()

plt.xlabel('Frequency[MHz]')
plt.ylabel('Voltage[nV/$\sqrt{Hz}$]')
plt.title('DAC noise power spectral density')

n_freqs = 2 * 10 ** 7
linspace_freqs = np.linspace(1, max(freqs), n_freqs)
linsp_amps_per_sqrt_hz = np.interp(linspace_freqs, freqs, amps_per_sqrt_hz)

linsp_amps_per_sqrt_hz *= 1e-9
signal_psd = linsp_amps_per_sqrt_hz ** 2

N = len(linspace_freqs)
phase = 2 * np.pi * np.random.randn(N)
magnitude = np.sqrt(signal_psd)
FFT = magnitude * np.exp(1j * phase)

plt.figure()
plt.plot(linspace_freqs, np.abs(FFT))
plt.plot(linspace_freqs, magnitude)
plt.title('DAC Noise PSD')
plt.ylabel('Voltage[V/$\sqrt{Hz}$]')

delta_f = max(freqs)
inv_samp_rate = 1 / delta_f
dt = inv_samp_rate


# taken from https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python
def noise_psd(N, psd=lambda f: 1, d=1):
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N, d))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(d=1):
    def psd_generator(f):
        return lambda N: noise_psd(N, f, d)

    return psd_generator


@PSDGenerator(inv_samp_rate)
def get_DAC_noise(f: np.ndarray) -> np.ndarray:
    return np.interp(f, freqs, amps_per_sqrt_hz)


def get_noise_time_signal(n_samples: int) -> tuple:
    ts = dt * np.arange(n_samples)
    return ts, get_DAC_noise(n_samples)


ts, noise = get_noise_time_signal(1000)
plt.figure()
plt.plot(ts, noise)
plt.xlabel('t')
plt.ylabel('Amplitude[V]')

plt.show()
