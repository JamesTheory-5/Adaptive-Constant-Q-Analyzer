# Adaptive-Constant-Q-Analyzer

awesome—here’s a compact, practical **Adaptive Constant-Q Filterbank Analyzer** that outputs **time–frequency energy tracks** (think: a streaming constant-Q spectrogram). It’s causal, block-based, and fast thanks to NumPy + Numba. You can use it for visualization, metering, and as the “analyzer front-end” that drives adaptive EQ/comp/FX.

---

# Adaptive Constant-Q Analyzer (bands × time)

```python
# adaptive_cq_analyzer.py
import numpy as np
import scipy.signal as sps
from numba import njit

#====================#
# Low-level kernels  #
#====================#

@njit(cache=True, fastmath=True)
def ewma_pow2_env(x, alpha):
    """
    One-pole RMS-like envelope on power: e[n] = alpha*e[n-1] + (1-alpha)*x[n]^2
    Returns sqrt(e[n]) i.e., amplitude envelope.
    """
    N = x.shape[0]
    env = np.empty(N, dtype=np.float64)
    e = 0.0
    a = alpha
    b = 1.0 - alpha
    for n in range(N):
        v = x[n]
        e = a * e + b * (v * v)
        env[n] = np.sqrt(e + 1e-20)
    return env


@njit(cache=True, fastmath=True)
def downsample_mean(x, hop):
    """Downsample by averaging non-overlapping blocks of size `hop` (causal)."""
    N = x.shape[0]
    M = N // hop
    out = np.empty(M, dtype=np.float64)
    s = 0.0
    c = 0
    m = 0
    for n in range(N):
        s += x[n]
        c += 1
        if c == hop:
            out[m] = s / hop
            s = 0.0
            c = 0
            m += 1
    return out


#========================================#
# Adaptive Constant-Q Filterbank (stream)#
#========================================#

class AdaptiveConstantQAnalyzer:
    """
    Streaming constant-Q filterbank analyzer producing per-band energy tracks.

    - Log-spaced bandpass SOS filters (constant Q)
    - Causal per-band envelope follower (EWMA on power, sqrt)
    - Block-based processing with internal filter states (zi)
    - Optional downsample of envelope for compact output

    Outputs: energy matrix shape (n_bands, n_frames), where each column is a time step.
    """

    def __init__(
        self,
        sr=48_000,
        fmin=80.0,
        fmax=12_000.0,
        bands_per_octave=6,
        order=2,            # SOS butter order per band edge (bandpass)
        Q=8.0,              # constant Q (fc / BW)
        env_tau=0.02,       # seconds, EWMA time constant (converted to alpha per block)
        block_size=1024,    # audio samples per processing block
        hop_env=8           # average env in groups of hop_env to reduce columns
    ):
        self.sr = float(sr)
        self.block = int(block_size)
        self.hop_env = int(hop_env)

        # centers (log spaced)
        self.centers = self._log_centers(fmin, fmax, bands_per_octave)

        # make constant-Q bandpass filters as SOS + states
        self.sos_list = []
        self.zi_list = []
        for fc in self.centers:
            sos = self._make_bandpass(fc, Q, order)
            self.sos_list.append(sos)
            # zi per SOS section (2 states per section, stereo=1)
            self.zi_list.append(sps.sosfilt_zi(sos) * 0.0)

        # envelope smoother: alpha for EWMA per-sample; keep per-block use
        # alpha = exp(-1/(tau*sr))
        self.env_alpha = float(np.exp(-1.0 / (max(env_tau, 1e-6) * self.sr)))

        # ring buffers for env downsample accumulation
        self._acc = np.zeros(len(self.centers), dtype=np.float64)
        self._acc_count = 0
        self._frames = []

    # ---------- filterbank construction ----------
    def _log_centers(self, fmin, fmax, bpo):
        fmin = float(fmin); fmax = float(fmax)
        n_oct = np.log2(fmax / fmin)
        n_bands = max(1, int(np.floor(n_oct * bpo)))
        idx = np.arange(n_bands, dtype=np.float64)
        return fmin * (2.0 ** (idx / bpo))

    def _make_bandpass(self, fc, Q, order):
        bw = fc / Q
        lo = max(10.0, fc - bw * 0.5)
        hi = min(self.sr * 0.5 * 0.999, fc + bw * 0.5)
        # guard if lo>=hi
        if hi <= lo:
            hi = min(self.sr * 0.5 * 0.999, lo * 1.1)
        sos = sps.butter(order, [lo, hi], btype='band', fs=self.sr, output='sos')
        return sos

    # ---------- streaming API ----------
    def reset(self):
        """Reset filter and envelope accumulators."""
        self.zi_list = [sps.sosfilt_zi(sos) * 0.0 for sos in self.sos_list]
        self._acc[:] = 0.0
        self._acc_count = 0
        self._frames = []

    def process_block(self, x_block):
        """
        Process a block of mono audio (length == self.block).
        Appends downsampled envelope column(s) internally.
        """
        x_block = np.asarray(x_block, dtype=np.float64)
        assert x_block.ndim == 1 and x_block.shape[0] == self.block, "block must be mono, length=block_size"

        band_env = []

        # 1) filter each band (causal)
        for i, sos in enumerate(self.sos_list):
            y, self.zi_list[i] = sps.sosfilt(sos, x_block, zi=self.zi_list[i])
            # 2) envelope (per-sample EWMA on power, sqrt)
            env = ewma_pow2_env(y, self.env_alpha)
            band_env.append(env)

        # shape: (n_bands, block)
        E = np.vstack(band_env)

        # 3) downsample env along time axis by averaging hop_env samples
        if self.hop_env > 1:
            # for each band, average across hop_env chunks
            M = E.shape[1] // self.hop_env
            if M > 0:
                # accumulate partials across blocks to make columns aligned
                # strategy: do simple within-block downsample; spill remainder into accumulator if needed
                trimmed = E[:, : M * self.hop_env]
                down = trimmed.reshape(E.shape[0], M, self.hop_env).mean(axis=2)
                # append columns
                for k in range(M):
                    self._frames.append(down[:, k].copy())
            # (optional) ignore remainder < hop_env; could carry over if desired
        else:
            # hop_env == 1: push each sample as a "frame" (heavy!)
            for n in range(E.shape[1]):
                self._frames.append(E[:, n].copy())

    def get_matrix(self):
        """
        Return accumulated energy matrix: shape (n_bands, n_frames).
        Also returns the center frequencies and frame rate (frames per second).
        """
        if len(self._frames) == 0:
            return self.centers.copy(), np.zeros((len(self.centers), 0)), 0.0
        M = np.stack(self._frames, axis=1)  # (bands, frames)
        fps = self.sr / (self.block * self.hop_env)
        return self.centers.copy(), M, float(fps)


#====================#
# Convenience (batch)#
#====================#

def analyze_constant_q_energy(
    x,
    sr=48_000,
    fmin=80.0,
    fmax=12_000.0,
    bands_per_octave=6,
    order=2,
    Q=8.0,
    env_tau=0.02,
    block_size=1024,
    hop_env=8
):
    """
    Batch helper: splits x into blocks, runs the streaming analyzer, returns (centers, energy, fps).

    energy: (n_bands, n_frames) constant-Q energy tracks (amplitude envelope per band).
    fps   : frames per second of the energy matrix (time step = 1/fps).
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    B = int(block_size)
    if N % B != 0:
        # pad end with zeros to full block
        pad = B - (N % B)
        x = np.pad(x, (0, pad))

    cq = AdaptiveConstantQAnalyzer(
        sr=sr, fmin=fmin, fmax=fmax, bands_per_octave=bands_per_octave,
        order=order, Q=Q, env_tau=env_tau, block_size=block_size, hop_env=hop_env
    )
    cq.reset()

    for i in range(0, len(x), B):
        cq.process_block(x[i:i+B])

    centers, M, fps = cq.get_matrix()
    return centers, M, fps
```

---

## How to use it (2–3 lines)

```python
import soundfile as sf
from adaptive_cq_analyzer import analyze_constant_q_energy

x, sr = sf.read("input.wav")
if x.ndim > 1:
    x = x.mean(axis=1)

centers, energy, fps = analyze_constant_q_energy(
    x, sr=sr, fmin=80, fmax=12000, bands_per_octave=6,
    Q=8.0, env_tau=0.02, block_size=1024, hop_env=8
)

print("bands:", len(centers), "frames:", energy.shape[1], "fps:", fps)
```

(Optional) quick visualize:

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(9,4))
plt.imshow(20*np.log10(energy + 1e-6), aspect='auto', origin='lower',
           extent=[0, energy.shape[1]/fps, centers[0], centers[-1]])
plt.yscale('log')
plt.ylabel("Frequency (Hz)"); plt.xlabel("Time (s)")
plt.title("Adaptive Constant-Q Energy (dB)")
plt.colorbar(label="dB")
plt.show()
```

---

## What you get (and why it’s useful)

* **Constant-Q decomposition** (octave / M-th octave spacing): perceptually sensible bands.
* **Per-band causal envelope** (RMS-like EWMA): stable **energy tracks**.
* **Streaming API** with block states: drop into real-time chains or offline passes.
* **Compact time axis** via `hop_env` downsampling (e.g., one energy frame per ~170 ms with 48k/1024/8).

Use these tracks to:

* drive **adaptive EQ / multiband compression** (per-band gain laws),
* detect **spectral brightness**, **de-essing zones**, **tonal vs noisy regions**,
* feed **visualizations** or **ML front-ends**.

---

## Tuning tips

* **`Q`** higher → narrower bands (better tonal isolation, slower envelopes).
* **`env_tau`** larger → smoother envelopes (less jitter).
* **`bands_per_octave`**: 3 (third-octave), 6, 12 (semi-tone-ish), etc.
* **`block_size` & `hop_env`** trade temporal resolution vs. frame rate.

---

If you want, I can bolt on a **per-band adaptive gain law** (turning this into a full **dynamic constant-Q EQ**) that consumes `energy` and emits processed audio — you already have 90% of the plumbing here.
