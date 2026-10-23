#!/usr/bin/env python3
"""Detector-independent wingbeat f0 measurement on real recordings.

For every 0.512 s frame (4096 samples at 8 kHz, Hann, 50 % overlap) the
harmonic sum  S(f) = P(f) + 0.7 P(2f) + 0.5 P(3f)  is maximised over
f0 in [250, 900] Hz (parabolic refinement in log power). A frame is kept only
if the fundamental is a clear tone: ≥ 12 dB above the median power of its
±150 Hz neighbourhood, with nothing comparable at f0/2 (sub-harmonic check —
a lower-pitched source such as a fly, bee or voice would light f0 too).

Output: per-file f0 percentiles, split into the first and second half of the
recording in time (fit on one, check on the other), as JSON on stdout.

    python3 -I tools/real/wingbeat_f0.py <wav> [<wav> ...] [--exclude seg.txt ...]
"""
import json
import sys
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, 'rb') as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if sw != 2:
            raise ValueError(f'{path}: only 16-bit PCM supported')
        x = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), sr


def segments(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    try:
                        out.append((float(p[0]), float(p[1])))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def frame_f0s(x, sr, exclude=()):
    nfft = 4096 if sr <= 16000 else 16384
    hop = nfft // 2
    win = np.hanning(nfft)
    bin_hz = sr / nfft
    k_lo, k_hi = int(250 / bin_hz), int(900 / bin_hz)
    nb = int(150 / bin_hz)
    starts = [s for s in range(0, len(x) - nfft, hop)
              if not any(a <= (s + nfft / 2) / sr < b for a, b in exclude)]
    if not starts:
        return []
    spec = np.array([np.abs(np.fft.rfft(x[s:s + nfft] * win)) ** 2 + 1e-20 for s in starts])
    # Whiten every bin by its own median over the recording: stationary lines
    # (mains, fans, the recorder itself — e.g. a 284.5 Hz line present in every
    # frame of the CDC culture-room recordings) drop out; a passing insect,
    # present only part of the time, stands out.
    floor = np.median(spec, axis=0)
    res = []
    for start, P in zip(starts, spec / floor):
        t = (start + nfft / 2) / sr
        ks = np.arange(k_lo, k_hi + 1)
        mx = lambda k: np.maximum(np.maximum(P[np.clip(k - 1, 0, len(P) - 1)], P[np.clip(k, 0, len(P) - 1)]),
                                  P[np.clip(k + 1, 0, len(P) - 1)])
        S = mx(ks) + 0.7 * mx(2 * ks) + 0.5 * mx(3 * ks)
        k = int(ks[np.argmax(S)])
        local = np.median(P[max(1, k - nb):k + nb + 1])
        if 10 * np.log10(P[k] / local) < 12 or P[k] < 16:
            continue                                     # not a clear, non-stationary tone
        if mx(np.array([k // 2]))[0] > 0.3 * P[k]:
            continue                                     # energy at f0/2: lower-pitched source
        a, b, c = np.log(P[k - 1]), np.log(P[k]), np.log(P[k + 1])
        d = a - 2 * b + c
        off = 0.5 * (a - c) / d if d < 0 else 0.0
        res.append((t, (k + max(-0.5, min(0.5, off))) * bin_hz))
    return res


def summary(vals):
    if not vals:
        return None
    v = np.array(vals)
    q = lambda p: round(float(np.percentile(v, p)), 1)
    return {'n': len(vals), 'p10': q(10), 'p25': q(25), 'median': q(50), 'p75': q(75), 'p90': q(90),
            'mean': round(float(v.mean()), 1), 'sd': round(float(v.std()), 1)}


def main(argv):
    wavs, excl = [], []
    i = 0
    while i < len(argv):
        if argv[i] == '--exclude':
            excl.append(argv[i + 1]); i += 2
        else:
            wavs.append(argv[i]); i += 1
    out = {}
    for j, w in enumerate(wavs):
        x, sr = read_wav(w)
        ex = segments(excl[j]) if j < len(excl) else []
        fr = frame_f0s(x, sr, ex)
        dur = len(x) / sr
        first = [f for t, f in fr if t < dur / 2]
        second = [f for t, f in fr if t >= dur / 2]
        out[w.rsplit('/', 1)[-1]] = {'seconds': round(dur, 1), 'all': summary([f for _, f in fr]),
                                     'firstHalf': summary(first), 'secondHalf': summary(second),
                                     'hist25Hz': np.histogram([f for _, f in fr], bins=np.arange(250, 925, 25))[0].tolist()}
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
