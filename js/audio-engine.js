/* MosquitoNet — microphone capture (Web Audio).
 *
 * Browser-only. Owns the MediaStream + AudioContext and exposes the frames the
 * classifier and the spectrogram consume. See js/detector.js for the contract.
 */
'use strict';
class MosquitoAudioEngine {
  constructor() {
    this.ctx      = null;
    this.analyser = null;
    this.source   = null;
    this.stream   = null;
    this.running  = false;

    this.FFT_SIZE  = 8192;   // 8192 → 5.4 Hz/bin at 44.1kHz (16384 above 64 kHz; set in init)
    this.sampleRate = 44100; // the ACTUAL context rate after init()
    this.fftSize    = 8192;
    this.hopSamples = 4096;
    this.SMOOTHING = 0.88;   // High smoothing reduces noise for display (classifier uses raw)

    this.freqBuffer = null;
    this.timeBuffer = null;

    // Second analyser with NO smoothing — used exclusively by classifier
    this.rawAnalyser  = null;
    this.rawFreqBuf   = null;
  }

  async init() {
    // iOS Safari: omit sampleRate from getUserMedia constraints
    const constraints = {
      audio: {
        echoCancellation:  false,
        noiseSuppression:  false,
        autoGainControl:   false,
      }
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);

    // iOS requires AudioContext created during/after a user gesture
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) throw new Error('Web Audio API not supported on this device');
    // Prefer 44.1 kHz, but never require it: some browsers ignore or reject the
    // option (old Safari), and Firefox throws when a MediaStreamSource is
    // connected to a context whose rate differs from the microphone's. Fall back
    // to the device's native rate — every detector parameter is derived from
    // the ACTUAL sampleRate / fftSize, so 48 kHz (iOS hardware) is fine.
    let source = null;
    try {
      this.ctx = new AudioCtx({ sampleRate: 44100 });
      source = this.ctx.createMediaStreamSource(this.stream);
    } catch (e) {
      try { if (this.ctx) this.ctx.close(); } catch (_) { /* ignore */ }
      this.ctx = new AudioCtx();
      source = this.ctx.createMediaStreamSource(this.stream);
    }
    // Resume in case browser suspended it (iOS does this)
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this.sampleRate = this.ctx.sampleRate;
    // Keep the analysis window ≈ 186 ms (5.4-5.9 Hz bins) at any rate.
    this.FFT_SIZE = this.sampleRate > 64000 ? 16384 : 8192;
    this.fftSize = this.FFT_SIZE;
    this.hopSamples = this.FFT_SIZE / 2;     // analyse once per this many new samples

    // ── Display analyser (smoothed — for spectrogram) ──
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize                = this.FFT_SIZE;
    this.analyser.smoothingTimeConstant  = this.SMOOTHING;
    this.analyser.minDecibels            = -100;
    this.analyser.maxDecibels            = -10;

    // ── Raw analyser (no smoothing — for classifier) ──
    this.rawAnalyser = this.ctx.createAnalyser();
    this.rawAnalyser.fftSize               = this.FFT_SIZE;
    this.rawAnalyser.smoothingTimeConstant = 0;  // Zero smoothing — classifier needs raw unsmoothed frames
    this.rawAnalyser.minDecibels           = -100;
    this.rawAnalyser.maxDecibels           = -10;

    this.source = source;
    this.source.connect(this.analyser);
    this.source.connect(this.rawAnalyser);
    // NOT connected to destination — no audio output ever

    const bufLen = this.analyser.frequencyBinCount; // FFT_SIZE / 2
    this.freqBuffer    = new Float32Array(bufLen);
    this.rawFreqBuf    = new Float32Array(bufLen);
    this.timeBuffer    = new Float32Array(this.FFT_SIZE);
    this.rawTimeBuffer = new Float32Array(this.FFT_SIZE);

    // Bin resolution: sampleRate / FFT_SIZE ≈ 5.4 Hz/bin
    this.binHz = this.ctx.sampleRate / this.FFT_SIZE;

    this.running = true;
    return true;
  }

  getFrequencyData()    { if (this.analyser)    { this.analyser.getFloatFrequencyData(this.freqBuffer);       return this.freqBuffer; }    return null; }
  getRawFrequencyData() { if (this.rawAnalyser) { this.rawAnalyser.getFloatFrequencyData(this.rawFreqBuf);    return this.rawFreqBuf; }    return null; }
  getTimeDomainData()   { if (this.rawAnalyser) { this.rawAnalyser.getFloatTimeDomainData(this.rawTimeBuffer); return this.rawTimeBuffer; } return null; }

  freqToBin(hz)  { return Math.min(Math.round(hz / this.binHz), this.freqBuffer ? this.freqBuffer.length - 1 : this.FFT_SIZE / 2 - 1); }
  binToHz(bin)   { return bin * this.binHz; }

  stop() {
    if (this.stream)  this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx)     this.ctx.close();
    this.running = false;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { MosquitoAudioEngine };
