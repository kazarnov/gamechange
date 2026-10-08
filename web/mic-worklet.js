// Captures mic audio, downsamples to 16 kHz mono PCM16 and posts ~40 ms chunks.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pending = [];   // input samples not yet consumed
    this.out = new Int16Array(640);
    this.outLen = 0;
    this.frac = 0;       // fractional read position carried between blocks
  }

  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this.pending.push(ch[i]);

    // Box-filter downsample: average each `ratio`-wide window of input samples
    let pos = this.frac;
    while (pos + this.ratio <= this.pending.length) {
      const start = Math.floor(pos), end = Math.floor(pos + this.ratio);
      let sum = 0;
      for (let j = start; j < end; j++) sum += this.pending[j];
      const s = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
      this.out[this.outLen++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.outLen === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.outLen = 0;
      }
      pos += this.ratio;
    }
    const consumed = Math.floor(pos);
    this.pending = this.pending.slice(consumed);
    this.frac = pos - consumed;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
