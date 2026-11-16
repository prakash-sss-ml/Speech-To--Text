// Browser microphone Float32 PCM -> mono 16 kHz Float32 PCM for faster-whisper.
class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;
    this.position = 0;
    this.samples = [];
  }
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    // Keep graph active but don't play microphone audio through speakers.
    if (output[0]) output[0].fill(0);
    if (!input || !input.length || !input[0].length) return true;
    const n = input[0].length;
    const channels = input.length;
    while (this.position < n) {
      const i = Math.floor(this.position);
      const frac = this.position - i;
      let s = 0;
      for (let c = 0; c < channels; c++) {
        const a = input[c][i];
        const b = input[c][Math.min(i + 1, n - 1)];
        s += a + (b - a) * frac;
      }
      this.samples.push(s / channels);
      this.position += this.step;
    }
    this.position -= n;
    if (this.samples.length >= 2048) {
      this.port.postMessage(new Float32Array(this.samples));
      this.samples = [];
    }
    return true;
  }
}
registerProcessor('pcm-worklet', PCMWorklet);
