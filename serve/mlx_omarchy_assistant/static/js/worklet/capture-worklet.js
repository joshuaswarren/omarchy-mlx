// AudioWorklet capture processor. Posts raw Float32 mono frames to the main
// thread; the main thread accumulates them, builds the WAV at 16 kHz mono
// PCM16 once recording stops, and never re-uploads raw audio.

class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (channel && channel.length) {
      // Copy to a fresh Float32Array because the worklet buffer is reused.
      this.port.postMessage(new Float32Array(channel));
    }
    return true;
  }
}

registerProcessor("mlx-capture", CaptureProcessor);
