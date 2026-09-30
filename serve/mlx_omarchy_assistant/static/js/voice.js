// Voice: mic recording + TTS playback queue.
//
// Recording uses getUserMedia + an AudioWorklet to capture mono float32 at
// the device sample rate. On stop the buffer is resampled to 16 kHz mono
// (Parakeet's input contract), encoded as 16-bit PCM WAV and POSTed to
// /api/transcribe. The recording hard-caps at 30 seconds, warns at 25, and
// never auto-starts; tab close stops the mic.
//
// TTS serialises /api/speak requests to a queue of two, plays each WAV
// through a Web Audio AudioBuffer, caps cumulative decoded playback at 10s
// per turn and offers "Continue reading" if the queue was truncated.

import { streamSpeak, transcribe } from "./api.js";
import { asString, asNumber } from "./util.js";

const SAMPLE_RATE = 16000;
const MAX_DURATION = 30;
const WARN_DURATION = 25;
const QUEUE_LIMIT = 2;
const PLAYBACK_CAP = 10;
const RETRY_INTERVAL_MS = 2000;
const CHUNK_MARGIN_S = 0.03;
const MAX_SPEAK_RETRIES = 10;
const WARN_LEVEL = 0.15;
const MIN_RMS_LEVEL = 0.005;

// ---------------------------------------------------------------------------
// WAV encoder / decoder
// ---------------------------------------------------------------------------

function encodeWav(float32, sampleRate) {
  // 16-bit PCM mono WAV (RIFF). Each float is clamped to [-1, 1] and scaled.
  const numSamples = float32.length;
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeAscii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);             // PCM
  view.setUint16(22, 1, true);             // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    let s = float32[i];
    if (!Number.isFinite(s)) s = 0;
    if (s > 1) s = 1; else if (s < -1) s = -1;
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

export class Recorder {
  constructor({ onMeter, onTick, onWarn, onStop, onDeviceLost } = {}) {
    this.state = "idle";          // idle | recording | stopping
    this.startedAt = 0;
    this._samples = [];
    this._ctx = null;
    this._stream = null;
    this._node = null;
    this._source = null;
    this._tickHandle = null;
    this._onMeter = onMeter;
    this._onTick = onTick;
    this._onWarn = onWarn;
    this._onStop = onStop;
    this._onDeviceLost = onDeviceLost;
    this._warned = false;
    this._maxSamples = 0;
    this._stopReason = null;       // 'user' | 'limit' | 'device' | null
  }

  async start() {
    if (this.state !== "idle") return false;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw new Error("Microphone API unavailable in this browser");
    if (!window.AudioWorkletNode)
      throw new Error("AudioWorklet is not available in this browser");
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, noiseSuppression: true }, video: false,
      });
    } catch (err) {
      const name = err && err.name;
      if (name === "NotAllowedError" || name === "SecurityError")
        throw new Error("Microphone permission denied. Allow microphone access for this page, then try again.");
      if (name === "NotFoundError" || name === "OverconstrainedError")
        throw new Error("No microphone found. Connect a microphone, then try again.");
      if (name === "NotReadableError")
        throw new Error("The microphone is in use by another application.");
      throw err;
    }
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule(new URL("./worklet/capture-worklet.js", import.meta.url));
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "mlx-capture");
    node.port.onmessage = (event) => {
      if (this.state !== "recording") return;
      const frame = event.data;
      this._samples.push(frame);
      this._maxSamples += frame.length;
      let sum = 0;
      for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
      const rms = Math.sqrt(sum / Math.max(1, frame.length));
      if (this._onMeter) this._onMeter(rms);
      const inputRate = ctx.sampleRate;
      const elapsed = this._maxSamples / inputRate;
      if (!this._warned && elapsed >= WARN_DURATION) {
        this._warned = true;
        if (this._onWarn) this._onWarn(elapsed);
      }
      if (elapsed >= MAX_DURATION) {
        // Auto-stop on the 30 s cap. The stop reason records the
        // difference from a user-initiated stop so the caller can
        // announce it correctly.
        this._stopReason = "limit";
        this.stop().catch(() => {});
      }
    };
    source.connect(node);
    // Track loss mid-recording: yank the cable, OS revokes permission,
    // or the user picks a different default device. We surface it as a
    // named event so the caller can return to Idle with the typed draft
    // intact instead of hanging.
    for (const track of stream.getAudioTracks()) {
      track.addEventListener("ended", () => {
        if (this.state !== "recording") return;
        this._stopReason = "device";
        this.stop().catch(() => {});
      });
    }
    this._ctx = ctx;
    this._source = source;
    this._node = node;
    this._stream = stream;
    this._samples = [];
    this._maxSamples = 0;
    this._warned = false;
    this._stopReason = null;
    this.state = "recording";
    this.startedAt = ctx.currentTime;
    this._tickHandle = window.setInterval(() => {
      if (this.state !== "recording") return;
      const inputRate = ctx.sampleRate;
      const elapsed = this._maxSamples / inputRate;
      if (this._onTick) this._onTick(elapsed);
    }, 100);
    return true;
  }

  async stop() {
    if (this.state !== "recording") return null;
    // Frames keep arriving while cleanup awaits; "stopping" makes this stop
    // the only one and drops those late frames.
    this.state = "stopping";
    if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
    const reason = this._stopReason || "user";
    const inputRate = this._ctx ? this._ctx.sampleRate : 48000;
    // The recognizer refuses anything over 30 s; the frame that crossed the
    // cap must not push the upload past it.
    const totalLength = Math.min(this._maxSamples, MAX_DURATION * inputRate);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const s of this._samples) {
      if (offset >= totalLength) break;
      const part = s.subarray(0, totalLength - offset);
      merged.set(part, offset); offset += part.length;
    }
    this._samples = [];
    const resampled = resampleTo16k(merged, inputRate);
    const blob = encodeWav(resampled, SAMPLE_RATE);
    await this._cleanup();
    let peakRms = 0;
    for (let i = 0; i < merged.length; i += Math.max(1, Math.floor(merged.length / 200))) {
      const v = Math.abs(merged[i] || 0);
      if (v > peakRms) peakRms = v;
    }
    this.state = "idle";
    if (reason === "device") {
      // Mid-recording device loss: there is no usable WAV to ship, but
      // the caller must still learn about it. The onDeviceLost hook is
      // the named surface; onStop is intentionally skipped so the UI
      // does not kick off a transcribe roundtrip on a silent blob.
      this._stopReason = null;
      if (this._onDeviceLost) this._onDeviceLost({ reason });
      return null;
    }
    this._stopReason = null;
    if (this._onStop) this._onStop({ blob, duration: merged.length / inputRate, peakRms, reason });
    return { blob, duration: merged.length / inputRate };
  }

  async cancel() {
    if (this.state !== "recording") return;
    this.state = "stopping";
    if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
    this._samples = [];
    this._stopReason = null;
    await this._cleanup();
    this.state = "idle";
  }

  async _cleanup() {
    if (this._source) { try { this._source.disconnect(); } catch {} }
    if (this._node) { try { this._node.disconnect(); } catch {} }
    if (this._stream) for (const t of this._stream.getTracks()) t.stop();
    if (this._ctx) { try { await this._ctx.close(); } catch {} }
    this._ctx = null; this._node = null; this._source = null; this._stream = null;
  }
}

function resampleTo16k(input, inputRate) {
  if (inputRate === SAMPLE_RATE) return input;
  const outLength = Math.max(1, Math.round(input.length * SAMPLE_RATE / inputRate));
  const output = new Float32Array(outLength);
  const ratio = inputRate / SAMPLE_RATE;
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const left = Math.floor(srcIndex);
    const right = Math.min(input.length - 1, left + 1);
    const t = srcIndex - left;
    output[i] = input[left] * (1 - t) + input[right] * t;
  }
  return output;
}

// ---------------------------------------------------------------------------
// TTS playback queue — streams real PCM16LE chunks from POST /api/speak (SSE)
// into the Web Audio clock. No waveform is drawn; playback is the signal.
// ---------------------------------------------------------------------------

function pcm16leToFloat32(base64) {
  const bin = atob(base64);
  const view = new DataView(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
  const count = bin.length >> 1;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const s = view.getInt16(i * 2, true);
    out[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
  }
  return out;
}

export { pcm16leToFloat32 };

// One-shot preview: decode a single PCM16LE base64 payload and play it
// through a fresh AudioContext. The shared SpeakQueue is for streamed
// sentence playback; preview is a single bounded render and reuses none
// of that machinery.
export async function playPreview({ sample_rate, encoding, data }) {
  if (encoding !== "pcm16le") {
    throw new Error(`unsupported preview encoding: ${encoding}`);
  }
  const rate = Number(sample_rate) || 24000;
  const ctx = new AudioContext();
  try {
    const samples = pcm16leToFloat32(data);
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.getChannelData(0).set(samples);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    await new Promise((resolve, reject) => {
      src.onended = () => resolve();
      src.onerror = (err) => reject(err);
      src.start();
    });
  } finally {
    try { await ctx.close(); } catch { /* ignore */ }
  }
}

export class SpeakQueue {
  constructor() {
    this._ctx = null;
    this._sources = [];      // scheduled AudioBufferSourceNodes (live + pending)
    this._queue = [];        // sentences waiting to start (limit QUEUE_LIMIT)
    this._pending = [];      // busy-parked sentences on a bounded retry timer
    this._inFlight = 0;
    this._truncated = false;
    this._totalDecoded = 0;
    this._turnId = null;
    this._abort = null;
    this._epoch = 0;
    this._retryTimer = null;
    this._completedSeqs = new Set();
    this._onTruncate = null;
    this._onAudioDone = null;
    this._onError = null;
    this._onSpeakingChange = null;
    this._stopped = false;
  }

  attachHooks({ onTruncate, onAudioDone, onError, onSpeakingChange } = {}) {
    if (onTruncate) this._onTruncate = onTruncate;
    if (onAudioDone) this._onAudioDone = onAudioDone;
    if (onError) this._onError = onError;
    if (onSpeakingChange) this._onSpeakingChange = onSpeakingChange;
  }

  async _ctxLazy() {
    if (!this._ctx) this._ctx = new AudioContext();
    return this._ctx;
  }

  completedSequences() {
    return new Set(this._completedSeqs);
  }

  enqueue(turnId, sentenceSequence, text) {
    if (this._stopped) return;
    if (turnId !== this._turnId) {
      this._turnId = turnId;
      this._completedSeqs = new Set();
    }
    if (this._totalDecoded >= PLAYBACK_CAP) {
      this._truncated = true;
      if (this._onTruncate) this._onTruncate();
      return;
    }
    if (this._inFlight >= QUEUE_LIMIT) {
      this._queue.push({ turnId, sentenceSequence, text });
      return;
    }
    this._inFlight += 1;
    this._speak(turnId, sentenceSequence, text).catch((err) => {
      if (this._onError) this._onError(err);
    });
  }

  _setSpeaking(active) {
    if (this._onSpeakingChange) this._onSpeakingChange(active);
  }

  _kickRetry() {
    if (this._retryTimer || this._stopped) return;
    if (this._pending.length === 0) return;
    this._retryTimer = window.setInterval(() => this._retryPending(), RETRY_INTERVAL_MS);
  }

  _retryPending() {
    if (this._stopped || this._pending.length === 0) {
      this._clearRetry();
      return;
    }
    if (this._inFlight >= QUEUE_LIMIT) return;
    const item = this._pending.shift();
    item.attempts += 1;
    if (item.attempts > MAX_SPEAK_RETRIES) {
      if (this._onError) {
        this._onError(new Error("Speech worker stayed busy; sentence dropped from the read queue"));
      }
      return;
    }
    this._inFlight += 1;
    this._speak(item.turnId, item.sentenceSequence, item.text).catch((err) => {
      if (this._onError) this._onError(err);
    });
  }

  _clearRetry() {
    if (this._retryTimer) {
      window.clearInterval(this._retryTimer);
      this._retryTimer = null;
    }
  }

  async _speak(turnId, sentenceSequence, text) {
    const epoch = this._epoch;
    this._setSpeaking(true);
    const abort = new AbortController();
    this._abort = abort;
    const ctx = await this._ctxLazy();
    let nextTime = ctx.currentTime + CHUNK_MARGIN_S;
    let streamClosed = false;
    let pendingChunks = 0;
    let streamError = null;
    const self = this;

    const tryFinish = () => {
      if (epoch !== self._epoch) return;
      if (!streamClosed || pendingChunks > 0) return;
      self._inFlight -= 1;
      const next = self._queue.shift();
      if (next && !self._stopped) {
        self._inFlight += 1;
        self._speak(next.turnId, next.sentenceSequence, next.text).catch((err) => {
          if (self._onError) self._onError(err);
        });
        return;
      }
      if (self._pending.length > 0) self._kickRetry();
      if (self._inFlight <= 0) {
        self._setSpeaking(false);
        if (self._onAudioDone) self._onAudioDone({ truncated: self._truncated });
      }
    };

    try {
      await streamSpeak(
        { conversation_id: this._conversationId, turn_id: turnId,
          text, sentence_sequence: sentenceSequence },
        { signal: abort.signal,
          onEvent: ({ event, data }) => {
            if (epoch !== self._epoch) return;
            if (event === "audio") {
              const b64 = asString(data && data.data);
              if (b64 === undefined) return;
              const rate = asNumber(data && data.sample_rate) || 16000;
              const samples = pcm16leToFloat32(b64);
              const duration = samples.length / rate;
              if (self._totalDecoded + duration > PLAYBACK_CAP) {
                self._truncated = true;
                abort.abort();
                if (self._onTruncate) self._onTruncate();
                return;
              }
              const buffer = ctx.createBuffer(1, samples.length, rate);
              buffer.getChannelData(0).set(samples);
              const src = ctx.createBufferSource();
              src.buffer = buffer;
              src.connect(ctx.destination);
              const at = Math.max(ctx.currentTime + CHUNK_MARGIN_S, nextTime);
              nextTime = at + duration;
              src.start(at);
              self._totalDecoded += duration;
              pendingChunks += 1;
              self._sources.push(src);
              src.onended = () => {
                self._sources = self._sources.filter((s) => s !== src);
                self._completedSeqs.add(sentenceSequence);
                pendingChunks -= 1;
                tryFinish();
              };
            } else if (event === "error") {
              streamError = new Error(asString(data && data.message) || "Speech failed");
            }
          } });
    } catch (err) {
      if (err && err.code === "busy") {
        // GPU busy before the stream started: park on the bounded retry
        // timer. Speak requests only — never a model turn.
        this._pending.push({ turnId, sentenceSequence, text, attempts: 0 });
        this._kickRetry();
        this._inFlight -= 1;
        if (this._inFlight <= 0) this._setSpeaking(false);
        return;
      }
      if (!(err && err.name === "AbortError") && this._onError) {
        this._onError(err);
      }
      this._inFlight -= 1;
      if (this._inFlight <= 0) this._setSpeaking(false);
      return;
    }
    if (streamError && epoch === this._epoch && this._onError) {
      this._onError(streamError);
    }
    streamClosed = true;
    tryFinish();
  }

  setConversationId(id) { this._conversationId = id; }

  async stop() {
    this._epoch += 1;
    this._stopped = true;
    this._clearRetry();
    this._pending = [];
    if (this._abort) { try { this._abort.abort(); } catch {} }
    for (const src of this._sources) {
      try { src.stop(); } catch {}
      try { src.disconnect(); } catch {}
    }
    this._sources = [];
    this._queue = [];
    this._inFlight = 0;
    this._setSpeaking(false);
  }

  reset() {
    this._stopped = false;
    this._truncated = false;
    this._totalDecoded = 0;
    this._completedSeqs = new Set();
  }

  resumeFromTruncation(remainingSentences) {
    // Same per-turn decoded budget applies; the user may need to invoke
    // this more than once for a long reply.
    this.reset();
    for (let i = 0; i < remainingSentences.length; i++) {
      const [seq, text] = remainingSentences[i];
      this.enqueue(this._turnId, seq, text);
    }
  }
}

export const RECORDING_CONSTRAINTS = { MAX_DURATION, WARN_DURATION, WARN_LEVEL, MIN_RMS_LEVEL };

export { transcribe };
