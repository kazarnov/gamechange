// The agent's WebSocket protocol (backend/session.py): mic audio up as PCM16 16 kHz, speech down
// as PCM16 24 kHz behind a 4-byte turn id, everything else as JSON events. Both pages use it:
// the agent page (index.html) and the test console (console.html).

export class VoiceLink {
  constructor({ onEvent = () => {}, onClose = () => {} } = {}) {
    this.onEvent = onEvent;
    this.onClose = onClose;
    this.ws = null;
    this.audioCtx = null;     // playback + capture
    this.micNode = null;
    this.micStream = null;
    this.analyser = null;
    this.micOn = false;
    this.workletLoaded = false;
    this.ttsRate = 24000;
    this.minTurn = 0;         // drop audio from turns older than this
    this.nextPlayTime = 0;
    this.sources = [];
    this.userSpeaking = false;
    this.thinking = false;
    this.bargeIn = () => true;  // false: the mic is muted while the assistant speaks (no headphones)
  }

  get open() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  get playing() {
    return !!this.audioCtx && this.audioCtx.currentTime < this.nextPlayTime;
  }

  /** disconnected | connected | listening | hearing | thinking | speaking */
  get state() {
    if (!this.open) return "disconnected";
    if (this.userSpeaking) return "hearing";
    if (this.playing) return "speaking";
    if (this.thinking) return "thinking";
    return this.micOn ? "listening" : "connected";
  }

  /** Connects to "ws" next to the page, so the page can live under any path (e.g. /assistant/). */
  connect() {
    if (this.open) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(new URL("ws", location.href).href.replace(/^http/, "ws"));
      ws.binaryType = "arraybuffer";
      ws.onopen = () => { this.ws = ws; resolve(); };
      ws.onerror = () => reject(new Error("Can't reach the assistant"));
      ws.onclose = () => {
        const was = this.ws === ws;
        this.stopMic();
        this.ws = null;
        this.userSpeaking = this.thinking = false;
        if (was) this.onClose();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") this._event(JSON.parse(ev.data));
        else this._audio(ev.data);
      };
    });
  }

  send(msg) {
    if (this.open) this.ws.send(JSON.stringify(msg));
  }

  /** Text typed instead of spoken. */
  async say(text) {
    this.ensureAudio();
    await this.connect();
    this.send({ type: "text", text });
  }

  interrupt() {
    this.send({ type: "interrupt" });
    this.stopPlayback();
  }

  _event(msg) {
    switch (msg.type) {
      case "ready": this.ttsRate = msg.tts_sample_rate; break;
      case "vad": this.userSpeaking = msg.speaking; break;
      case "transcript": if (msg.final && msg.text) this.thinking = true; break;
      case "assistant_delta": case "assistant_done": case "error": this.thinking = false; break;
      case "interrupt": this.minTurn = msg.turn; this.stopPlayback(); this.thinking = false; break;
    }
    this.onEvent(msg);
  }

  // ---------- Playback ----------

  ensureAudio() {
    if (!this.audioCtx) this.audioCtx = new AudioContext();
    if (this.audioCtx.state === "suspended") this.audioCtx.resume();
  }

  _audio(buf) {
    const turn = new DataView(buf).getUint32(0, true);
    if (turn < this.minTurn || !this.audioCtx) return;
    const pcm = new Int16Array(buf, 4);
    const audio = this.audioCtx.createBuffer(1, pcm.length, this.ttsRate);
    const ch = audio.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

    const src = this.audioCtx.createBufferSource();
    src.buffer = audio;
    src.connect(this.audioCtx.destination);
    const start = Math.max(this.audioCtx.currentTime + 0.03, this.nextPlayTime);
    src.start(start);
    this.nextPlayTime = start + audio.duration;
    this.sources.push(src);
    src.onended = () => { this.sources = this.sources.filter((s) => s !== src); };
  }

  stopPlayback() {
    for (const s of this.sources) { try { s.stop(); } catch {} }
    this.sources = [];
    this.nextPlayTime = 0;
  }

  // ---------- Microphone ----------

  async startMic() {
    this.ensureAudio();
    await this.connect();
    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (!this.workletLoaded) {
      await this.audioCtx.audioWorklet.addModule(new URL("mic-worklet.js", import.meta.url));
      this.workletLoaded = true;
    }
    const source = this.audioCtx.createMediaStreamSource(this.micStream);
    this.micNode = new AudioWorkletNode(this.audioCtx, "mic-processor");
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 512;
    source.connect(this.analyser);
    source.connect(this.micNode);
    this.micNode.port.onmessage = (ev) => {
      if (!this.open) return;
      // Without barge-in, don't let the assistant's own voice reach the server
      if (!this.bargeIn() && this.playing) return;
      this.ws.send(ev.data);
    };
    this.micOn = true;
  }

  stopMic() {
    this.micOn = false;
    if (this.micNode) { this.micNode.port.onmessage = null; this.micNode.disconnect(); this.micNode = null; }
    if (this.micStream) { this.micStream.getTracks().forEach((t) => t.stop()); this.micStream = null; }
    this.analyser = null;
    this.userSpeaking = false;
  }

  /** Mic loudness, 0 to 1. */
  level() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    return Math.min(1, Math.sqrt(sum / data.length) * 6);
  }
}

/** A picture, video or audio file from the conversation, or a link to it. */
export function mediaElement({ kind, url, name = "", alt = "" }) {
  if (kind === "image") {
    const img = document.createElement("img");
    img.src = url;
    img.alt = alt || name;
    img.loading = "lazy";
    return img;
  }
  if (kind === "video" || kind === "audio") {
    const m = document.createElement(kind);
    m.src = url; m.controls = true; m.loop = true; m.preload = "metadata";
    if (kind === "video") { m.muted = true; m.autoplay = true; m.playsInline = true; }
    return m;
  }
  const a = document.createElement("a");
  a.href = url; a.target = "_blank"; a.textContent = `📄 ${name}`;
  return a;
}

const MAX_FILE = 10 * 1024 * 1024;  // the WebSocket takes up to 16 MB per message (base64 adds a third)

function readDataURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

async function shrinkImage(file, max = 2048) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, max / Math.max(bmp.width, bmp.height));
  if (scale === 1 && file.size <= MAX_FILE) { bmp.close(); return readDataURL(file); }
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  let url = canvas.toDataURL(file.type === "image/png" ? "image/png" : "image/jpeg", 0.92);
  if (url.length > MAX_FILE * 1.3) url = canvas.toDataURL("image/jpeg", 0.9);
  return url;
}

/** Sends an attached file: a picture, video or audio file, or a ComfyUI workflow (.json). */
export async function uploadFile(link, file) {
  link.ensureAudio();
  await link.connect();
  const name = file.name || "pasted.png";
  if (name.toLowerCase().endsWith(".json") || file.type === "application/json") {
    let workflow;
    try { workflow = JSON.parse(await file.text()); } catch { throw new Error(`${name} is not valid JSON`); }
    link.send({ type: "upload", name, workflow });
    return;
  }
  let data;
  if (file.type.startsWith("image/") && file.type !== "image/gif") data = await shrinkImage(file);
  else if (file.size > MAX_FILE) throw new Error(`${name} is too large (max 10 MB)`);
  else data = await readDataURL(file);
  link.send({ type: "upload", name, data });
}

/** Files pasted or dropped anywhere on the page. */
export function acceptFiles(onFiles, dropClass = "dropping") {
  document.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])];
    if (!files.length) return;
    e.preventDefault();
    onFiles(files);
  });
  document.addEventListener("dragover", (e) => {
    if (![...e.dataTransfer.types].includes("Files")) return;
    e.preventDefault();
    document.body.classList.add(dropClass);
  });
  document.addEventListener("dragleave", (e) => {
    if (!e.relatedTarget) document.body.classList.remove(dropClass);
  });
  document.addEventListener("drop", (e) => {
    document.body.classList.remove(dropClass);
    if (!e.dataTransfer.files.length) return;
    e.preventDefault();
    onFiles([...e.dataTransfer.files]);
  });
}
