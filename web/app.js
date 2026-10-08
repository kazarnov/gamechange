const $ = (id) => document.getElementById(id);
const log = $("log");

let ws = null;
let audioCtx = null;      // playback + capture context
let micNode = null, micStream = null, analyser = null;
let micOn = false;
let workletLoaded = false;
let ttsRate = 24000;
let minTurn = 0;          // drop audio from turns older than this
let nextPlayTime = 0;
let sources = [];
let partialEl = null;     // live (non-final) user transcript bubble
let botEl = null, botTurn = -1;
let userSpeaking = false, thinking = false;

// ---------- UI helpers ----------

function add(cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function playing() {
  return audioCtx && audioCtx.currentTime < nextPlayTime;
}

function updateState() {
  const dot = $("dot"), state = $("state");
  let s = "disconnected";
  if (ws && ws.readyState === WebSocket.OPEN) {
    s = userSpeaking ? "hearing" : playing() ? "speaking" : thinking ? "thinking" : micOn ? "listening" : "connected";
  }
  dot.className = "dot " + s;
  state.textContent = s;
}
setInterval(updateState, 100);

// ---------- WebSocket ----------

function connect() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => resolve();
    ws.onerror = () => reject(new Error("WebSocket error"));
    ws.onclose = () => { add("sys", "disconnected"); stopMic(); ws = null; };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") onEvent(JSON.parse(ev.data));
      else onAudio(ev.data);
    };
  });
}

async function ensureConnected() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  await connect();
}

function onEvent(msg) {
  switch (msg.type) {
    case "ready":
      ttsRate = msg.tts_sample_rate;
      add("sys", `connected · ASR ${msg.asr_mode}`);
      break;
    case "vad":
      userSpeaking = msg.speaking;
      break;
    case "transcript":
      if (!msg.final) {
        if (!partialEl) partialEl = add("user partial", "");
        partialEl.textContent = msg.text || "…";
      } else {
        if (partialEl) partialEl.remove();
        partialEl = null;
        if (msg.text) { add("user", msg.text); thinking = true; }
      }
      log.scrollTop = log.scrollHeight;
      break;
    case "assistant_delta":
      thinking = false;
      if (!botEl || botTurn !== msg.turn) { botEl = add("bot", ""); botTurn = msg.turn; }
      botEl.textContent += msg.text;
      log.scrollTop = log.scrollHeight;
      break;
    case "assistant_done":
      thinking = false;
      botEl = null;
      break;
    case "tool_call":
      add("tool", `→ ${msg.name}(${JSON.stringify(msg.arguments)})`);
      botEl = null;
      break;
    case "tool_result":
      add("tool", `← ${msg.result}`);
      break;
    case "interrupt":
      minTurn = msg.turn;
      stopPlayback();
      if (botEl && msg.was_speaking) botEl.classList.add("cut");
      botEl = null;
      thinking = false;
      break;
    case "error":
      thinking = false;
      add("err", msg.message);
      break;
  }
}

// ---------- Playback ----------

function ensureAudioCtx() {
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function onAudio(buf) {
  const turn = new DataView(buf).getUint32(0, true);
  if (turn < minTurn || !audioCtx) return;
  const pcm = new Int16Array(buf, 4);
  const audio = audioCtx.createBuffer(1, pcm.length, ttsRate);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

  const src = audioCtx.createBufferSource();
  src.buffer = audio;
  src.connect(audioCtx.destination);
  const start = Math.max(audioCtx.currentTime + 0.03, nextPlayTime);
  src.start(start);
  nextPlayTime = start + audio.duration;
  sources.push(src);
  src.onended = () => { sources = sources.filter((s) => s !== src); };
}

function stopPlayback() {
  for (const s of sources) { try { s.stop(); } catch {} }
  sources = [];
  nextPlayTime = 0;
}

// ---------- Microphone ----------

async function startMic() {
  ensureAudioCtx();
  await ensureConnected();
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  if (!workletLoaded) { await audioCtx.audioWorklet.addModule("mic-worklet.js"); workletLoaded = true; }
  const source = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, "mic-processor");
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);
  source.connect(micNode);
  micNode.port.onmessage = (ev) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Without barge-in, don't let the assistant's own voice reach the server
    if (!$("bargein").checked && playing()) return;
    ws.send(ev.data);
  };
  micOn = true;
  $("mic").textContent = "Stop talking";
  $("mic").classList.add("on");
  drawMeter();
}

function stopMic() {
  micOn = false;
  if (micNode) { micNode.port.onmessage = null; micNode.disconnect(); micNode = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  analyser = null;
  userSpeaking = false;
  $("mic").textContent = "Start talking";
  $("mic").classList.remove("on");
  $("meter").firstElementChild.style.width = "0";
}

function drawMeter() {
  if (!analyser) return;
  const data = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(data);
  let sum = 0;
  for (const v of data) sum += v * v;
  const level = Math.min(1, Math.sqrt(sum / data.length) * 6);
  $("meter").firstElementChild.style.width = `${level * 100}%`;
  requestAnimationFrame(drawMeter);
}

// ---------- Controls ----------

$("mic").onclick = async () => {
  try {
    if (micOn) stopMic();
    else await startMic();
  } catch (e) {
    add("err", `Microphone error: ${e.message}`);
    stopMic();
  }
};

$("stop").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "interrupt" }));
  stopPlayback();
};

$("reset").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
  stopPlayback();
  log.innerHTML = "";
  add("sys", "conversation reset");
};

$("form").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  try {
    ensureAudioCtx();
    await ensureConnected();
    ws.send(JSON.stringify({ type: "text", text }));
    $("text").value = "";
  } catch (err) {
    add("err", err.message);
  }
};
