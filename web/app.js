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
    case "browser_task":
      onBrowserTask(msg);
      break;
    case "browser_step":
      onBrowserStep(msg);
      break;
    case "media_job":
      onMediaJob(msg);
      break;
    case "media":
      onMedia(msg);
      break;
  }
}

// ---------- Pictures and videos (ComfyUI) ----------

const jobCards = {};  // media job id -> {el, cap, started, label}

function mediaCard(cls) {
  const el = add("bot media " + cls, "");
  const cap = document.createElement("div");
  cap.className = "cap";
  el.appendChild(cap);
  return { el, cap };
}

function setCaption(card, title, detail) {
  card.cap.innerHTML = "";
  const b = document.createElement("b");
  b.textContent = title;
  card.cap.append(b, detail ? " · " + detail : "");
}

function elapsed(card) {
  const s = Math.round((Date.now() - card.started) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function onMediaJob(msg) {
  let card = jobCards[msg.id];
  if (msg.status === "started") {
    card = jobCards[msg.id] = { ...mediaCard("pending"), started: Date.now(), workflow: msg.workflow };
    card.label = msg.kind === "install"
      ? `Adding workflow ${msg.workflow}`
      : `Making ${msg.makes === "video" ? "a video" : "a picture"} with ${msg.workflow}`;
    card.detail = msg.prompt || "";
    if (msg.kind === "install") card.el.classList.add("tool");
  }
  if (!card) return;
  if (msg.status === "running" && msg.text) card.detail = msg.text;
  if (msg.status === "started" || msg.status === "running") {
    setCaption(card, `${card.label}… ${elapsed(card)}`, card.detail);
    return;
  }
  card.el.classList.remove("pending");
  card.done = true;
  if (msg.status === "done") {
    if (msg.kind === "install") setCaption(card, `Workflow ${card.workflow} is ready`, msg.text);
    else if (!card.el.querySelector("img, video, audio, a")) setCaption(card, `${card.label}: done`, msg.text);
    else card.cap.append(` · ${msg.text}`);
  } else {
    card.el.classList.add("failed");
    setCaption(card, `${card.label}: ${msg.status}`, msg.text || "");
  }
  log.scrollTop = log.scrollHeight;
}

setInterval(() => {
  for (const card of Object.values(jobCards)) {
    if (!card.done) setCaption(card, `${card.label}… ${elapsed(card)}`, card.detail);
  }
}, 1000);

function mediaElement(msg) {
  if (msg.kind === "image") {
    const a = document.createElement("a");
    a.href = msg.url; a.target = "_blank";
    const img = document.createElement("img");
    img.src = msg.url; img.alt = msg.prompt || msg.name;
    img.onload = () => { log.scrollTop = log.scrollHeight; };
    a.appendChild(img);
    return a;
  }
  if (msg.kind === "video" || msg.kind === "audio") {
    const m = document.createElement(msg.kind);
    m.src = msg.url; m.controls = true; m.loop = true; m.preload = "metadata";
    if (msg.kind === "video") { m.muted = true; m.autoplay = true; m.playsInline = true; }
    return m;
  }
  const a = document.createElement("a");
  a.href = msg.url; a.target = "_blank"; a.textContent = `📄 ${msg.name}`;
  return a;
}

function onMedia(msg) {
  let card = msg.job != null ? jobCards[msg.job] : null;
  if (card && card.el.querySelector("img, video, audio")) card = null;  // batch: one card per file
  if (!card) {
    card = mediaCard("");
    if (msg.source === "upload") card.el.className = "msg user media";
  }
  card.el.insertBefore(mediaElement(msg), card.cap);
  const what = msg.kind === "workflow" ? "workflow file" : msg.kind;
  setCaption(card, `#${msg.id} ${what}`,
    msg.source === "upload" ? msg.name : [msg.workflow, msg.prompt].filter(Boolean).join(" — "));
  card.done = true;
  card.el.classList.remove("pending");
  log.scrollTop = log.scrollHeight;
}

// ---------- Attachments ----------

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

async function uploadFile(file) {
  try {
    ensureAudioCtx();
    await ensureConnected();
    const name = file.name || "pasted.png";
    if (name.toLowerCase().endsWith(".json") || file.type === "application/json") {
      let workflow;
      try { workflow = JSON.parse(await file.text()); } catch { throw new Error(`${name} is not valid JSON`); }
      ws.send(JSON.stringify({ type: "upload", name, workflow }));
      return;
    }
    let data;
    if (file.type.startsWith("image/") && file.type !== "image/gif") data = await shrinkImage(file);
    else if (file.size > MAX_FILE) throw new Error(`${name} is too large (max 10 MB)`);
    else data = await readDataURL(file);
    ws.send(JSON.stringify({ type: "upload", name, data }));
  } catch (e) {
    add("err", `Upload failed: ${e.message}`);
  }
}

$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  for (const f of $("file").files) await uploadFile(f);
  $("file").value = "";
};
document.addEventListener("paste", async (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (!files.length) return;
  e.preventDefault();
  for (const f of files) await uploadFile(f);
});
document.addEventListener("dragover", (e) => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  e.preventDefault();
  document.body.classList.add("dropping");
});
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("dropping");
});
document.addEventListener("drop", async (e) => {
  document.body.classList.remove("dropping");
  if (!e.dataTransfer.files.length) return;
  e.preventDefault();
  for (const f of e.dataTransfer.files) await uploadFile(f);
});

// ---------- Browser agent panel ----------

function onBrowserTask(msg) {
  $("browser").hidden = false;
  const status = $("bstatus");
  status.className = "badge " + (msg.status === "started" ? "running" : msg.status);
  status.textContent = msg.status === "started" ? "running" : msg.status;
  $("bcancel").hidden = msg.status !== "started";
  if (msg.status === "started") {
    $("btask").textContent = `#${msg.id} ${msg.text}`;
    $("steps").innerHTML = "";
    add("tool", `🌐 task #${msg.id} started: ${msg.text}`);
  } else {
    add("tool", `🌐 task #${msg.id} ${msg.status}${msg.text ? ": " + msg.text : ""}`);
  }
}

function onBrowserStep(msg) {
  const li = document.createElement("li");
  li.textContent = `${msg.action} ${JSON.stringify(msg.args)} → ${msg.result}`;
  $("steps").appendChild(li);
  $("steps").scrollTop = $("steps").scrollHeight;
  if (msg.screenshot) {
    $("bshot").src = "data:image/jpeg;base64," + msg.screenshot;
    $("bshot").hidden = false;
  }
}

$("bcancel").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "cancel_task" }));
};

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
  for (const id in jobCards) delete jobCards[id];
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
