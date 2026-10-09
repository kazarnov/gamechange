// The test console: every event of the protocol shown as it comes, tool calls included.
import { VoiceLink, acceptFiles, mediaElement as fileElement, uploadFile } from "./voice.js";

const $ = (id) => document.getElementById(id);
const log = $("log");

const link = new VoiceLink({
  onEvent,
  onClose: () => { add("sys", "disconnected"); micOff(); },
});
link.bargeIn = () => $("bargein").checked;
let partialEl = null;     // live (non-final) user transcript bubble
let botEl = null, botTurn = -1;

// ---------- UI helpers ----------

function add(cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function updateState() {
  $("dot").className = "dot " + link.state;
  $("state").textContent = link.state;
}
setInterval(updateState, 100);

function onEvent(msg) {
  switch (msg.type) {
    case "ready":
      add("sys", `connected · ASR ${msg.asr_mode}`);
      break;
    case "transcript":
      if (!msg.final) {
        if (!partialEl) partialEl = add("user partial", "");
        partialEl.textContent = msg.text || "…";
      } else {
        if (partialEl) partialEl.remove();
        partialEl = null;
        if (msg.text) add("user", msg.text);
      }
      log.scrollTop = log.scrollHeight;
      break;
    case "assistant_delta":
      if (!botEl || botTurn !== msg.turn) { botEl = add("bot", ""); botTurn = msg.turn; }
      botEl.textContent += msg.text;
      log.scrollTop = log.scrollHeight;
      break;
    case "assistant_done":
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
      if (botEl && msg.was_speaking) botEl.classList.add("cut");
      botEl = null;
      break;
    case "error":
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
    case "draft":
      onDraft(msg);
      break;
    case "flowai":
      add(msg.error ? "err" : "sys", msg.error ? `FlowAI: ${msg.error}`
        : `FlowAI · ${msg.user.name} · ${msg.accounts.map((a) => a.label).join(", ") || "no accounts"}`);
      break;
    case "saved":
      add("tool", `💾 draft ${msg.draft} v${msg.version} → FlowAI post ${msg.post} (${msg.status}${msg.when ? ", " + msg.when : ""})`);
      break;
    case "approval":
      onApproval(msg);
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
  const el = fileElement({ kind: msg.kind, url: msg.url, name: msg.name, alt: msg.prompt });
  if (msg.kind !== "image") return el;
  el.onload = () => { log.scrollTop = log.scrollHeight; };
  const a = document.createElement("a");
  a.href = msg.url; a.target = "_blank";
  a.appendChild(el);
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

// ---------- Approvals (scheduling) ----------

const approvalCards = {};

function onApproval(msg) {
  const el = approvalCards[msg.id] || (approvalCards[msg.id] = add("tool", ""));
  el.textContent = `🗓 ${msg.text}` + (msg.status === "waiting" ? " " : ` · ${msg.status}${msg.detail ? ": " + msg.detail : ""}`);
  if (msg.status !== "waiting") return;
  for (const [label, type] of [["Approve", "approve"], ["Decline", "decline"]]) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = () => link.send({ type, id: msg.id });
    el.append(" ", b);
  }
}

// ---------- Post drafts ----------

const draftCards = {};  // draft id -> its latest card; older versions stay above, dimmed

function onDraft(msg) {
  if (draftCards[msg.id]) draftCards[msg.id].classList.add("old");
  const el = draftCards[msg.id] = add("bot draft", "");

  const head = document.createElement("div");
  head.className = "head";
  const name = document.createElement("b");
  name.textContent = `Draft ${msg.id}${msg.title ? " · " + msg.title : ""}`;
  const meta = document.createElement("span");
  meta.textContent = `${msg.label} · ${msg.size.join("×")} · v${msg.version}`;
  head.append(name, meta);

  const slides = document.createElement("div");
  slides.className = "slides";
  for (const slide of msg.slides) {
    slides.appendChild(mediaElement({ kind: slide.kind, url: slide.url, name: `slide`, prompt: "" }));
  }

  const caption = document.createElement("div");
  caption.className = "caption" + (msg.caption ? "" : " empty");
  caption.textContent = msg.caption || (msg.caption_limit ? "No caption yet" : "This placement has no caption");

  const problems = (msg.check.checks || []).filter((c) => c.status !== "pass");
  let checks;
  if (problems.length) {
    checks = document.createElement("ul");
    for (const c of problems) {
      const li = document.createElement("li");
      li.className = c.status;
      li.textContent = `${c.label}: ${c.detail}`;
      checks.appendChild(li);
    }
  } else {
    checks = document.createElement("div");
    checks.className = "ok";
    checks.textContent = `✓ Fits ${msg.check.label}` + (msg.caption_limit ? ` · ${[...msg.caption].length}/${msg.caption_limit}` : "");
  }

  el.append(head, ...(msg.slides.length ? [slides] : []), caption, checks);
  log.scrollTop = log.scrollHeight;
}

// ---------- Attachments ----------

async function upload(files) {
  for (const f of files) {
    try {
      await uploadFile(link, f);
    } catch (e) {
      add("err", `Upload failed: ${e.message}`);
    }
  }
}

$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  await upload([...$("file").files]);
  $("file").value = "";
};
acceptFiles(upload);
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

$("bcancel").onclick = () => link.send({ type: "cancel_task" });

// ---------- Microphone ----------

function micOff() {
  link.stopMic();
  $("mic").textContent = "Start talking";
  $("mic").classList.remove("on");
  $("meter").firstElementChild.style.width = "0";
}

function drawMeter() {
  if (!link.micOn) return;
  $("meter").firstElementChild.style.width = `${link.level() * 100}%`;
  requestAnimationFrame(drawMeter);
}

// ---------- Controls ----------

$("mic").onclick = async () => {
  try {
    if (link.micOn) {
      micOff();
    } else {
      await link.startMic();
      $("mic").textContent = "Stop talking";
      $("mic").classList.add("on");
      drawMeter();
    }
  } catch (e) {
    add("err", `Microphone error: ${e.message}`);
    micOff();
  }
};

$("stop").onclick = () => link.interrupt();

$("reset").onclick = () => {
  link.send({ type: "reset" });
  link.stopPlayback();
  log.innerHTML = "";
  for (const id in jobCards) delete jobCards[id];
  for (const id in draftCards) delete draftCards[id];
  for (const id in approvalCards) delete approvalCards[id];
  add("sys", "conversation reset");
};

$("form").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  try {
    await link.say(text);
    $("text").value = "";
  } catch (err) {
    add("err", err.message);
  }
};
