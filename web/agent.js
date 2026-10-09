// The agent's page: talk on the left, the draft as it will look in the middle, drafts, FlowAI posts and
// pictures on the right. Every change comes from the assistant (or a button that does without it), so
// the page only shows events (backend/session.py, top of file) and sends the user's clicks back.
import { VoiceLink, acceptFiles, mediaElement, uploadFile } from "./voice.js";

const $ = (id) => document.getElementById(id);
const log = $("log");

const drafts = {};      // id -> latest draft event
const saved = {};       // draft id -> saved event (its FlowAI post)
const approvals = {};   // id -> approval event
const pictures = {};    // media number -> media event
const jobs = {};        // media job id -> {label, started, status, text}
const slideShown = {};  // draft id -> slide index on screen
let current = null;     // the draft on stage
let focus = {};         // what the user selected: {draft, slide, text}
let accounts = [];
let partialEl = null, botEl = null, botTurn = -1;
let lost = false;       // the connection dropped: the server's session (and its drafts) is gone
const acts = [];        // activity lines waiting for their tool results

const IDEAS = [
  "Make an Instagram post for our new Fig & Cedar candle, launching Friday",
  "What do we have scheduled?",
  "Turn our latest Instagram post into an X post",
];

const link = new VoiceLink({
  onEvent,
  onClose: () => {
    $("conn").textContent = "offline";
    note("Connection lost. Drafts that weren't saved to FlowAI are gone.", "err");
    lost = true;
    reconnect();
  },
});
link.bargeIn = () => $("bargein").checked;

// ---------- helpers ----------

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function svg(paths) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.innerHTML = paths;
  return s;
}

const ICONS = {
  heart: '<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/>',
  comment: '<path d="M20 12a8 8 0 0 1-11.6 7.1L4 20l1-4.3A8 8 0 1 1 20 12z"/>',
  send: '<path d="M21 3L10 14M21 3l-7 18-4-7-7-4z"/>',
  save: '<path d="M6 3h12v18l-6-4-6 4z"/>',
  repost: '<path d="M17 2l3 3-3 3M4 11V9a4 4 0 0 1 4-4h12M7 22l-3-3 3-3M20 13v2a4 4 0 0 1-4 4H4"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  share: '<path d="M12 3v13M7 8l5-5 5 5M5 14v6h14v-6"/>',
  left: '<path d="M15 6l-6 6 6 6"/>',
  right: '<path d="M9 6l6 6-6 6"/>',
  out: '<path d="M7 17L17 7M9 7h8v8"/>',
};

function scrollLog() {
  log.scrollTop = log.scrollHeight;
}

function add(cls, text) {
  const e = el("div", cls, text);
  log.appendChild(e);
  scrollLog();
  return e;
}

function note(text, cls = "sys") {
  return add(cls, text);
}

function handle(platform) {
  return accounts.find((a) => a.platform === platform)?.handle || "yourbrand";
}

function minutes(started) {
  const s = Math.round((Date.now() - started) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- events ----------

function onEvent(msg) {
  switch (msg.type) {
    case "ready":
      if (lost) clearSession();
      lost = false;
      $("conn").textContent = "online";
      link.send({ type: "hello", timezone: Intl.DateTimeFormat().resolvedOptions().timeZone });
      break;
    case "transcript":
      if (!msg.final) {
        if (!partialEl) partialEl = add("msg user partial", "");
        partialEl.textContent = msg.text || "…";
        scrollLog();
      } else {
        partialEl?.remove();
        partialEl = null;
        if (msg.text) add("msg user", msg.text);
      }
      break;
    case "assistant_delta":
      if (!botEl || botTurn !== msg.turn) { botEl = add("msg bot", ""); botTurn = msg.turn; }
      botEl.textContent += msg.text;
      scrollLog();
      break;
    case "assistant_done":
      botEl = null;
      break;
    case "tool_call":
      acts.push({ name: msg.name, el: add("act", describe(msg.name, msg.arguments || {})) });
      botEl = null;
      break;
    case "tool_result":
      onToolResult(msg);
      break;
    case "interrupt":
      if (botEl && msg.was_speaking) botEl.classList.add("cut");
      botEl = null;
      break;
    case "error":
      note(msg.message, "err");
      break;
    case "media":
      onMedia(msg);
      break;
    case "media_job":
      onJob(msg);
      break;
    case "draft":
      drafts[msg.id] = msg;
      current = msg.id;
      renderStage();
      renderDrafts();
      break;
    case "saved":
      saved[msg.draft] = msg;
      renderStage();
      renderDrafts();
      break;
    case "approval":
      approvals[msg.id] = msg;
      renderApprovals();
      break;
    case "posts":
      renderPosts(msg.posts);
      break;
    case "flowai":
      onFlowAI(msg);
      break;
    case "browser_task":
      note(`Website task ${msg.status}${msg.text ? ": " + msg.text : ""}`, "act");
      break;
  }
}

/** What a tool call does, in the words a person would use. */
function describe(name, a) {
  switch (name) {
    case "load_skill": return `Reading the ${String(a.name || "").replaceAll("-", " ")} playbook`;
    case "create_draft": return a.platform === "x" ? "Starting an X post" : `Starting an Instagram ${a.placement || "post"}`;
    case "update_draft":
      return a.caption != null ? "Writing the caption" : a.media ? "Changing the pictures" : "Updating the draft";
    case "add_text": return `Adding “${a.text}”`;
    case "edit_text": return a.text ? `Changing text ${a.text_id} to “${a.text}”` : `Restyling text ${a.text_id}`;
    case "remove_text": return `Removing text ${a.text_id}`;
    case "undo_draft": return "Undoing the last change";
    case "show_draft": return "Showing the draft";
    case "generate_media":
      if (a.draft != null) return /edit/.test(a.workflow || "") ? "Editing the picture" : "Making the picture";
      return "Making a picture";
    case "save_draft": return "Saving to FlowAI";
    case "schedule_post": return "Asking you to approve the time";
    case "find_posts": return "Looking through FlowAI posts";
    case "open_post": return `Opening post ${a.post}`;
    case "find_assets": return "Looking through the gallery";
    case "use_assets": return "Bringing in pictures from the gallery";
    case "browser_task": return `Website task: ${a.instruction}`;
    case "get_current_time": return "Checking the time";
    default: return name.replaceAll("_", " ");
  }
}

function onToolResult(msg) {
  const i = acts.findIndex((a) => a.name === msg.name);
  if (i < 0) return;
  const [act] = acts.splice(i, 1);
  let error = null;
  try { error = JSON.parse(msg.result).error; } catch {}
  if (error) {
    act.el.classList.add("bad");
    act.el.append(" ", el("span", "why", `· ${error}`));
  } else {
    act.el.classList.add("done");
  }
  scrollLog();
}

function onFlowAI(msg) {
  accounts = msg.accounts || [];
  const box = $("accounts");
  box.innerHTML = "";
  if (msg.error) {
    box.appendChild(el("span", "chip off", "FlowAI offline: drafts can't be saved"));
    note(`FlowAI: ${msg.error}`, "err");
  }
  for (const a of accounts) {
    const chip = el("span", "chip");
    chip.append(el("span", "p", a.platform === "x" ? "X" : "IG"), `@${a.handle}`);
    box.appendChild(chip);
  }
  if (msg.dashboard_url) {
    $("flowai-link").href = `${msg.dashboard_url}/dashboard`;
    $("flowai-link").hidden = false;
  }
  if (current) renderStage();
}

// ---------- pictures and generations ----------

function onMedia(msg) {
  if (!["image", "video"].includes(msg.kind)) return;
  pictures[msg.id] = msg;
  $("pictures-none").hidden = true;
  const tile = el("button", "pic");
  tile.title = `Picture ${msg.id}${msg.prompt ? ": " + msg.prompt : ""} (click to mention it)`;
  tile.append(mediaElement({ kind: msg.kind, url: msg.url, alt: msg.prompt || msg.name }), el("b", "", `#${msg.id}`));
  tile.onclick = () => {
    const input = $("text");
    input.value = `${input.value.trim()} picture ${msg.id} `.trimStart();
    input.focus();
  };
  $("pictures").prepend(tile);
  if (msg.source === "upload") note(`You attached picture ${msg.id}`, "act done");
}

function onJob(msg) {
  if (msg.kind === "install") return;
  if (msg.status === "started") {
    jobs[msg.id] = { label: msg.makes === "video" ? "Making the video" : "Making the picture", started: Date.now() };
  }
  const job = jobs[msg.id];
  if (!job) return;
  job.status = msg.status;
  job.text = msg.text || "";
  if (msg.status === "done" || msg.status === "cancelled") delete jobs[msg.id];
  if (msg.status === "failed") setTimeout(() => { delete jobs[msg.id]; renderJobs(); }, 8000);
  renderJobs();
}

function renderJobs() {
  const box = $("jobs");
  box.innerHTML = "";
  for (const job of Object.values(jobs)) {
    const failed = job.status === "failed";
    box.appendChild(el("span", "job" + (failed ? " failed" : ""),
      failed ? `Couldn't make it: ${job.text}` : `${job.label} · ${minutes(job.started)}`));
  }
}
setInterval(() => { if (Object.keys(jobs).length) renderJobs(); }, 1000);

// ---------- the stage ----------

function renderStage() {
  const d = drafts[current];
  $("empty").hidden = !!d;
  $("draft").hidden = !d;
  $("preview").innerHTML = "";
  $("checks").innerHTML = "";
  if (!d) return;

  $("draft-title").textContent = d.title || `Draft ${d.id}`;
  $("draft-meta").textContent = `Draft ${d.id} · ${d.label} · ${d.size.join("×")} · v${d.version}`;
  $("undo").disabled = d.version < 2;

  const s = saved[d.id];
  const fresh = s && s.version === d.version;
  $("save").hidden = !!fresh || !accounts.length;
  $("save").querySelector("span").textContent = s ? "Save changes" : "Save to FlowAI";
  $("saved").hidden = !s;
  if (s) {
    $("saved").className = "saved" + (fresh ? "" : " stale");
    $("saved").textContent = fresh
      ? `Saved · post ${s.post} · ${s.status}${s.when ? " " + s.when : ""}`
      : `Post ${s.post} · unsaved changes`;
  }
  $("open-post").hidden = !s;
  if (s) $("open-post").href = s.url;

  $("preview").appendChild(d.platform === "x" ? tweet(d) : d.placement === "story" ? story(d) : instagram(d));
  renderChecks(d);
}

function slideFrame(d, i) {
  const slide = d.slides[i];
  const frame = el("div", "frame");
  frame.style.aspectRatio = `${d.size[0]} / ${d.size[1]}`;
  if (focus.draft === d.id && focus.slide === i + 1 && !focus.text) frame.classList.add("picked");
  if (slide) frame.appendChild(mediaElement({ kind: slide.kind, url: slide.url, alt: `Slide ${i + 1}` }));
  for (const t of slide?.texts || []) {
    const [x, y, w, h] = t.box;
    const hit = el("button", "hit");
    hit.style.cssText = `left:${x * 100}%;top:${y * 100}%;width:${w * 100}%;height:${h * 100}%`;
    hit.title = `Text ${t.id}: ${t.words}. Click to select it, then say what to change.`;
    hit.appendChild(el("span", "tag", `Text ${t.id}`));
    if (focus.draft === d.id && focus.text === t.id) hit.classList.add("on");
    hit.onclick = (e) => {
      e.stopPropagation();
      select(focus.text === t.id ? { draft: d.id, slide: i + 1 } : { draft: d.id, slide: i + 1, text: t.id });
    };
    frame.appendChild(hit);
  }
  frame.onclick = () => {
    const same = focus.draft === d.id && focus.slide === i + 1 && !focus.text;
    select(same ? { draft: d.id } : { draft: d.id, slide: i + 1 });
  };
  return frame;
}

/** One slide at a time, with arrows and dots, like an Instagram carousel. */
function carousel(d, withBars) {
  const n = d.slides.length;
  const i = Math.min(slideShown[d.id] ?? 0, Math.max(0, n - 1));
  slideShown[d.id] = i;
  const frame = slideFrame(d, i);
  if (!n) {
    frame.appendChild(el("div", "count", "no picture yet"));
    return { frame, dots: null };
  }
  if (withBars) {
    const bars = el("div", "bars");
    for (let k = 0; k < n; k++) bars.appendChild(el("i", k === i ? "on" : ""));
    frame.appendChild(bars);
  }
  let dots = null;
  if (n > 1) {
    frame.appendChild(el("div", "count", `${i + 1}/${n}`));
    for (const [cls, step, icon] of [["prev", -1, "left"], ["next", 1, "right"]]) {
      if ((step < 0 && i === 0) || (step > 0 && i === n - 1)) continue;
      const b = el("button", `nav ${cls}`);
      b.title = step < 0 ? "Previous slide" : "Next slide";
      b.appendChild(svg(ICONS[icon]));
      b.onclick = (e) => { e.stopPropagation(); slideShown[d.id] = i + step; renderStage(); };
      frame.appendChild(b);
    }
    dots = el("div", "dots");
    for (let k = 0; k < n; k++) dots.appendChild(el("i", k === i ? "on" : ""));
  }
  return { frame, dots };
}

function captionText(text, limit, into) {
  const chars = [...text];
  into.append(chars.slice(0, limit).join(""));
  if (chars.length > limit) into.appendChild(el("mark", "over", chars.slice(limit).join("")));
}

function instagram(d) {
  const card = el("article", "post ig");
  const head = el("div", "ig-head");
  head.append(el("span", "avatar"), handle("instagram"));
  const { frame, dots } = carousel(d, false);
  const actions = el("div", "ig-actions");
  actions.append(svg(ICONS.heart), svg(ICONS.comment), svg(ICONS.send));
  const end = svg(ICONS.save);
  end.classList.add("end");
  actions.appendChild(end);
  const caption = el("p", "caption");
  if (d.caption) {
    caption.appendChild(el("span", "who", handle("instagram")));
    const body = el("span");
    captionText(d.caption, d.caption_limit, body);
    caption.appendChild(body);
  } else {
    caption.classList.add("empty");
    caption.textContent = d.caption_limit ? "No caption yet" : "This placement shows no caption";
  }
  card.append(head, frame, ...(dots ? [dots] : []), actions, caption);
  return card;
}

function story(d) {
  const card = el("article", "post story");
  const { frame } = carousel(d, true);
  const head = el("div", "story-head");
  head.append(el("span", "avatar"), handle("instagram"));
  frame.appendChild(head);
  card.appendChild(frame);
  return card;
}

function tweet(d) {
  const card = el("article", "post x");
  const body = el("div", "x-body");
  const head = el("div", "x-head");
  const brand = accounts.find((a) => a.platform === "x");
  head.append(el("b", "", brand?.name || handle("x")), el("span", "", `@${handle("x")} · now`));
  const text = el("p", "x-text" + (d.caption ? "" : " empty"));
  if (d.caption) captionText(d.caption, d.caption_limit, text);
  else text.textContent = "No text yet";
  body.append(head, text);
  if (d.slides.length) {
    const media = el("div", "x-media" + (d.slides.length > 1 ? " many" : ""));
    d.slides.forEach((_, i) => media.appendChild(slideFrame(d, i)));
    body.appendChild(media);
  }
  const actions = el("div", "x-actions");
  actions.append(svg(ICONS.comment), svg(ICONS.repost), svg(ICONS.heart), svg(ICONS.chart), svg(ICONS.share));
  body.appendChild(actions);
  card.append(el("span", "avatar"), body);
  return card;
}

function renderChecks(d) {
  const box = $("checks");
  const problems = (d.check.checks || []).filter((c) => c.status !== "pass");
  if (!problems.length) {
    const used = d.caption_limit ? ` · caption ${[...d.caption].length}/${d.caption_limit}` : "";
    box.appendChild(el("li", "", `Fits ${d.check.label}${used}`));
    return;
  }
  for (const c of problems) box.appendChild(el("li", c.status, `${c.label}: ${c.detail}`));
}

// ---------- selection: what "this" means ----------

function select(next) {
  focus = next;
  link.send({ type: "focus", ...focus });
  const d = drafts[focus.draft];
  let what = "";
  if (d && focus.text) {
    const t = d.slides.flatMap((s) => s.texts || []).find((t) => t.id === focus.text);
    what = `Text ${focus.text}${t ? ` “${t.words}”` : ""}`;
  } else if (d && focus.slide) {
    what = `Slide ${focus.slide} of draft ${d.id}`;
  }
  $("selection").hidden = !what;
  $("selection-text").textContent = what ? `Selected: ${what}. Say what to change.` : "";
  renderStage();
}

$("selection-clear").onclick = () => select(current ? { draft: current } : {});

// ---------- side: drafts, posts ----------

function renderDrafts() {
  const box = $("drafts");
  box.innerHTML = "";
  const list = Object.values(drafts).sort((a, b) => b.id - a.id);
  if (!list.length) box.appendChild(el("div", "none", "Nothing yet."));
  for (const d of list) {
    const row = el("button", "row" + (d.id === current ? " on" : ""));
    const first = d.slides.find((s) => s.kind === "image");
    const thumb = first ? mediaElement({ kind: "image", url: first.url }) : el("span");
    thumb.className = "thumb";
    const txt = el("span", "txt");
    const s = saved[d.id];
    txt.append(el("b", "", d.title || `Draft ${d.id}`),
      el("span", "", `${d.label} · v${d.version}${s ? (s.version === d.version ? ` · post ${s.post}` : " · unsaved") : ""}`));
    row.append(thumb, txt);
    if (s) row.appendChild(el("span", `status ${s.status}`, s.status));
    row.onclick = () => { current = d.id; select({ draft: d.id }); renderDrafts(); };
    box.appendChild(row);
  }
}

function renderPosts(posts) {
  const box = $("posts");
  box.innerHTML = "";
  if (!posts.length) box.appendChild(el("div", "none", "No posts in FlowAI yet."));
  for (const p of posts) {
    const row = el("button", "row");
    row.title = `Open post ${p.post} here to change it`;
    const txt = el("span", "txt");
    txt.append(el("b", "", p.title || `Post ${p.post}`), el("span", "", [p.kind, p.when].filter(Boolean).join(" · ")));
    const out = el("a", "out");
    out.href = p.url; out.target = "_blank"; out.rel = "noopener"; out.title = "Open in FlowAI's Composer";
    out.appendChild(svg(ICONS.out));
    out.onclick = (e) => e.stopPropagation();
    row.append(txt, el("span", `status ${p.status}`, p.status), out);
    row.onclick = () => action({ name: "open_post", post: p.post });
    box.appendChild(row);
  }
}

function renderApprovals() {
  const box = $("approvals");
  box.innerHTML = "";
  const list = Object.values(approvals).sort((a, b) => b.id - a.id);
  const shown = [...list.filter((a) => a.status === "waiting"), ...list.filter((a) => a.status !== "waiting").slice(0, 2)];
  for (const a of shown) {
    const card = el("div", "approval" + (a.status === "waiting" ? "" : ` closed ${a.status}`));
    const what = el("div", "what");
    const title = { waiting: "Needs your approval", approved: "Approved", declined: "Declined",
      failed: "Couldn't schedule", outdated: "No longer current" }[a.status] || a.status;
    what.append(el("b", "", title), el("span", "", a.text));
    if (a.detail && a.status !== "approved") what.append(el("div", "hint", a.detail));  // approved: the text says it
    card.appendChild(what);
    if (a.status === "waiting") {
      const no = el("button", "ghost", "Decline");
      no.onclick = () => link.send({ type: "decline", id: a.id });
      const yes = el("button", "primary", "Approve");
      yes.onclick = () => { yes.disabled = true; link.send({ type: "approve", id: a.id }); };
      card.append(no, yes);
    }
    box.appendChild(card);
  }
}

// ---------- buttons that need no assistant ----------

async function action(msg) {
  try {
    await link.connect();
    link.send({ type: "action", ...msg });
  } catch (e) {
    note(e.message, "err");
  }
}

$("undo").onclick = () => action({ name: "undo", draft: current });
$("save").onclick = () => action({ name: "save", draft: current });

// ---------- talking ----------

function updateVoice() {
  const st = link.state;
  const words = {
    disconnected: ["Offline", "Reconnecting…"],
    connected: ["Tap to talk", "or type below. You can talk over the assistant."],
    listening: ["Listening", "Go ahead, I'm listening."],
    hearing: ["Hearing you", "Pause when you're done."],
    thinking: ["Working on it", "Say “stop” or tap ■ to cut in."],
    speaking: ["Speaking", "Talk over me to interrupt."],
  }[st];
  $("state").textContent = words[0];
  $("state-hint").textContent = words[1];
  $("stop").hidden = !["thinking", "speaking"].includes(st);
  const level = link.micOn ? link.level() : 0;
  $("mic").style.boxShadow = link.micOn ? `0 0 0 ${3 + level * 14}px rgb(229 72 77 / .25)` : "";
  requestAnimationFrame(updateVoice);
}
requestAnimationFrame(updateVoice);

$("mic").onclick = async () => {
  try {
    if (link.micOn) {
      link.stopMic();
    } else {
      await link.startMic();
    }
  } catch (e) {
    note(`Microphone: ${e.message}`, "err");
    link.stopMic();
  }
  $("mic").classList.toggle("on", link.micOn);
  $("mic").setAttribute("aria-label", link.micOn ? "Stop talking" : "Start talking");
};

$("stop").onclick = () => link.interrupt();

async function say(text) {
  try {
    await link.say(text);
  } catch (e) {
    note(e.message, "err");
  }
}

$("form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  $("text").value = "";
  say(text);
};

for (const idea of IDEAS) {
  const b = el("button", "idea", `“${idea}”`);
  b.onclick = () => say(idea);
  $("ideas").appendChild(b);
}

function clearSession() {
  for (const box of [drafts, saved, approvals, jobs, slideShown, pictures]) for (const k in box) delete box[k];
  current = null;
  focus = {};
  acts.length = 0;
  $("selection").hidden = true;
  $("pictures").innerHTML = "";
  $("pictures-none").hidden = false;
  renderStage(); renderDrafts(); renderApprovals(); renderJobs();
}

$("reset").onclick = () => {
  if (!confirm("Start over? The conversation and its drafts are cleared; posts saved to FlowAI stay.")) return;
  link.send({ type: "reset" });
  link.stopPlayback();
  log.innerHTML = "";
  clearSession();
  note("New session");
};

// ---------- attachments ----------

async function upload(files) {
  for (const f of files) {
    try {
      await uploadFile(link, f);
    } catch (e) {
      note(`Upload failed: ${e.message}`, "err");
    }
  }
}

$("attach").onclick = () => $("file").click();
$("file").onchange = async () => {
  await upload([...$("file").files]);
  $("file").value = "";
};
acceptFiles(upload);

// ---------- connection ----------

let retry = 0;
async function reconnect() {
  try {
    await link.connect();
    retry = 0;
  } catch {
    retry = Math.min(retry + 1, 5);
    setTimeout(reconnect, 1000 * 2 ** retry);
  }
}

// On a phone the composer is fixed at the bottom: keep the page's end clear of it
new ResizeObserver(([entry]) => {
  document.body.style.setProperty("--composer-h", `${Math.ceil(entry.borderBoxSize[0].blockSize)}px`);
}).observe(document.querySelector(".composer"));

// Connect right away: FlowAI's accounts and recent posts show before the first word
reconnect();
