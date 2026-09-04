// CONCERTO 6 · distillation
//
// A simulated distillation run, built as one causal chain:
//
//     racks  ->  capacity (tok/s)  ->  live throughput  ->  sessions
//     sessions x per-session speed  ->  samples  ->  judge  ->  dataset
//
// Nothing downstream is a free parameter: you pick how many NVL72 Rubin racks
// you are renting and how fast one session is allowed to decode, and every
// other number on screen follows from that.
//
// The code in the panels is pre-written and replayed; only the timing is
// simulated — deliberately uneven, the way a real fleet is: per-session speed
// jitters, sessions wait for prefill, some sit queued, a few are stragglers.

(() => {
"use strict";

// ---------------------------------------------------------------- palette
const C = {
  bg:      "#000000",
  panel:   "#080808",
  panelAlt:"#050505",
  border:  "#1c1c1c",
  head:    "#4a4a4a",
  headNum: "#8a8a8a",
  base:    "#9a9a9a",
  kw:      "#ffffff",
  str:     "#d0d0d0",
  num:     "#d0d0d0",
  com:     "#4a4a4a",
  punct:   "#6a6a6a",
  ident:   "#aaaaaa",
  cursor:  "#ffffff",
  keep:    "#ffffff",
  drop:    "#3d3d3d",
};

const KEYWORDS = new Set((
  "def class return if elif else for while import from in not and or is None True False " +
  "self async await try except finally raise with yield lambda pass break continue global " +
  "assert del as match case struct impl fn let mut pub use crate enum trait where const static " +
  "unsafe move ref type function var new export default extern package interface implements " +
  "extends throw catch this super typeof instanceof void delete do switch break case go defer " +
  "chan range map make nil err int float bool char size_t uint8_t uint32_t sizeof template " +
  "public private protected virtual override final abstract " +
  "fi then do done esac local readonly export set trap shift source until exit echo"
).split(" "));

const FILES = SNIPPETS.map((s, i) => ({ title: s.t, toks: LEXER.tokenize(s.code), idx: i }));
const MEAN_SAMPLE = FILES.reduce((a, f) => a + f.toks.length, 0) / FILES.length;

// ---------------------------------------------------------------- run
const SIM_CAP = 1024;        // sessions actually running in this browser
const CORPUS_MIN = 5e9;      // what a previous run already distilled
const CORPUS_MAX = 10e9;
const CORPUS_TARGET = 100e9; // what the dataset needs
const JUDGE_PASS = 0.61;     // share of samples the judge lets through

const PER_SESSION = [10, 20, 40, 60, 100, 200, 750];
const SHOW_COUNTS = [1, 4, 9, 16, 32, 64, 128];
const SIZES = [
  { id: "dense",  name: "dense",  cols: 62 },
  { id: "normal", name: "normal", cols: 48 },
  { id: "big",    name: "big",    cols: 38 },
  { id: "huge",   name: "huge",   cols: 28 },
];

const state = {
  racks: 1,
  teacher: HARDWARE.TEACHERS[0],
  per: 40,
  show: 9,
  size: 38,      // target columns per panel; fewer columns = bigger text
  page: 0,
  paused: false,
  focus: null,
};

let capacity = 0;       // tok/s the rented racks can do, flat out
let fleetRate = 0;      // tok/s they are doing right now
let sessions = 1;       // live, = fleetRate / per-session speed
let streams = [];       // the sample of sessions actually running here
let panels = [];
let clock = 0;

let painted = 0;
let samplesRate = 0;
let keptRatio = 0;
let keptSim = 0, doneSim = 0;
let avgSample = 0;
let corpusStart = CORPUS_MIN;
let corpus = CORPUS_MIN;
let corpusRate = 0;
let fps = 60;
let seed = 1;

function rnd() { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }
const blankLine = () => ({ runs: [], len: 0, txt: "" });

// A session's own speed: mostly within ±30 % of nominal, with the odd
// straggler stuck behind a long context.
function newMult() {
  if (rnd() < 0.07) return 0.28 + rnd() * 0.25;
  return 0.74 + rnd() * 0.52;
}

function makeStream(i, sim) {
  const file = FILES[Math.floor(rnd() * FILES.length)];
  return {
    id: Math.floor((i * sessions) / sim) + 1,
    file,
    p: Math.floor(rnd() * file.toks.length * 0.6),
    acc: 0,
    mult: newMult(),
    jitterAt: rnd() * 5,
    phase: "decode",     // decode | prefill | queued
    wait: rnd() * 1.5 * Math.max(0.22, Math.min(1.5, 40 / state.per)),
    rateNow: 0,
    sampleTok: 0,
    partial: true,
    lines: [blankLine()],
    cols: 30,
    rows: 8,
    dirty: true,
  };
}

function rebuild() {
  capacity = HARDWARE.rackCapacity(state.teacher) * state.racks;
  fleetRate = capacity * 0.94;
  sessions = Math.max(1, Math.round(fleetRate / state.per));
  const affordable = Math.floor(MAX_FRAME_TOTAL * 40 / state.per);
  const sim = Math.max(Math.min(sessions, state.show),
    Math.min(sessions, SIM_CAP, affordable));
  streams = [];
  for (let i = 0; i < sim; i++) streams.push(makeStream(i, sim));
  clock = 0;
  painted = 0;
  keptSim = doneSim = 0;
  avgSample = MEAN_SAMPLE;

  // The judge's pass rate is a property of the corpus: start from the design
  // value, let the measured one take over once enough samples have landed.
  keptRatio = JUDGE_PASS;
  samplesRate = fleetRate / MEAN_SAMPLE;
  corpusRate = fleetRate * keptRatio;
  corpusStart = Math.round((CORPUS_MIN + rnd() * (CORPUS_MAX - CORPUS_MIN)) / 1e8) * 1e8;
  corpus = corpusStart;
  state.page = 0;
  state.focus = null;
  describe();
  layout();
}

// ---------------------------------------------------------------- canvas
const wall = document.getElementById("wall");
const cv = document.getElementById("cv");
const ctx = cv.getContext("2d", { alpha: false });

let W = 0, H = 0, dpr = 1, fs = 12, lineH = 16, charW = 7, headH = 20;
let flatDraw = false;
let visible = [];
let pages = 1;
let idPad = 6;

function fontStack(px) {
  return px + 'px ui-monospace,"JetBrains Mono","IBM Plex Mono","SF Mono",Menlo,Consolas,monospace';
}

function pageWindow() {
  if (state.focus !== null) return [Math.min(state.focus, streams.length - 1)];
  const show = Math.min(state.show, streams.length);
  pages = Math.max(1, Math.ceil(streams.length / show));
  if (state.page >= pages) state.page = 0;
  const start = state.page * show;
  const out = [];
  for (let i = start; i < Math.min(start + show, streams.length); i++) out.push(i);
  return out;
}

function layout() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = wall.clientWidth;
  H = wall.clientHeight;
  cv.width = Math.max(1, Math.round(W * dpr));
  cv.height = Math.max(1, Math.round(H * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.textBaseline = "top";

  visible = pageWindow();
  const n = visible.length;

  let bestCols = 1, bestScore = Infinity;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const pw = W / cols, ph = H / rows;
    // Empty slots look like a bug, so they cost more than an odd aspect.
    const score = Math.abs(pw / ph - 1.5) + (cols * rows - n) * 0.25
      + (pw < 90 ? (90 - pw) * 0.05 : 0) + (ph < 60 ? (60 - ph) * 0.05 : 0);
    if (score < bestScore) { bestScore = score; bestCols = cols; }
  }
  const cols = bestCols;
  const rows = Math.ceil(n / cols);
  const pw = Math.floor(W / cols);
  const ph = Math.floor(H / rows);

  // Size the text from the panel width so a panel always holds about
  // `state.size` columns of code — that is what makes it readable.
  const byWidth = (pw - 14) / (state.size * 0.6);
  const byHeight = ph / 11;
  fs = Math.max(7, Math.min(30, Math.round(Math.min(byWidth, byHeight))));
  lineH = Math.round(fs * 1.35);
  headH = fs + 11;
  ctx.font = fontStack(fs);
  charW = ctx.measureText("MMMMMMMMMM").width / 10;
  flatDraw = fs < 10;

  const textRows = Math.max(2, Math.floor((ph - headH - 6) / lineH));
  const textCols = Math.max(10, Math.floor((pw - 14) / charW));

  panels = visible.map((sIdx, k) => {
    const cx = k % cols, cy = (k / cols) | 0;
    return {
      s: sIdx,
      x: cx * pw,
      y: cy * ph,
      w: cx === cols - 1 ? W - cx * pw : pw,
      h: cy === rows - 1 ? H - cy * ph : ph,
    };
  });

  for (const s of streams) {
    if (s.cols !== textCols || s.rows !== textRows) {
      s.cols = textCols;
      s.rows = textRows;
      s.lines = [blankLine()];
    }
    s.dirty = true;
  }

  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  for (const p of panels) drawPanel(p);
  updatePageInfo();
}

// ---------------------------------------------------------------- text buffer
function newline(s) {
  s.lines.push(blankLine());
  while (s.lines.length > s.rows) s.lines.shift();
}

function pushText(s, text, cls) {
  let rest = text;
  while (rest.length) {
    const line = s.lines[s.lines.length - 1];
    const room = s.cols - line.len;
    if (room <= 0) { newline(s); continue; }
    const take = rest.slice(0, room);
    rest = rest.slice(room);
    const last = line.runs[line.runs.length - 1];
    if (last && last.c === cls) last.t += take;
    else line.runs.push({ t: take, c: cls });
    line.len += take.length;
    line.txt += take;
    if (rest.length) newline(s);
  }
}

function feed(s, tok) {
  if (tok.t === "\n") { newline(s); return; }
  pushText(s, tok.t, tok.c);
}

// A finished sample is scored by the judge; only what passes becomes corpus.
function finishSample(s) {
  const score = 0.12 + rnd() * 0.88;
  const kept = score >= 0.46;
  doneSim++;
  if (s.partial) s.partial = false;
  else avgSample += (s.sampleTok - avgSample) * 0.02;
  // Measured on the sample, but it is a property of the corpus, not of how
  // many sessions this browser managed to run.
  if (kept) keptSim++;
  if (doneSim > 40) keptRatio = keptSim / doneSim;

  newline(s);
  pushText(s, kept ? "✓ kept · score " + score.toFixed(2) + " → dataset"
                   : "✗ dropped · score " + score.toFixed(2), kept ? "keep" : "drop");
  newline(s);

  s.sampleTok = 0;
  s.file = FILES[(s.file.idx + 1 + Math.floor(rnd() * (FILES.length - 1))) % FILES.length];
  s.p = 0;

  // Nobody starts the next one on the same tick: prefill, or wait for a slot.
  const g = waitScale();
  const r = rnd();
  if (r < 0.55) { s.phase = "prefill"; s.wait = (0.25 + rnd() * 1.1) * g; }
  else if (r < 0.78) { s.phase = "queued"; s.wait = (0.8 + rnd() * 3.4) * g; }
  else { s.phase = "decode"; s.wait = (0.05 + rnd() * 0.2) * g; }
}

// A fast session runs on a latency-tuned config, so its gaps shrink with it.
function waitScale() {
  return Math.max(0.22, Math.min(1.5, 40 / state.per));
}

function startSample(s) {
  s.phase = "decode";
  pushText(s, "── " + s.file.title, "com");
  newline(s);
}

// ---------------------------------------------------------------- clock
const MAX_PER_FRAME = 100;
const MAX_FRAME_TOTAL = 12000;
let rr = 0;
let wander = 1;

// Fleet utilisation never sits still: a slow swell, a faster ripple, drift.
function utilisation(t) {
  wander += (rnd() - 0.5) * 0.010;
  wander = Math.max(0.90, Math.min(1.0, wander));
  return Math.min(1, wander * (1 + 0.035 * Math.sin(t / 17) + 0.02 * Math.sin(t / 3.1 + 1.7)));
}

function step(dt, realDt) {
  const n = streams.length;
  if (!n) return;
  clock += realDt;

  // The whole chain starts here: what the racks are actually delivering.
  const util = utilisation(clock);
  const a = Math.min(1, realDt * 3);
  fleetRate += (capacity * util - fleetRate) * a;
  sessions = Math.max(1, Math.round(fleetRate / state.per));

  const stepDt = Math.min(dt, 1 / 25);
  let emitted = 0;

  for (let k = 0; k < n; k++) {
    const s = streams[(rr + k) % n];

    if (s.wait > 0) {
      s.wait -= stepDt;
      s.rateNow = 0;
      if (s.wait <= 0) { startSample(s); s.dirty = true; }
      continue;
    }

    if (clock > s.jitterAt) { s.mult = newMult(); s.jitterAt = clock + 2 + rnd() * 6; }

    const rate = state.per * s.mult * util;
    s.rateNow = rate;

    s.acc = Math.min(s.acc + rate * stepDt, rate * 4 + MAX_PER_FRAME);
    let take = Math.floor(s.acc);
    if (take <= 0) continue;
    if (take > MAX_PER_FRAME) take = MAX_PER_FRAME;
    if (emitted + take > MAX_FRAME_TOTAL) take = MAX_FRAME_TOTAL - emitted;
    if (take <= 0) { rr = (rr + k) % n; break; }
    s.acc -= take;

    const toks = s.file.toks;
    for (let i = 0; i < take; i++) {
      if (s.p >= toks.length) { finishSample(s); break; }
      feed(s, toks[s.p++]);
      s.sampleTok++;
    }
    emitted += take;
    s.dirty = true;
  }
  rr = (rr + 1) % n;

  painted += (emitted / Math.max(realDt, 1e-4) - painted) * a;

  // Every token the fleet emits belongs to some sample, and the judge keeps a
  // measured share of them: the dataset therefore fills at fleet x kept, and
  // samples/s is simply throughput / sample length. No extrapolation from a
  // starved sample, so a slow browser can never fake a slow run.
  samplesRate = fleetRate / Math.max(avgSample, 1);
  corpusRate = fleetRate * keptRatio;
  corpus = Math.min(CORPUS_TARGET, corpus + corpusRate * realDt);
}

// ---------------------------------------------------------------- drawing
function drawPanel(p) {
  const s = streams[p.s];
  if (!s) return;
  const { x, y, w, h } = p;

  ctx.fillStyle = (p.s & 1) ? C.panel : C.panelAlt;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = C.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.font = fontStack(fs);

  const pad = Math.round(fs * 0.5);
  const idx = "#" + String(s.id).padStart(idPad, "0");
  ctx.fillStyle = C.headNum;
  ctx.fillText(idx, x + pad, y + 4);

  const statusTxt = s.phase === "decode" ? Math.round(s.rateNow) + " tok/s" : s.phase;
  const statusW = ctx.measureText(statusTxt).width;
  ctx.fillStyle = s.phase === "decode" ? C.head : C.drop;
  if (statusW < w * 0.45) ctx.fillText(statusTxt, x + w - pad - statusW, y + 4);

  const titleX = x + pad + charW * (idx.length + 1);
  const titleRoom = Math.floor((w - pad * 2 - statusW - charW * (idx.length + 2)) / charW);
  if (titleRoom > 3) {
    let title = s.file.title;
    if (title.length > titleRoom) title = title.slice(0, titleRoom - 1) + "…";
    ctx.fillStyle = C.head;
    ctx.fillText(title, titleX, y + 4);
  }
  ctx.fillStyle = C.border;
  ctx.fillRect(x + 1, y + headH - 4, w - 2, 1);

  const top = y + headH;
  const rowsToDraw = Math.min(s.lines.length, s.rows);
  const first = s.lines.length - rowsToDraw;
  for (let r = 0; r < rowsToDraw; r++) {
    const line = s.lines[first + r];
    const ly = top + r * lineH;
    if (ly + lineH > y + h) break;
    let col = line.len;
    if (flatDraw) {
      ctx.fillStyle = C.base;
      ctx.fillText(line.txt, x + pad, ly);
    } else {
      col = 0;
      for (const run of line.runs) {
        ctx.fillStyle = C[run.c] || C.base;
        ctx.fillText(run.t, x + pad + col * charW, ly);
        col += run.t.length;
      }
    }
    if (first + r === s.lines.length - 1 && s.phase === "decode") {
      ctx.fillStyle = C.cursor;
      ctx.globalAlpha = 0.55;
      ctx.fillRect(x + pad + col * charW, ly + 1, Math.max(2, charW * 0.75), fs);
      ctx.globalAlpha = 1;
    }
  }
}

// ---------------------------------------------------------------- hud
const $ = (id) => document.getElementById(id);
const el = {
  brand: $("brand"), hw: $("hwLine"), tokrate: $("tokrate"), maxLine: $("maxLine"),
  gauge: $("gaugeFill"), state: $("state"),
  sSessions: $("sSessions"), sSamples: $("sSamples"), sAvg: $("sAvg"),
  sCorpus: $("sCorpus"), sEta: $("sEta"),
  pageInfo: $("pageInfo"), simInfo: $("simInfo"),
};

const fmt = (n) => Math.round(n).toLocaleString("fr-FR").replace(/ | /g, " ");
const short = (n) => n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B"
  : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 2) + "M"
  : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(Math.round(n));

function hms(sec) {
  if (!isFinite(sec) || sec <= 0 || sec > 86400 * 60) return "—";
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60);
  if (d) return d + "d" + String(h).padStart(2, "0");
  if (h) return h + "h" + String(m).padStart(2, "0");
  return m + "m" + String(Math.floor(sec % 60)).padStart(2, "0");
}

function describe() {
  const m = state.teacher;
  el.brand.textContent = m.label;
  document.title = m.label + " · distillation";
  const size = m.total >= 1e12 ? (m.total / 1e12).toFixed(1) + "T" : Math.round(m.total / 1e9) + "B";
  el.hw.textContent = size + "-A" + Math.round(m.active / 1e9) + "B · NVFP4 · "
    + m.mtp + "× MTP · RUBIN NVL72";
  el.maxLine.textContent = "max " + short(capacity);
  idPad = Math.max(4, String(Math.round(capacity / state.per)).length);
}

function updatePageInfo() {
  el.pageInfo.textContent = state.focus !== null
    ? "focus #" + String(streams[state.focus] ? streams[state.focus].id : 0).padStart(idPad, "0")
    : "page " + (state.page + 1) + "/" + pages;
}

let hudAcc = 0;
function updateHud(dt) {
  hudAcc += dt;
  if (hudAcc < 0.12) return;
  hudAcc = 0;

  el.tokrate.textContent = fmt(fleetRate);
  el.sSessions.textContent = fmt(sessions);
  el.sSamples.textContent = samplesRate >= 100 ? fmt(samplesRate) : samplesRate.toFixed(1);
  el.sAvg.textContent = Math.round(avgSample) + " tok";
  el.sCorpus.textContent = (corpus / 1e9).toFixed(1) + "B / 100B";

  const left = Math.max(0, CORPUS_TARGET - corpus);
  el.sEta.textContent = hms(left / Math.max(corpusRate, 1));
  el.gauge.style.width =
    (100 * (corpus - corpusStart) / (CORPUS_TARGET - corpusStart)).toFixed(2) + "%";

  el.simInfo.textContent = "sim " + fmt(streams.length) + " · painted " + short(painted)
    + " · " + Math.round(fps) + " fps";

  const behind = painted < streams.length * state.per * 0.75;
  el.state.textContent = state.paused ? "held" : behind ? "throttled" : "running";
  el.state.classList.toggle("on", !state.paused && !behind);
}

// ---------------------------------------------------------------- controls
function segment(host, items, isOn, onPick) {
  host.innerHTML = "";
  for (const item of items) {
    const b = document.createElement("button");
    b.textContent = item.name;
    if (isOn(item)) b.classList.add("on");
    b.onclick = () => onPick(item);
    host.appendChild(b);
  }
}

function renderControls() {
  segment($("racks"), HARDWARE.RACKS, (r) => r.id === state.racks,
    (r) => { state.racks = r.id; rebuild(); renderControls(); });

  segment($("teachers"), HARDWARE.TEACHERS, (m) => m.id === state.teacher.id,
    (m) => { state.teacher = m; rebuild(); renderControls(); });

  segment($("perStream"), PER_SESSION.map((c) => ({ id: c, name: String(c) })),
    (c) => c.id === state.per,
    (c) => { state.per = c.id; rebuild(); renderControls(); });

  segment($("shown"), SHOW_COUNTS.map((c) => ({ id: c, name: String(c) })),
    (c) => c.id === state.show,
    (c) => { state.show = c.id; state.focus = null; state.page = 0; layout(); renderControls(); });

  segment($("density"), SIZES, (d) => d.cols === state.size,
    (d) => { state.size = d.cols; layout(); renderControls(); });
}

cv.onclick = (ev) => {
  const r = cv.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  if (state.focus !== null) { state.focus = null; layout(); return; }
  const hit = panels.find((p) => mx >= p.x && mx < p.x + p.w && my >= p.y && my < p.y + p.h);
  if (hit) { state.focus = hit.s; layout(); }
};

function turnPage(d) {
  if (state.focus !== null) return;
  state.page = (state.page + d + pages) % pages;
  layout();
}

window.addEventListener("keydown", (ev) => {
  if (ev.code === "Space") { ev.preventDefault(); state.paused = !state.paused; }
  else if (ev.key === "]" || ev.key === "ArrowRight") turnPage(1);
  else if (ev.key === "[" || ev.key === "ArrowLeft") turnPage(-1);
  else if (ev.key === "r" || ev.key === "R") rebuild();
  else if (ev.key === "h" || ev.key === "H") {
    // Hide the settings row only — the run's numbers stay on screen.
    const row = $("settings");
    row.style.display = row.style.display === "none" ? "" : "none";
    layout();
  } else if (ev.key === "Escape" && state.focus !== null) { state.focus = null; layout(); }
});

let resizeTimer = 0;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(layout, 80);
});

// ---------------------------------------------------------------- loop
let last = performance.now();
let lastFrameAt = last;

function tick(now) {
  const realDt = Math.max((now - last) / 1000, 1e-4);
  const dt = Math.min(0.25, realDt);
  last = now;
  fps += (1 / realDt - fps) * 0.08;
  if (!state.paused) step(dt, realDt);
  for (const p of panels) {
    const s = streams[p.s];
    if (s && s.dirty) { drawPanel(p); s.dirty = false; }
  }
  updateHud(dt);
}

function frame(now) {
  lastFrameAt = performance.now();
  tick(now);
  requestAnimationFrame(frame);
}

// Some hosts (background tabs, embedded previews) never fire rAF. Keep the
// run alive on a timer whenever frames stop arriving.
setInterval(() => {
  if (performance.now() - lastFrameAt > 400) tick(performance.now());
}, 40);

renderControls();
rebuild();
requestAnimationFrame(frame);
})();
