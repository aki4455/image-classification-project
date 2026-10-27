/* Car Classifier web UI: photos, live camera, and a page about the model.
   Talks to web/server.py: GET /api/info once, then POST /api/predict for every photo or camera frame. */
'use strict';

const LOW_P = 0.5;             // brand confidence below this is flagged "uncertain"
const LIVE_MAX_SIDE = 1280;    // camera frames are scaled down to this before they are sent
const LIVE_MIN_INTERVAL = 66;  // ms between camera requests: at most ~15 per second
const SETTING_KEYS = ['min_conf', 'min_rel', 'max_cars', 'min_model_p', 'flip'];
const CAR_COLOURS = {
  white: '#ffffff', black: '#151515', grey: '#8c8c8c', red: '#d0302f',
  blue: '#2a62c9', green: '#2e8b3d', brown: '#7a4b2a', yellow: '#f2c200',
};

// ---- small helpers ----------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const SVG_NS = 'http://www.w3.org/2000/svg';

/** Build an element. Text always goes in as text nodes: file names and labels are data, not HTML. */
function h(tag, attrs, ...children) { return build(document.createElement(tag), attrs, children); }
function s(tag, attrs, ...children) { return build(document.createElementNS(SVG_NS, tag), attrs, children); }
function build(node, attrs, children) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === '') continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const icon = id => s('svg', {viewBox: '0 0 24 24', 'aria-hidden': 'true'}, s('use', {href: `#${id}`}));

const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`;
const ms = sec => `${Math.round(sec * 1000)} ms`;
const sleep = t => new Promise(resolve => setTimeout(resolve, t));
const area = b => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
const argmax = a => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
/** "Lacetti_gentra" (folder name used in training) -> "Lacetti/Gentra" */
const pretty = name => name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join('/');
const carName = car => (car.car_model ? `${car.brand} ${pretty(car.car_model)}` : car.brand);

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const hgt = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || hgt <= 0) return 0;
  return (w * hgt) / (area(a) + area(b) - w * hgt);
}

/** Most frequent entry; a tie goes to the most recent one. */
function mostCommon(list) {
  const counts = new Map();
  let best = list[list.length - 1], bestN = 0;
  for (let i = list.length - 1; i >= 0; i--) {
    const n = (counts.get(list[i]) || 0) + 1;
    counts.set(list[i], n);
    if (n > bestN) { best = list[i]; bestN = n; }
  }
  return best;
}

function stamp() {
  const d = new Date(), p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

/** Per-browser memory for the theme and the settings; the page works the same without it. */
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private window: not remembered */ }
  },
};

let toastTimer;
function toast(message) {
  const el = $('#toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2200);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h('textarea', {style: 'position:fixed;opacity:0'}, text);
    document.body.append(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
  toast('Copied to clipboard');
}

function setStatus(kind, text) {
  const el = $('#status');
  el.className = `status ${kind}`;
  el.title = text;
  $('#status-text').textContent = text;
}

// ---- state and server calls ---------------------------------------------------------------

const state = {
  info: null,        // /api/info: classes, defaults, device, training history
  settings: {},      // the options sent with every request
  version: 0,        // bumped when the settings change; answers computed with older settings are dropped
  items: [],         // the photos in the strip
  currentId: null,
  queue: [],
  busy: false,
};
let nextId = 1;

async function predict(blob, name, crops) {
  const params = new URLSearchParams(SETTING_KEYS.map(k => [k, state.settings[k]]));
  params.set('crops', crops);
  const body = new FormData();
  body.append('file', blob, name);
  const res = await fetch(`/api/predict?${params}`, {method: 'POST', body});
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail) message = typeof j.detail === 'string' ? j.detail : j.detail.map(d => d.msg).join('; ');
    } catch { /* not JSON */ }
    throw new Error(message);
  }
  return res.json();
}

/** The "with models" line, as in the command line's MODELS: output. */
function modelsLine(cars) {
  if (!cars.some(c => c.car_model)) return null;
  return cars.map(c => `${carName(c)} ${c.colour}`).join('; ');
}

// ---- views --------------------------------------------------------------------------------

const VIEWS = ['photo', 'live', 'model'];

function route() {
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'photo';
  for (const v of VIEWS) {
    $(`#view-${v}`).hidden = v !== name;
    $(`#tab-${v}`).setAttribute('aria-selected', String(v === name));
  }
  if (name !== 'live') stopCamera();
  if (name === 'model') renderModel();
}

function bindTheme() {
  $('#theme-btn').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    store.set('theme', root.dataset.theme);
  });
}

// ---- settings -----------------------------------------------------------------------------

let settingsTimer;

function bindSettings() {
  const panel = $('#settings'), btn = $('#settings-btn');
  const open = show => { panel.hidden = !show; btn.setAttribute('aria-expanded', String(show)); };
  btn.addEventListener('click', () => open(panel.hidden));
  $('#settings-close').addEventListener('click', () => open(false));
  $('#settings-reset').addEventListener('click', () => {
    state.settings = {...state.info.defaults};
    renderSettings();
    settingsChanged();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') open(false); });
  document.addEventListener('pointerdown', e => {
    if (!panel.hidden && !panel.contains(e.target) && !btn.contains(e.target)) open(false);
  });
  for (const key of SETTING_KEYS) {
    const input = $(`#s-${key}`);
    input.addEventListener('input', () => {
      if (!state.info) return;
      state.settings[key] = input.type === 'checkbox' ? input.checked : Number(input.value);
      renderSettingValues();
      clearTimeout(settingsTimer);
      settingsTimer = setTimeout(settingsChanged, 300);   // re-run once the slider stops moving
    });
  }
}

function loadSettings() {
  const defaults = state.info.defaults, saved = store.get('settings', {}) || {};
  const out = {...defaults};
  for (const k of SETTING_KEYS) if (typeof saved[k] === typeof defaults[k]) out[k] = saved[k];
  return out;
}

function renderSettings() {
  for (const key of SETTING_KEYS) {
    const input = $(`#s-${key}`);
    if (input.type === 'checkbox') input.checked = state.settings[key];
    else input.value = state.settings[key];
  }
  renderSettingValues();
}

function renderSettingValues() {
  const st = state.settings;
  $('#o-min_conf').textContent = st.min_conf.toFixed(2);
  $('#o-min_rel').textContent = pct(st.min_rel);
  $('#o-max_cars').textContent = st.max_cars;
  $('#o-min_model_p').textContent = pct(st.min_model_p);
  const changed = SETTING_KEYS.some(k => st[k] !== state.info.defaults[k]);
  $('#settings-btn').classList.toggle('changed', changed);
  $('#settings-btn').title = changed ? 'Detection settings (changed from the defaults)' : 'Detection settings';
}

function settingsChanged() {
  store.set('settings', state.settings);
  state.version++;
  for (const item of state.items) {
    if (item.status === 'done' || item.status === 'error') Object.assign(item, {status: 'idle', result: null, error: null});
  }
  renderStrip();
  const item = currentItem();
  if (item) {
    if (item.status === 'idle') enqueue(item, true);
    renderCurrent();
  }
  live.tracks = [];   // the camera picks the new settings up with its next frame
}

// ---- photos ---------------------------------------------------------------------------------

const currentItem = () => state.items.find(i => i.id === state.currentId) || null;
const isImage = f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|gif|heic|heif)$/i.test(f.name);

function newItem(props) {
  const item = {id: nextId++, status: 'idle', result: null, error: null, thumb: null, ...props};
  state.items.push(item);
  return item;
}

function addFiles(fileList) {
  const files = [...fileList].filter(isImage);
  if (!files.length) {
    toast('No image files found');
    return;
  }
  const added = files.map(file => {
    const item = newItem({name: file.name || 'pasted image.png', blob: file, url: URL.createObjectURL(file)});
    makeThumb(file).then(url => { item.thumb = url || item.url; updateStripItem(item); });
    return item;
  });
  if (location.hash && location.hash !== '#photo') location.hash = '#photo';
  renderStrip();
  added.forEach(item => enqueue(item));
  select(added[0].id);
}

function addSamples(names) {
  for (const name of names) {
    const q = encodeURIComponent(name);
    newItem({name, sample: true, url: `/api/samples/${q}`, thumb: `/api/samples/${q}?thumb=true`});
  }
  renderStrip();
}

/** A small JPEG for the strip, so dozens of 4K phone photos don't all stay decoded in memory. */
async function makeThumb(blob) {
  try {
    const bmp = await createImageBitmap(blob, {resizeHeight: 192, resizeQuality: 'medium'});
    const c = h('canvas', {width: bmp.width, height: bmp.height});
    c.getContext('2d').drawImage(bmp, 0, 0);
    bmp.close?.();
    const out = await new Promise(resolve => c.toBlob(resolve, 'image/jpeg', 0.8));
    return out ? URL.createObjectURL(out) : null;
  } catch {
    return null;   // e.g. HEIC, which the browser can't decode either
  }
}

function enqueue(item, first = false) {
  if (item.status === 'queued' || item.status === 'busy') {
    if (first) prioritise(item);
    return;
  }
  Object.assign(item, {status: 'queued', error: null});
  if (first) state.queue.unshift(item);
  else state.queue.push(item);
  updateStripItem(item);
  updateToolbar();
  pump();
}

function prioritise(item) {
  const i = state.queue.indexOf(item);
  if (i > 0) {
    state.queue.splice(i, 1);
    state.queue.unshift(item);
  }
}

/** Analyse queued photos one at a time (the server runs one photo at a time anyway). */
async function pump() {
  if (state.busy || !state.info) return;
  const item = state.queue.shift();
  if (!item) return;
  state.busy = true;
  item.status = 'busy';
  updateStripItem(item);
  if (item.id === state.currentId) renderCurrent();
  const version = state.version;
  try {
    if (!item.blob) {
      const res = await fetch(item.url);
      if (!res.ok) throw new Error(`Could not load ${item.name} (${res.status})`);
      item.blob = await res.blob();
    }
    const result = await predict(item.blob, item.name, true);
    if (version === state.version) Object.assign(item, {status: 'done', result});
    else item.status = 'idle';   // the settings changed meanwhile, so this answer is out of date
  } catch (e) {
    Object.assign(item, {status: 'error', error: e.message || String(e)});
  }
  state.busy = false;
  updateStripItem(item);
  updateToolbar();
  if (item.id === state.currentId) {
    if (item.status === 'idle') enqueue(item, true);
    renderCurrent();
  }
  pump();
}

function select(id) {
  state.currentId = id;
  for (const btn of $$('.strip-item')) btn.setAttribute('aria-current', String(Number(btn.dataset.id) === id));
  const btn = $(`.strip-item[data-id="${id}"]`), strip = $('#strip');
  if (btn && strip.clientWidth) {   // centre it in the strip; only the strip scrolls, never the page
    const r = btn.getBoundingClientRect(), sr = strip.getBoundingClientRect();
    strip.scrollBy({left: r.left - sr.left - (sr.width - r.width) / 2, behavior: 'smooth'});
  }
  const item = currentItem();
  if (item && (item.status === 'idle' || item.status === 'error')) enqueue(item, true);
  else if (item?.status === 'queued') prioritise(item);
  renderCurrent();
}

function renderCurrent() {
  const item = currentItem();
  $('#photo-empty').hidden = Boolean(item);
  $('#photo-stage').hidden = !item;
  const img = $('#photo');
  if (!item) {
    img.removeAttribute('src');
    renderPhotoResults(null);
    return;
  }
  if (img.dataset.id !== String(item.id)) {
    img.dataset.id = item.id;
    img.alt = item.name;
    img.src = item.url;
  }
  $('#photo-busy').hidden = !(item.status === 'busy' || item.status === 'queued');
  drawBoxes(item.result);
  renderPhotoResults(item);
}

/** Boxes are positioned in percent of the photo, so they stay on the cars at any display size.
    The biggest car (number 1) is added last, so its label stays on top where boxes overlap. */
function drawBoxes(result) {
  const layer = $('#photo-boxes');
  layer.replaceChildren();
  if (!result) return;
  const {width: W, height: H} = result;
  result.cars.forEach((car, i) => {
    const [x1, y1, x2, y2] = car.box;
    const flat = (y2 - y1) / H < 0.05 && y1 / H > 0.05;   // too flat for a label inside, and room above it
    const cls = ['box', car.p < LOW_P && 'low', x1 / W > 0.5 && 'right', flat && 'above'].filter(Boolean).join(' ');
    const style = `left:${(x1 / W) * 100}%;top:${(y1 / H) * 100}%;width:${((x2 - x1) / W) * 100}%;height:${((y2 - y1) / H) * 100}%`;
    const label = h('span', {class: 'box-label'});
    label.fits = [`${i + 1} ${carName(car)}`, `${i + 1} ${car.brand}`, String(i + 1)];   // longest first
    layer.prepend(h('div', {class: cls, style, dataset: {i}, title: `${i + 1}. ${carName(car)}, ${car.colour}, ${pct(car.p)}`,
      onmouseenter: () => highlight(i, true), onmouseleave: () => highlight(i, false)}, label));
  });
  fitLabels();
}

/** Give each box the longest label that fits inside it: "2 Chevrolet Tracker", "2 Chevrolet" or "2". */
function fitLabels() {
  for (const label of $$('#photo-boxes .box-label')) {
    for (const text of label.fits) {
      label.textContent = text;
      if (label.scrollWidth <= label.clientWidth) break;
    }
  }
}

function highlight(i, on) {
  for (const el of $$(`#photo-boxes .box[data-i="${i}"], #photo-results .car[data-i="${i}"]`)) el.classList.toggle('hot', on);
}

function renderPhotoResults(item) {
  const panel = $('#photo-results');
  if (!item) {
    panel.replaceChildren(h('p', {class: 'placeholder'},
      'Every car in the photo is listed here with its brand, Chevrolet model, colour and how sure the model is.'));
    return;
  }
  const parts = [h('div', {class: 'file-name', title: item.name}, item.name)];
  if (item.status === 'error') {
    parts.push(errorBox(item.error));
  } else if (!item.result) {
    parts.push(h('p', {class: 'placeholder'}, h('span', {class: 'spinner'}), 'Analysing…'));
  } else {
    const r = item.result, n = r.cars.length, t = r.seconds;
    const stages = [['find', t.detect], ['classify', t.classify], ['colour', t.colour]]
      .filter(([, v]) => v != null).map(([k, v]) => `${k} ${ms(v)}`).join(' · ');
    parts.push(answerBox(r.answer, modelsLine(r.cars)));
    parts.push(h('p', {class: 'timing'},
      `${n} car${n === 1 ? '' : 's'} · ${ms(t.total)} (${stages})${r.options.flip ? ' · mirror averaging' : ''}`));
    if (!n) parts.push(h('p', {class: 'placeholder'}, 'No car found. If a car was missed, lower “Detector confidence” in the settings.'));
    parts.push(h('ol', {class: 'cars'}, r.cars.map((car, i) => carCard(car, i, true))));
  }
  panel.replaceChildren(...parts);
}

function errorBox(message) {
  return h('div', {class: 'error-box', role: 'alert'}, icon('i-warn'), h('span', {}, message));
}

function answerBox(answer, models) {
  return h('div', {class: 'answer-box'},
    h('div', {class: 'answer-label'}, h('span', {}, 'Answer'),
      h('button', {class: 'btn tiny ghost', type: 'button', title: 'Copy the answer line', onclick: () => copyText(answer)}, 'Copy')),
    h('div', {class: 'answer'}, answer),
    models && h('div', {class: 'answer-models'}, `With models: ${models}`));
}

/** One car: the crop the classifier saw, brand and confidence, colour, runner-up brands. */
function carCard(car, i, interactive) {
  const low = car.p < LOW_P;
  const [x1, y1, x2, y2] = car.box;
  const next = car.top_brands.slice(1).filter(t => t.p >= 0.005).map(t => `${t.name} ${pct(t.p)}`).join(' · ');
  const unsure = !car.car_model && car.top_models.length ? car.top_models[0] : null;
  return h('li', {class: `car${low ? ' low' : ''}${car.crop ? '' : ' no-crop'}`, dataset: {i},
    onmouseenter: interactive && (() => highlight(i, true)), onmouseleave: interactive && (() => highlight(i, false))},
  car.crop && h('img', {class: 'car-crop', src: car.crop, alt: `Car ${i + 1} as the classifier saw it`}),
  h('div', {},
    h('div', {class: 'car-head'},
      h('span', {class: 'num'}, i + 1),
      h('span', {class: 'car-brand'}, car.brand),
      car.car_model && h('span', {class: 'car-model'}, `${pretty(car.car_model)} · ${pct(car.p_model)}`),
      low && h('span', {class: 'tag', title: `Brand confidence below ${pct(LOW_P)}`}, icon('i-warn'), 'uncertain')),
    meter(car.p),
    h('div', {class: 'facts'},
      h('span', {}, h('span', {class: 'swatch', style: `--c:${CAR_COLOURS[car.colour] || '#888'}`}), car.colour),
      h('span', {}, `detector ${car.det_conf.toFixed(2)}`),
      h('span', {}, `${Math.round(x2 - x1)}×${Math.round(y2 - y1)} px`)),
    next && h('div', {class: 'alts'}, `Next: ${next}`),
    unsure && h('div', {class: 'alts'}, `Model not named: best guess ${pretty(unsure.name)} at ${pct(unsure.p)} is below the threshold`)));
}

function meter(p) {
  return h('div', {class: 'meter'},
    h('div', {class: 'meter-track', role: 'meter', 'aria-label': 'Brand confidence',
      'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(p * 100)},
    h('div', {class: 'meter-fill', style: `width:${(p * 100).toFixed(1)}%`})),
    h('span', {class: 'meter-val'}, pct(p)));
}

// ---- the photo strip ------------------------------------------------------------------------

function renderStrip() {
  $('#strip-wrap').hidden = !state.items.length;
  $('#strip-title').textContent = `Photos (${state.items.length})`;
  $('#strip').replaceChildren(...state.items.map(item => h('li', {}, stripButton(item))));
  updateToolbar();
}

function stripButton(item) {
  const btn = h('button', {class: 'strip-item', type: 'button', title: item.name, dataset: {id: item.id},
    'aria-current': String(item.id === state.currentId), onclick: () => select(item.id)},
  h('span', {class: 'strip-thumb-wrap'},
    h('img', {class: 'strip-thumb', alt: '', loading: 'lazy', decoding: 'async'}),
    h('span', {class: 'strip-badge'})),
  h('span', {class: 'strip-name'}, item.name),
  h('span', {class: 'strip-answer'}));
  fillStripButton(btn, item);
  return btn;
}

function updateStripItem(item) {
  const btn = $(`.strip-item[data-id="${item.id}"]`);
  if (btn) fillStripButton(btn, item);
}

function fillStripButton(btn, item) {
  const img = $('.strip-thumb', btn);
  if (item.thumb && img.getAttribute('src') !== item.thumb) img.src = item.thumb;
  const badge = $('.strip-badge', btn);
  badge.className = `strip-badge ${item.status}`;
  badge.hidden = item.status === 'idle';
  badge.title = item.status === 'done' ? `${item.result.cars.length} car(s)` : item.status;
  badge.replaceChildren(item.status === 'done' ? String(item.result.cars.length)
    : item.status === 'error' ? '!' : h('span', {class: 'spinner'}));
  $('.strip-answer', btn).textContent = {
    done: item.result?.answer, error: 'Could not analyse', idle: 'Not analysed yet', queued: 'Waiting…', busy: 'Analysing…',
  }[item.status];
}

function updateToolbar() {
  $('#analyse-all').disabled = !state.items.some(i => i.status === 'idle' || i.status === 'error');
  $('#export-csv').disabled = !state.items.some(i => i.status === 'done');
}

function exportCsv() {
  const rows = [['photo', 'car', 'brand', 'brand_p', 'runner_up', 'runner_up_p', 'chevrolet_model', 'model_p',
    'colour', 'detector_conf', 'x1', 'y1', 'x2', 'y2', 'photo_answer']];
  for (const item of state.items) {
    if (item.status !== 'done') continue;
    const r = item.result;
    if (!r.cars.length) rows.push([item.name, 0, '', '', '', '', '', '', '', '', '', '', '', '', r.answer]);
    r.cars.forEach((c, i) => {
      const second = c.top_brands[1];
      rows.push([item.name, i + 1, c.brand, c.p.toFixed(4), second ? second.name : '', second ? second.p.toFixed(4) : '',
        c.car_model ? pretty(c.car_model) : '', c.car_model ? c.p_model.toFixed(4) : '', c.colour, c.det_conf.toFixed(4),
        ...c.box.map(v => Math.round(v)), r.answer]);
    });
  }
  const text = rows.map(row => row.map(csvCell).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([`﻿${text}`], {type: 'text/csv;charset=utf-8'}));   // BOM: Excel reads UTF-8
  const a = h('a', {href: url, download: `car-predictions_${stamp()}.csv`});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast(`Exported ${rows.length - 1} row${rows.length === 2 ? '' : 's'}`);
}

function csvCell(value) {
  let t = String(value);
  if (typeof value === 'string' && /^[=+\-@]/.test(t)) t = `'${t}`;   // a spreadsheet must not run a file name as a formula
  return /[",\r\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

function bindPhoto() {
  for (const input of $$('.file-input')) {
    input.addEventListener('change', () => {
      if (input.files.length) addFiles(input.files);
      input.value = '';   // choosing the same file again should work too
    });
  }
  $('#analyse-all').addEventListener('click', () => {
    state.items.filter(i => i.status === 'idle' || i.status === 'error').forEach(i => enqueue(i));
  });
  $('#export-csv').addEventListener('click', exportCsv);
  $('#photo').addEventListener('load', fitLabels);   // box sizes are known once the photo is laid out
  new ResizeObserver(fitLabels).observe($('#photo-boxes'));
  document.addEventListener('keydown', e => {
    if ($('#view-photo').hidden || e.altKey || e.ctrlKey || e.metaKey) return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
    const j = state.items.findIndex(it => it.id === state.currentId) + (e.key === 'ArrowRight' ? 1 : -1);
    if (j >= 0 && j < state.items.length) {
      e.preventDefault();
      select(state.items[j].id);
    }
  });
}

function bindDrop() {
  const overlay = $('#drop-overlay');
  const hasFiles = e => [...(e.dataTransfer?.types || [])].includes('Files');
  let depth = 0;
  window.addEventListener('dragenter', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    overlay.hidden = false;
  });
  window.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('dragleave', e => {
    if (hasFiles(e) && --depth <= 0) {
      depth = 0;
      overlay.hidden = true;
    }
  });
  window.addEventListener('drop', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    addFiles(e.dataTransfer.files);
  });
  window.addEventListener('paste', e => {
    const files = [...(e.clipboardData?.files || [])];
    if (!files.length) return;
    e.preventDefault();
    addFiles(files);
  });
}

function bindLightbox() {
  const box = $('#lightbox');
  box.addEventListener('click', () => { box.hidden = true; });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') box.hidden = true; });
}

function openLightbox(src, alt) {
  const img = $('#lightbox img');
  img.src = src;
  img.alt = alt;
  $('#lightbox').hidden = false;
}

// ---- live camera ------------------------------------------------------------------------------

const live = {
  stream: null, running: false, session: 0,
  tracks: [],   // cars followed from frame to frame (for smoothing)
  views: [],    // what is drawn: one entry per tracked car, biggest first
  frame: 0, stamps: [], rtt: 0, server: 0, error: null, raf: 0,
  grab: document.createElement('canvas'),
};

function bindLive() {
  $('#cam-toggle').addEventListener('click', () => (live.running ? stopCamera() : startCamera()));
  $('#cam-select').addEventListener('change', e => startCamera(e.target.value));
  $('#cam-snap').addEventListener('click', snapshot);
  $('#cam-smooth').addEventListener('change', () => { live.tracks = []; });
  if (!navigator.mediaDevices?.getUserMedia) {
    $('#cam-toggle').disabled = true;
    $('#cam-message').textContent = window.isSecureContext
      ? 'This browser gives no access to a camera.'
      : 'Browsers allow the camera only on localhost or HTTPS. Open the page on the computer that runs the server '
        + '(http://127.0.0.1:8765) or through its public link (--public), or use “Take photo” in the Photos tab on a phone.';
  }
}

async function startCamera(deviceId) {
  stopCamera();
  const session = ++live.session;
  const size = {width: {ideal: 1280}, height: {ideal: 720}};
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: deviceId ? {deviceId: {exact: deviceId}, ...size} : {facingMode: {ideal: 'environment'}, ...size},
    });
  } catch (e) {
    $('#cam-message').textContent = e.name === 'NotAllowedError'
      ? 'Camera access was blocked. Allow it with the camera icon in the address bar, then try again.'
      : e.name === 'NotFoundError' ? 'No camera was found on this device.' : `Could not open the camera: ${e.message}`;
    return;
  }
  if (session !== live.session) {   // stopped or switched while the browser was asking for permission
    stream.getTracks().forEach(t => t.stop());
    return;
  }
  Object.assign(live, {stream, running: true, tracks: [], views: [], frame: 0, stamps: [], error: null});
  const video = $('#video');
  video.srcObject = stream;
  try { await video.play(); } catch { /* a muted video may always play; nothing to do */ }
  $('#cam-empty').hidden = true;
  $('#cam-stage').hidden = false;
  $('#cam-toggle span').textContent = 'Stop camera';
  $('#cam-snap').disabled = false;
  fillCameraList(stream);
  renderLive();
  liveLoop(session);
  drawLoop(session);
}

function stopCamera() {
  live.session++;   // ends the request loop and the drawing loop
  if (live.stream) live.stream.getTracks().forEach(t => t.stop());
  Object.assign(live, {stream: null, running: false, tracks: [], views: []});
  cancelAnimationFrame(live.raf);
  $('#video').srcObject = null;
  $('#cam-stage').hidden = true;
  $('#cam-empty').hidden = false;
  $('#cam-toggle span').textContent = 'Start camera';
  $('#cam-snap').disabled = true;
  $('#cam-stats').textContent = '';
  renderLive();
}

async function fillCameraList(stream) {
  try {
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const active = stream.getVideoTracks()[0]?.getSettings().deviceId;
    const select = $('#cam-select');
    select.replaceChildren(...cams.map((d, i) =>
      h('option', {value: d.deviceId, selected: d.deviceId === active}, d.label || `Camera ${i + 1}`)));
    select.hidden = cams.length < 2;
  } catch { /* the camera list is optional */ }
}

function grabFrame(video, maxSide, canvas) {
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob(
    blob => (blob ? resolve({blob, scale}) : reject(new Error('Could not capture a camera frame'))), 'image/jpeg', 0.88));
}

/** Send a frame, wait for the answer, send the next one: never more than one request in flight. */
async function liveLoop(session) {
  const video = $('#video');
  while (session === live.session) {
    const t0 = performance.now();
    if (video.readyState >= 2 && video.videoWidth) {
      try {
        const {blob, scale} = await grabFrame(video, LIVE_MAX_SIDE, live.grab);
        const res = await predict(blob, 'frame.jpg', false);
        if (session !== live.session) return;
        live.frame++;
        updateTracks(res.cars.map(c => ({...c, box: c.box.map(v => v / scale)})));
        Object.assign(live, {rtt: performance.now() - t0, server: res.seconds.total * 1000, error: null});
        live.stamps.push(performance.now());
        if (live.stamps.length > 12) live.stamps.shift();
      } catch (e) {
        live.error = e.message || String(e);
        await sleep(700);
      }
      if (session === live.session) renderLive();
    }
    await sleep(Math.max(10, LIVE_MIN_INTERVAL - (performance.now() - t0)));
  }
}

/** Follow each car across frames (by box overlap) and average its brand probabilities over time,
    so one blurry frame doesn't flip the answer. Without smoothing every frame stands alone. */
function updateTracks(cars) {
  const smooth = $('#cam-smooth').checked;
  const alpha = smooth ? 0.35 : 1;   // weight of the newest frame
  const pairs = [];
  live.tracks.forEach((t, ti) => cars.forEach((c, ci) => {
    const overlap = iou(t.box, c.box);
    if (overlap > 0.3) pairs.push([overlap, ti, ci]);
  }));
  pairs.sort((a, b) => b[0] - a[0]);
  const usedT = new Set(), usedC = new Set();
  for (const [, ti, ci] of pairs) {
    if (usedT.has(ti) || usedC.has(ci)) continue;
    usedT.add(ti);
    usedC.add(ci);
    const t = live.tracks[ti], c = cars[ci];
    const mix = (old, now) => old.map((v, k) => v + alpha * (now[k] - v));
    t.box = mix(t.box, c.box);
    t.probs = mix(t.probs, c.brand_probs);
    if (c.model_probs) t.modelProbs = t.modelProbs ? mix(t.modelProbs, c.model_probs) : c.model_probs;
    t.colours = smooth ? [...t.colours.slice(-6), c.colour] : [c.colour];
    t.det = c.det_conf;
    t.seen = live.frame;
  }
  cars.forEach((c, ci) => {
    if (!usedC.has(ci)) {
      live.tracks.push({box: c.box, probs: c.brand_probs, modelProbs: c.model_probs, colours: [c.colour], det: c.det_conf, seen: live.frame});
    }
  });
  const patience = smooth ? 2 : 0;   // frames a car may be missed before it is dropped
  live.tracks = live.tracks.filter(t => live.frame - t.seen <= patience);
}

/** A track in the same shape as a car from the server, so carCard() can show it. */
function trackView(t) {
  const {brands, car_models: models, car_model_brand: modelBrand} = state.info;
  const order = t.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const view = {
    box: t.box, det_conf: t.det, colour: mostCommon(t.colours),
    brand: brands[order[0][1]], p: order[0][0],
    top_brands: order.slice(0, 3).map(([p, i]) => ({name: brands[i], p})),
    car_model: null, p_model: 0, top_models: [],
  };
  if (view.brand === modelBrand && t.modelProbs) {
    const mi = argmax(t.modelProbs);
    view.top_models = [{name: models[mi], p: t.modelProbs[mi]}];
    if (t.modelProbs[mi] >= state.settings.min_model_p) Object.assign(view, {car_model: models[mi], p_model: t.modelProbs[mi]});
  }
  return view;
}

function renderLive() {
  live.views = live.running && state.info ? live.tracks.map(trackView).sort((a, b) => area(b.box) - area(a.box)) : [];
  const views = live.views, models = modelsLine(views);
  $('#live-answer').textContent = !live.running ? 'Camera is off'
    : views.length ? views.map(v => `${v.brand} ${v.colour}`).join('; ') : live.frame ? '(no car found)' : 'Starting…';
  $('#live-models').hidden = !models;
  $('#live-models').textContent = models ? `With models: ${models}` : '';
  $('#live-cars').replaceChildren(...views.map((v, i) => carCard(v, i, false)));
  const n = live.stamps.length;
  const fps = n > 1 ? (n - 1) / ((live.stamps[n - 1] - live.stamps[0]) / 1000) : 0;
  $('#cam-stats').textContent = !live.running ? ''
    : live.error ? `Error: ${live.error}`
      : n > 1 ? `${fps.toFixed(1)} frames/s · ${Math.round(live.rtt)} ms per frame (models ${Math.round(live.server)} ms)` : '';
}

/** Redraw the boxes on every screen refresh; they change whenever an answer arrives. */
function drawLoop(session) {
  const video = $('#video'), canvas = $('#cam-overlay'), ctx = canvas.getContext('2d');
  const step = () => {
    if (session !== live.session) return;
    if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const k = canvas.width / Math.max(1, canvas.clientWidth);   // canvas pixels per screen pixel
    const css = getComputedStyle(document.documentElement);
    const colours = {accent: css.getPropertyValue('--accent').trim(), warn: css.getPropertyValue('--warn').trim()};
    for (let i = live.views.length - 1; i >= 0; i--) drawLiveBox(ctx, live.views[i], i, k, colours);   // biggest on top
    live.raf = requestAnimationFrame(step);
  };
  live.raf = requestAnimationFrame(step);
}

function drawLiveBox(ctx, v, i, k, colours) {
  const [x1, y1, x2, y2] = v.box, low = v.p < LOW_P, colour = low ? colours.warn : colours.accent;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.lineWidth = 4.5 * k;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2.5 * k;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  const text = `${i + 1} ${carName(v)} · ${v.colour} · ${pct(v.p)}`;
  ctx.font = `600 ${12.5 * k}px system-ui, -apple-system, "Segoe UI", sans-serif`;
  const pad = 7 * k, hgt = 22 * k, w = ctx.measureText(text).width + 2 * pad;
  const x = Math.max(0, Math.min(x1 - 1.25 * k, ctx.canvas.width - w));
  const y = y1 - hgt - 3 * k >= 0 ? y1 - hgt - 3 * k : y1 + 3 * k;
  ctx.fillStyle = colour;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, hgt, 6 * k);
  else ctx.rect(x, y, w, hgt);
  ctx.fill();
  ctx.fillStyle = low ? '#0b0b0b' : '#ffffff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pad, y + hgt / 2 + 0.5 * k);
}

/** Send the current frame at full resolution to the Photos tab for the complete result. */
async function snapshot() {
  const video = $('#video');
  if (!video.videoWidth) return;
  const {blob} = await grabFrame(video, 4096, document.createElement('canvas'));
  const file = new File([blob], `camera_${stamp()}.jpg`, {type: 'image/jpeg'});
  location.hash = '#photo';
  addFiles([file]);
}

// ---- about the model ----------------------------------------------------------------------------

const ACC_SERIES = [   // fixed order = fixed colours (the palette's order is what keeps neighbours distinguishable)
  {key: 'train_acc', label: 'Training (Car-1000 + ours)', color: '--s1'},
  {key: 'val_acc', label: 'Car-1000 validation', color: '--s2'},
  {key: 'own_val_brand_acc', label: 'Our photos: brand', color: '--s3'},
  {key: 'own_val_car_model_acc', label: 'Our photos: Chevrolet model', color: '--s4'},
];
const TABLE_COLUMNS = [
  ['epoch', 'Epoch', v => v], ['train_loss', 'Train loss', v => v.toFixed(3)], ['train_acc', 'Train', v => pct(v, 1)],
  ['val_acc', 'Car-1000 val', v => pct(v, 1)], ['val_macro', 'Balanced', v => pct(v, 1)],
  ['own_val_brand_acc', 'Ours: brand', v => pct(v, 1)], ['own_val_car_model_acc', 'Ours: model', v => pct(v, 1)],
  ['seconds', 'Seconds', v => v.toFixed(0)],
];
let modelRendered = false;

/** train.py keeps the epoch with the best balanced (macro) validation accuracy; the first one on ties. */
const savedEpoch = history => history.reduce((best, row) => (row.val_macro > best.val_macro ? row : best), history[0]);

function renderModel() {
  if (modelRendered || !state.info) return;
  modelRendered = true;
  const {brands, car_models: models, colours, history, figures, device, weights} = state.info;
  $('#n-brands').textContent = brands.length;
  $('#n-models').textContent = models.length;
  $('#n-colours').textContent = colours.length;
  renderTiles(history, brands, models);
  renderCharts(history);
  renderClasses();
  renderFigures(figures);
  $('#model-foot').textContent = `Weights: ${weights} · running on ${device}.`;
}

function renderTiles(history, brands, models) {
  if (!history.length) {
    $('#tiles-note').textContent = 'There is no history.csv next to the weights, so there are no training numbers to show.';
    return;
  }
  const best = savedEpoch(history);
  const tiles = [
    ['Car-1000 validation', best.val_acc, `brand accuracy over ${brands.length} brands`],
    ['Balanced accuracy', best.val_macro, 'mean of the per-brand accuracies'],
    ['Our Chevrolet photos', best.own_val_brand_acc, 'brand accuracy on held-out photos'],
    ['Chevrolet model', best.own_val_car_model_acc, `${models.length} models, held-out photos`],
  ].filter(([, v]) => Number.isFinite(v));
  $('#tiles').replaceChildren(...tiles.map(([label, v, note]) => h('div', {class: 'card tile'},
    h('div', {class: 'tile-label'}, label), h('div', {class: 'tile-value'}, pct(v, 1)), h('div', {class: 'tile-note'}, note))));
  $('#tiles-note').textContent = `Validation results of the saved checkpoint: epoch ${best.epoch} of ${history.length}, the one with `
    + 'the best balanced accuracy. Test-set results are in the confusion matrices below.';
}

function renderCharts(history) {
  if (!history.length) {
    for (const card of $$('.chart-card')) card.hidden = true;
    return;
  }
  const saved = savedEpoch(history).epoch;
  const acc = ACC_SERIES.filter(se => history.every(r => Number.isFinite(r[se.key])));
  $('#acc-legend').replaceChildren(...acc.map(se =>
    h('span', {}, h('span', {class: 'legend-key', style: `--c: var(${se.color})`}), se.label)));
  lineChart($('#acc-chart'), {rows: history, series: acc, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1],
    fmt: v => pct(v), tipFmt: v => pct(v, 1), mark: saved});
  const lossMax = Math.ceil(Math.max(...history.map(r => r.train_loss)));
  lineChart($('#loss-chart'), {rows: history, series: [{key: 'train_loss', label: 'Training loss', color: '--s1'}],
    yMax: lossMax, yTicks: Array.from({length: lossMax + 1}, (_, i) => i), fmt: v => String(v), tipFmt: v => v.toFixed(3), mark: saved});
  const cols = TABLE_COLUMNS.filter(([key]) => history.every(r => Number.isFinite(r[key])));
  $('#acc-table').replaceChildren(h('table', {class: 'data-table'},
    h('thead', {}, h('tr', {}, cols.map(([, label]) => h('th', {scope: 'col'}, label)))),
    h('tbody', {}, history.map(r => h('tr', {class: r.epoch === saved ? 'saved' : null, title: r.epoch === saved ? 'saved checkpoint' : null},
      cols.map(([key, , f]) => h('td', {}, f(r[key]))))))));
}

/** Line chart in SVG: one y-axis, 2px lines, a crosshair tooltip that lists every series. */
function lineChart(root, opt) {
  const draw = () => {
    const W = Math.round(root.clientWidth);
    if (!W) return;
    const H = 250, m = {l: 44, r: 14, t: 24, b: 34};
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = opt.rows.map(r => r.epoch), x0 = xs[0], xn = xs[xs.length - 1];
    const X = v => m.l + (xn > x0 ? ((v - x0) / (xn - x0)) * iw : iw / 2);
    const Y = v => m.t + ih - (v / opt.yMax) * ih;
    const svg = s('svg', {class: 'chart-svg', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', tabindex: 0,
      'aria-label': `${opt.series.map(se => se.label).join(', ')} per epoch. Arrow keys read the values.`});

    for (const t of opt.yTicks) {
      svg.append(s('line', {class: t === 0 ? 'axis' : 'grid', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t)}));
      svg.append(s('text', {class: 'tick', x: m.l - 8, y: Y(t), 'text-anchor': 'end', 'dominant-baseline': 'middle'}, opt.fmt(t)));
    }
    const every = iw / xs.length < 26 ? 2 : 1;   // fewer epoch labels on narrow screens
    xs.forEach((v, i) => {
      if (i % every === 0 || i === xs.length - 1) svg.append(s('text', {class: 'tick', x: X(v), y: H - m.b + 18, 'text-anchor': 'middle'}, v));
    });
    svg.append(s('text', {class: 'tick', x: W - m.r, y: H - 1, 'text-anchor': 'end'}, 'epoch'));
    if (opt.mark != null) {
      const mx = X(opt.mark);
      svg.append(s('line', {class: 'mark', x1: mx, x2: mx, y1: m.t - 4, y2: m.t + ih}));
      svg.append(s('text', {class: 'mark-label', x: mx, y: m.t - 9, 'text-anchor': mx + 55 > W ? 'end' : 'middle'}, 'saved checkpoint'));
    }
    for (const se of opt.series) {
      const d = opt.rows.map((r, i) => `${i ? 'L' : 'M'}${X(r.epoch).toFixed(1)},${Y(r[se.key]).toFixed(1)}`).join(' ');
      svg.append(s('path', {class: 'line', d, style: `stroke: var(${se.color})`}));
    }
    const last = opt.rows[opt.rows.length - 1];
    for (const se of opt.series) svg.append(s('circle', {class: 'dot', cx: X(last.epoch), cy: Y(last[se.key]), r: 4, style: `fill: var(${se.color})`}));

    const cross = s('line', {class: 'crosshair', y1: m.t, y2: m.t + ih, visibility: 'hidden'});
    const dots = opt.series.map(se => s('circle', {class: 'dot', r: 4.5, style: `fill: var(${se.color})`, visibility: 'hidden'}));
    const hit = s('rect', {x: m.l - 12, y: 0, width: iw + 24, height: H, fill: 'transparent'});
    svg.append(cross, ...dots, hit);
    const tip = h('div', {class: 'tooltip', hidden: true});
    let index = null;
    const show = i => {
      index = i;
      const r = opt.rows[i], x = X(r.epoch);
      cross.setAttribute('x1', x);
      cross.setAttribute('x2', x);
      cross.setAttribute('visibility', 'visible');
      dots.forEach((c, k) => {
        c.setAttribute('cx', x);
        c.setAttribute('cy', Y(r[opt.series[k].key]));
        c.setAttribute('visibility', 'visible');
      });
      tip.replaceChildren(h('div', {class: 'tooltip-title'}, `Epoch ${r.epoch}${r.epoch === opt.mark ? ' · saved checkpoint' : ''}`),
        ...opt.series.map(se => h('div', {class: 'tooltip-row'},
          h('span', {class: 'legend-key', style: `--c: var(${se.color})`}), h('strong', {}, opt.tipFmt(r[se.key])), h('span', {}, se.label))));
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = `${x + 14 + tw > W ? Math.max(0, x - 14 - tw) : x + 14}px`;
      tip.style.top = `${m.t}px`;
    };
    const hide = () => {
      index = null;
      cross.setAttribute('visibility', 'hidden');
      dots.forEach(c => c.setAttribute('visibility', 'hidden'));
      tip.hidden = true;
    };
    const nearest = clientX => {
      const px = clientX - svg.getBoundingClientRect().left;
      return xs.reduce((best, v, i) => (Math.abs(X(v) - px) < Math.abs(X(xs[best]) - px) ? i : best), 0);
    };
    hit.addEventListener('pointermove', e => show(nearest(e.clientX)));
    hit.addEventListener('pointerleave', hide);
    svg.addEventListener('keydown', e => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const step = e.key === 'ArrowRight' ? 1 : -1;
      show(Math.max(0, Math.min(xs.length - 1, index == null ? xs.length - 1 : index + step)));
    });
    svg.addEventListener('blur', hide);
    root.replaceChildren(svg, tip);
  };
  draw();
  let width = root.clientWidth;
  new ResizeObserver(() => {
    if (root.clientWidth !== width) {
      width = root.clientWidth;
      draw();
    }
  }).observe(root);
}

function renderClasses() {
  const {brands, car_models: models, colours, car_model_brand: modelBrand} = state.info;
  const group = (title, chips) => h('div', {class: 'class-group'}, h('h4', {}, title), h('div', {class: 'chips'}, chips));
  $('#classes').replaceChildren(
    group(`Brands (${brands.length})`, brands.map(b => h('span', {class: 'chip'}, b))),
    group(`${modelBrand} models (${models.length}), named only for cars recognised as ${modelBrand}`,
      models.map(m => h('span', {class: 'chip'}, pretty(m)))),
    group(`Colours (${colours.length})`, colours.map(c =>
      h('span', {class: 'chip'}, h('span', {class: 'swatch', style: `--c:${CAR_COLOURS[c] || '#888'}`}), c))));
}

function renderFigures(figures) {
  const sets = {car1000_val: 'Car-1000 validation', car1000_test: 'Car-1000 test', uzbek_val: 'Our photos, validation'};
  const caption = name => {
    const base = name.replace(/\.png$/, '').replace(/^confusion_/, '');
    const what = base.endsWith('_car_model') ? 'Chevrolet model' : 'brand';
    const set = base.replace(/_car_model$/, '');
    return `${sets[set] || set} · ${what}`;
  };
  $('#figures').replaceChildren(...figures.map(name => {
    const src = `/api/figures/${encodeURIComponent(name)}`, text = caption(name);
    return h('figure', {},
      h('button', {type: 'button', title: 'Enlarge', onclick: () => openLightbox(src, text)}, h('img', {src, alt: text, loading: 'lazy'})),
      h('figcaption', {}, text));
  }));
}

function bindTables() {
  for (const btn of $$('[data-table-toggle]')) {
    btn.addEventListener('click', () => {
      const table = $(`#${btn.dataset.tableToggle}`);
      table.hidden = !table.hidden;
      btn.textContent = table.hidden ? 'Show table' : 'Hide table';
      btn.setAttribute('aria-expanded', String(!table.hidden));
    });
  }
}

// ---- start --------------------------------------------------------------------------------------

async function init() {
  bindTheme();
  bindSettings();
  bindPhoto();
  bindLive();
  bindDrop();
  bindLightbox();
  bindTables();
  window.addEventListener('hashchange', route);
  renderPhotoResults(null);
  route();
  try {
    const res = await fetch('/api/info');
    if (!res.ok) throw new Error(res.statusText);
    state.info = await res.json();
  } catch {
    setStatus('bad', 'Server not reachable');
    return;
  }
  const gpu = state.info.device.replace(/^NVIDIA (GeForce )?/, '');
  setStatus('ok', state.info.device === 'CPU' ? 'Ready · CPU' : `Ready · ${gpu}`);
  state.settings = loadSettings();
  renderSettings();
  if (state.info.samples.length) addSamples(state.info.samples);
  const wanted = new URLSearchParams(location.search).get('sample');
  const sample = wanted && state.items.find(i => i.name === wanted);
  if (sample) select(sample.id);
  route();
  pump();   // photos dropped before the server answered
}

init();
