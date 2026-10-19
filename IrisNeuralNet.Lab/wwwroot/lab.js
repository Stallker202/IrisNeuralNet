// Страница лаборатории. Все вычисления и вся отрисовка — на сервере: здесь только
// контролы, опрос состояния, вставка готовых SVG, холст для цифр и экспорт PNG.

const POLL_MS = 200;
const HIDDEN_POLL_MS = 1000;
const LIVE_VIEW_MS = 250;
const LIVE_DIGIT_MS = 500;
const SLIDER_MAX = 1000;
const LR_MIN_LOG = -4;
const LR_DECADES = 3;
const PAD_SIDE = 280;
const STROKE_WIDTH = 22;
const MNIST_SIDE = 28;
const MAX_UPLOAD_SIDE = 512;

const $ = (id) => document.getElementById(id);

const ui = {
  status: $('status'),
  dataset: $('dataset'),
  scaling: $('scaling'),
  validation: $('validation'),
  hidden: $('hidden'),
  hiddenError: $('hidden-error'),
  optimizer: $('optimizer'),
  lrLabel: $('lr-label'),
  lr: $('lr'),
  epochs: $('epochs'),
  batch: $('batch'),
  early: $('early'),
  patience: $('patience'),
  overlay: $('overlay'),
  start: $('start'),
  stop: $('stop'),
  hint: $('hint'),
  sweepEpochs: $('sweep-epochs'),
  sweepRun: $('sweep-run'),
  sweepCancel: $('sweep-cancel'),
  sweepStatus: $('sweep-status'),
  digits: document.querySelector('.digits'),
  digitsMessage: $('digits-message'),
  pad: $('pad'),
  padClear: $('pad-clear'),
  padOpen: $('pad-open'),
  padSample: $('pad-sample'),
  padFile: $('pad-file'),
  preview: $('preview'),
  verdict: $('verdict'),
  truth: $('truth'),
  bars: $('bars'),
};

const state = {
  catalog: null,
  server: null,
  suppressHints: false,
  dataRequest: 0,
};

// ---------- HTTP ----------

async function api(method, url, body) {
  const options = { method };
  if (body instanceof Uint8Array) {
    options.body = body;
    options.headers = { 'content-type': 'application/octet-stream' };
  } else if (body !== undefined) {
    options.body = JSON.stringify(body);
    options.headers = { 'content-type': 'application/json' };
  }

  const response = await fetch(url, options);
  if (response.status === 204) return null;

  const isJson = (response.headers.get('content-type') ?? '').includes('json');
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new Error(payload?.error ?? `${response.status} ${response.statusText}`);
  return payload;
}

const debounce = (fn, ms) => {
  let timer = 0;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// ---------- Левая панель ----------

const sliderToRate = (value) => 10 ** (LR_MIN_LOG + (value / SLIDER_MAX) * LR_DECADES);
const rateToSlider = (rate) => clamp(Math.round(((Math.log10(rate) - LR_MIN_LOG) / LR_DECADES) * SLIDER_MAX), 0, SLIDER_MAX);
const learningRate = () => sliderToRate(Number(ui.lr.value));
const currentSpec = () => state.catalog.datasets[ui.dataset.selectedIndex];

function updateLearningRateLabel() {
  ui.lrLabel.textContent = `learning rate: ${learningRate().toFixed(4)} (live)`;
}

function fillSelect(select, options) {
  select.replaceChildren(...options.map(({ value, label }) => new Option(label, value)));
}

function setHint(text) {
  if (!state.suppressHints) ui.hint.textContent = text;
}

/** Значения по умолчанию из методички соответствующей лабораторной. */
function applyDefaults(spec) {
  state.suppressHints = true;
  const d = spec.defaults;
  ui.scaling.value = d.scaling;
  ui.validation.disabled = spec.hasPredefinedTestSplit;
  if (!spec.hasPredefinedTestSplit) ui.validation.value = d.validationFraction;
  ui.hidden.value = d.hidden;
  ui.hiddenError.textContent = '';
  ui.optimizer.value = d.optimizer;
  ui.lr.value = rateToSlider(d.learningRate);
  ui.epochs.value = d.epochs;
  ui.batch.value = d.batchSize;
  ui.sweepEpochs.value = d.epochs;
  updateLearningRateLabel();
  ui.hint.textContent = '';
  state.suppressHints = false;
}

function trainRequest() {
  return {
    hidden: ui.hidden.value,
    optimizer: ui.optimizer.value,
    learningRate: learningRate(),
    epochs: Number(ui.epochs.value),
    batchSize: Number(ui.batch.value),
    earlyStoppingPatience: ui.early.checked ? Number(ui.patience.value) : 0,
    keepPrevious: ui.overlay.checked,
  };
}

async function prepareData() {
  const request = ++state.dataRequest;
  try {
    await api('POST', '/api/data', {
      dataset: currentSpec().key,
      scaling: ui.scaling.value,
      validationFraction: Number(ui.validation.value),
      learningRate: learningRate(),
    });
  } catch (error) {
    if (request === state.dataRequest) ui.hint.textContent = error.message;
  }

  views.invalidateAll();
}

const validateArchitecture = debounce(async () => {
  try {
    const result = await api('GET', `/api/architecture?text=${encodeURIComponent(ui.hidden.value)}`);
    ui.hiddenError.textContent = result.valid ? '' : result.error;
  } catch {
    // Проверка — подсказка, не блокер: Start всё равно проверит на сервере.
  }
}, 150);

const sendLearningRate = debounce(() => {
  api('PUT', '/api/train/learning-rate', { value: learningRate() }).catch(() => {});
}, 100);

async function startTraining() {
  ui.hint.textContent = '';
  try {
    await api('POST', '/api/train/start', trainRequest());
    views.invalidateAll();
  } catch (error) {
    ui.hint.textContent = error.message;
  }
}

function wirePanel() {
  ui.dataset.addEventListener('change', () => {
    applyDefaults(currentSpec());
    prepareData();
  });
  ui.scaling.addEventListener('change', prepareData);
  ui.validation.addEventListener('input', debounce(prepareData, 300));
  ui.hidden.addEventListener('input', () => {
    setHint('архитектура изменена — жми Start/Restart');
    validateArchitecture();
  });
  ui.optimizer.addEventListener('change', () => setHint('оптимизатор изменён — жми Start/Restart'));
  ui.lr.addEventListener('input', () => {
    updateLearningRateLabel();
    sendLearningRate();
  });
  ui.early.addEventListener('change', () => { ui.patience.disabled = !ui.early.checked; });
  ui.overlay.addEventListener('change', () => views.invalidate('loss', 'metric'));
  ui.start.addEventListener('click', startTraining);
  ui.stop.addEventListener('click', () => api('POST', '/api/train/stop').catch((error) => { ui.hint.textContent = error.message; }));
}

// ---------- Вкладки ----------

function wireTabs() {
  for (const tab of document.querySelectorAll('.tab')) {
    tab.addEventListener('click', () => {
      for (const other of document.querySelectorAll('.tab')) {
        const active = other === tab;
        other.classList.toggle('active', active);
        other.setAttribute('aria-selected', String(active));
        $(`page-${other.dataset.tab}`).classList.toggle('active', active);
      }

      views.refresh();
      if (tab.dataset.tab === 'digits') digits.refresh(true);
    });
  }
}

// ---------- SVG-виды ----------

/** Вид = контейнер, в который вставляется SVG с сервера, плюс кнопка PNG и подсказка при наведении. */
class ViewSlot {
  constructor(name, element) {
    this.name = name;
    this.element = element;
    this.host = document.createElement('div');
    this.host.className = 'view-host';
    this.crosshair = document.createElement('div');
    this.crosshair.className = 'crosshair';
    this.crosshair.hidden = true;
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'tooltip';
    this.tooltip.hidden = true;
    const png = document.createElement('button');
    png.className = 'png';
    png.type = 'button';
    png.textContent = 'PNG';
    png.title = 'Сохранить как PNG';
    png.addEventListener('click', () => exportPng(this.host.querySelector('svg'), name));
    element.append(this.host, this.crosshair, this.tooltip, png);
    element.addEventListener('mousemove', (event) => this.hover(event));
    element.addEventListener('mouseleave', () => this.hideHover());

    this.dirty = true;
    this.inFlight = false;
    this.size = '';
    this.fetchedAt = 0;
    this.hoverData = null;
  }

  get visible() {
    return this.element.closest('.page').classList.contains('active');
  }

  setSvg(svg) {
    this.host.innerHTML = svg;
    const data = this.host.firstElementChild?.dataset.hover;
    this.hoverData = data ? JSON.parse(data) : null;
    if (!this.hoverData) this.hideHover();
  }

  /** Перекрестие и значения рядов в ближайшей эпохе. */
  hover(event) {
    const data = this.hoverData;
    const svg = this.host.firstElementChild;
    if (!data || !svg || data.count < 1) return;

    const rect = svg.getBoundingClientRect();
    const scale = rect.width / svg.viewBox.baseVal.width;
    const x = (event.clientX - rect.left) / scale;
    const y = (event.clientY - rect.top) / scale;
    const [left, top, width, height] = data.plot;
    if (x < left || x > left + width || y < top || y > top + height) {
      this.hideHover();
      return;
    }

    const index = data.count === 1 ? 0 : Math.round(((x - left) / (width - 1)) * (data.count - 1));
    const lineX = left + (index * (width - 1)) / Math.max(1, data.count - 1);
    this.crosshair.style.left = `${lineX * scale}px`;
    this.crosshair.style.top = `${top * scale}px`;
    this.crosshair.style.height = `${height * scale}px`;
    this.crosshair.hidden = false;

    const rows = [`<div>эпоха ${index + 1}</div>`];
    for (const series of data.series) {
      const value = series.values[index];
      if (value === undefined || value === null) continue;
      rows.push(`<div><span class="swatch" style="background:${series.color}"></span>${escapeHtml(series.name)}: ${value.toFixed(4)}</div>`);
    }

    this.tooltip.innerHTML = rows.join('');
    this.tooltip.hidden = false;
    const bounds = this.element.getBoundingClientRect();
    const tipWidth = this.tooltip.offsetWidth;
    const pointerX = event.clientX - bounds.left;
    const tipLeft = pointerX + 14 + tipWidth > bounds.width ? pointerX - 14 - tipWidth : pointerX + 14;
    this.tooltip.style.left = `${Math.max(0, tipLeft)}px`;
    this.tooltip.style.top = `${clamp(event.clientY - bounds.top + 14, 0, bounds.height - this.tooltip.offsetHeight)}px`;
  }

  hideHover() {
    this.crosshair.hidden = true;
    this.tooltip.hidden = true;
  }
}

const views = {
  slots: [...document.querySelectorAll('.view[data-view]')].map((element) => new ViewSlot(element.dataset.view, element)),

  invalidate(...names) {
    for (const slot of this.slots) if (names.includes(slot.name)) slot.dirty = true;
    this.refresh();
  },

  invalidateAll() {
    for (const slot of this.slots) slot.dirty = true;
    this.refresh();
  },

  /** Перерисовываются только видимые виды: при изменении данных, размера или (во время обучения) по таймеру. */
  refresh() {
    const now = performance.now();
    const running = state.server?.running ?? false;
    for (const slot of this.slots) {
      if (!slot.visible || slot.inFlight) continue;

      const width = Math.floor(slot.element.clientWidth);
      const height = Math.floor(slot.element.clientHeight);
      if (width < 50 || height < 50) continue;

      const size = `${width}x${height}`;
      const live = running && (slot.name === 'topology' || slot.name === 'boundary') && now - slot.fetchedAt > LIVE_VIEW_MS;
      if (slot.dirty || live || size !== slot.size) this.load(slot, width, height, size);
    }
  },

  async load(slot, width, height, size) {
    slot.inFlight = true;
    slot.dirty = false;
    slot.size = size;
    slot.fetchedAt = performance.now();
    try {
      const overlay = ui.overlay.checked ? 'true' : 'false';
      slot.setSvg(await api('GET', `/api/view/${slot.name}.svg?w=${width}&h=${height}&overlay=${overlay}`));
    } catch {
      slot.dirty = true;
    } finally {
      slot.inFlight = false;
    }
  },
};

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/** SVG → PNG в двойном разрешении: для вставки графика в отчёт. */
async function exportPng(svg, name) {
  if (!svg) return;

  const width = svg.viewBox.baseVal.width;
  const height = svg.viewBox.baseVal.height;
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }));
  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = url;
    });

    const scale = 2;
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const context = canvas.getContext('2d');
    context.scale(scale, scale);
    context.drawImage(image, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${name}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } finally {
    URL.revokeObjectURL(url);
  }
}

// ---------- Sweep ----------

function plural(count, one, few, many) {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return many;
  if (last === 1) return one;
  return last >= 2 && last <= 4 ? few : many;
}

function renderSweep(sweep) {
  ui.sweepRun.textContent = `Запустить sweep (${sweep.total} ${plural(sweep.total, 'конфигурация', 'конфигурации', 'конфигураций')})`;
  ui.sweepRun.disabled = sweep.running || !state.server?.ready;
  ui.sweepCancel.disabled = !sweep.running;
  ui.sweepStatus.textContent = sweep.status;
}

function wireSweep() {
  ui.sweepRun.addEventListener('click', async () => {
    try {
      await api('POST', '/api/sweep/start', { template: trainRequest(), epochs: Number(ui.sweepEpochs.value) });
    } catch (error) {
      ui.sweepStatus.textContent = error.message;
    }
  });
  ui.sweepCancel.addEventListener('click', () => api('POST', '/api/sweep/cancel').catch(() => {}));
}

// ---------- Цифры (MNIST) ----------

const digits = {
  context: ui.pad.getContext('2d', { willReadFrequently: true }),
  drawing: false,
  last: null,
  pending: 0,
  mnist: null,
  truth: '',
  predictedAt: 0,
  revision: -1,
  enabled: false,

  clear() {
    this.context.fillStyle = '#000';
    this.context.fillRect(0, 0, PAD_SIDE, PAD_SIDE);
    this.mnist = null;
    this.truth = '';
    ui.preview.removeAttribute('src');
    ui.verdict.textContent = '';
    ui.truth.textContent = '';
    this.renderBars('');
  },

  point(event) {
    const rect = ui.pad.getBoundingClientRect();
    return { x: ((event.clientX - rect.left) * PAD_SIDE) / rect.width, y: ((event.clientY - rect.top) * PAD_SIDE) / rect.height };
  },

  stroke(from, to) {
    const c = this.context;
    c.strokeStyle = '#fff';
    c.lineWidth = STROKE_WIDTH;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    c.beginPath();
    c.moveTo(from.x, from.y);
    c.lineTo(to.x === from.x && to.y === from.y ? to.x + 0.5 : to.x, to.y);
    c.stroke();
  },

  /** Пересчёт не чаще раза в 120 мс, пока рисуешь: отправляется весь холст 280×280. */
  schedule() {
    if (this.pending) return;
    this.pending = setTimeout(() => {
      this.pending = 0;
      this.predict('Drawing', this.padGray(), PAD_SIDE, PAD_SIDE);
    }, 120);
  },

  padGray() {
    const rgba = this.context.getImageData(0, 0, PAD_SIDE, PAD_SIDE).data;
    const gray = new Uint8Array(PAD_SIDE * PAD_SIDE);
    for (let i = 0; i < gray.length; i++) gray[i] = rgba[i * 4];
    return gray;
  },

  async predict(source, gray, width, height) {
    const size = `bw=${ui.bars.clientWidth}&bh=${ui.bars.clientHeight}`;
    const truth = encodeURIComponent(this.truth);
    try {
      this.show(await api('POST', `/api/digits/predict?source=${source}&w=${width}&h=${height}&${size}&truth=${truth}`, gray));
    } catch (error) {
      ui.verdict.textContent = '';
      ui.truth.textContent = error.message;
    }
  },

  show(result) {
    this.predictedAt = performance.now();
    this.revision = state.server?.revision ?? -1;
    this.mnist = Uint8Array.from(atob(result.pixels), (c) => c.charCodeAt(0));
    this.truth = result.truth;
    ui.preview.src = result.preview;
    ui.verdict.textContent = result.verdict;
    ui.truth.textContent = result.truth;
    this.renderBars(result.bars);
  },

  renderBars(svg) {
    let host = ui.bars.querySelector('.view-host');
    if (!host) {
      host = document.createElement('div');
      host.className = 'view-host';
      ui.bars.append(host);
    }

    host.innerHTML = svg;
  },

  /** Готовые 28×28 крупно на холст — поверх можно дорисовать. */
  drawMnist(pixels) {
    const small = document.createElement('canvas');
    small.width = MNIST_SIDE;
    small.height = MNIST_SIDE;
    const smallContext = small.getContext('2d');
    const image = smallContext.createImageData(MNIST_SIDE, MNIST_SIDE);
    for (let i = 0; i < pixels.length; i++) {
      image.data.set([pixels[i], pixels[i], pixels[i], 255], i * 4);
    }

    smallContext.putImageData(image, 0, 0);
    this.context.imageSmoothingEnabled = false;
    this.context.drawImage(small, 0, 0, PAD_SIDE, PAD_SIDE);
  },

  /** Картинку декодирует браузер (любой формат, который он умеет), на сервер уходят оттенки серого. */
  async open(file) {
    try {
      const bitmap = await createImageBitmap(file);
      const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d');
      context.fillStyle = '#fff'; // прозрачность — на белом фоне, как картинка на бумаге
      context.fillRect(0, 0, width, height);
      context.drawImage(bitmap, 0, 0, width, height);
      const rgba = context.getImageData(0, 0, width, height).data;
      const gray = new Uint8Array(width * height);
      for (let i = 0; i < gray.length; i++) {
        gray[i] = Math.round(0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]);
      }

      this.truth = `файл: ${file.name}`;
      await this.predict('Image', gray, width, height);
      if (this.mnist) this.drawMnist(this.mnist);
    } catch (error) {
      ui.truth.textContent = `Не удалось открыть изображение: ${error.message}`;
    }
  },

  async sample() {
    try {
      this.show(await api('GET', `/api/digits/sample?bw=${ui.bars.clientWidth}&bh=${ui.bars.clientHeight}`));
      this.drawMnist(this.mnist);
    } catch (error) {
      ui.truth.textContent = error.message;
    }
  },

  /** Пока сеть учится, вероятности для текущей цифры пересчитываются вживую. */
  refresh(force = false) {
    if (!this.enabled || !this.mnist || !$('page-digits').classList.contains('active')) return;

    const server = state.server;
    const due = server?.running ? performance.now() - this.predictedAt > LIVE_DIGIT_MS : server?.revision !== this.revision;
    if (force || due) this.predict('Mnist', this.mnist, MNIST_SIDE, MNIST_SIDE);
  },

  setEnabled(enabled) {
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    ui.digits.classList.toggle('disabled', !enabled);
    ui.digitsMessage.textContent = enabled
      ? 'Рисуй цифру белым по чёрному (зажми кнопку мыши) — предобработка как в MNIST: рамка 20×20 и центр масс в центре 28×28.'
      : 'Вкладка для MNIST: выбери датасет MNIST в левой панели и нажми Start.';
    if (!enabled) this.clear();
  },

  wire() {
    ui.pad.addEventListener('pointerdown', (event) => {
      if (!this.enabled) return;
      ui.pad.setPointerCapture(event.pointerId);
      this.drawing = true;
      this.truth = '';
      this.last = this.point(event);
      this.stroke(this.last, this.last);
      this.schedule();
    });
    ui.pad.addEventListener('pointermove', (event) => {
      if (!this.drawing) return;
      const point = this.point(event);
      this.stroke(this.last, point);
      this.last = point;
      this.schedule();
    });
    const end = () => {
      this.drawing = false;
    };
    ui.pad.addEventListener('pointerup', end);
    ui.pad.addEventListener('pointercancel', end);
    ui.padClear.addEventListener('click', () => this.clear());
    ui.padOpen.addEventListener('click', () => ui.padFile.click());
    ui.padFile.addEventListener('change', () => {
      const file = ui.padFile.files[0];
      ui.padFile.value = '';
      if (file) this.open(file);
    });
    ui.padSample.addEventListener('click', () => this.sample());
    this.clear();
  },
};

// ---------- Опрос состояния ----------

async function poll() {
  try {
    const server = await api('GET', '/api/state');
    const previous = state.server;
    state.server = server;
    ui.status.textContent = server.status;
    if (!previous || previous.revision !== server.revision || previous.dataVersion !== server.dataVersion) {
      views.invalidate('loss', 'metric', 'topology', 'boundary');
    }

    if (!previous || previous.sweep.revision !== server.sweep.revision) views.invalidate('sweep');
    renderSweep(server.sweep);
    ui.start.disabled = !server.ready;
    digits.setEnabled(server.supportsDigits);
    views.refresh();
    digits.refresh();
  } catch {
    ui.status.textContent = 'нет связи с сервером лаборатории — он запущен?';
  }

  setTimeout(poll, document.hidden ? HIDDEN_POLL_MS : POLL_MS);
}

async function init() {
  wirePanel();
  wireTabs();
  wireSweep();
  digits.wire();
  new ResizeObserver(() => views.refresh()).observe(document.querySelector('.workspace'));

  try {
    state.catalog = await api('GET', '/api/catalog');
  } catch (error) {
    ui.status.textContent = `не удалось загрузить каталог: ${error.message}`;
    return;
  }

  fillSelect(ui.dataset, state.catalog.datasets.map((d) => ({ value: d.key, label: d.name })));
  fillSelect(ui.scaling, state.catalog.scalings);
  fillSelect(ui.optimizer, state.catalog.optimizers);

  // Перезагрузка страницы не должна сбрасывать идущее обучение: подхватываем датасет сервера.
  const server = await api('GET', '/api/state').catch(() => null);
  const index = server?.dataset ? state.catalog.datasets.findIndex((d) => d.key === server.dataset) : -1;
  ui.dataset.selectedIndex = Math.max(0, index);
  applyDefaults(currentSpec());
  if (index < 0) await prepareData();

  poll();
}

init();
