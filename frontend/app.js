const $ = (id) => document.getElementById(id);
const form = $('prediction-form');
const hourInput = $('hour');
let currentMode = 'synthetic_demo';
const featureControls = {
  'Distance_km':['field-distance','Route distance'], 'Traffic':['field-traffic','Traffic level'],
  'Weather':['field-weather','Weather'], 'Vehicle':['field-vehicle','Vehicle'], 'Area':['field-area','Area type'],
  'Agent_Rating':['field-rating','Agent rating'], 'Agent_Age':['field-age','Agent age'],
  'Multiple_Deliveries':['field-deliveries','Other active deliveries'], 'Festival':['field-festival','Festival conditions'],
  'Hour':['field-hour','Dispatch hour'], 'Category':['field-category','Order category']
};

function formatHour(value) {
  const h = Number(value), suffix = h >= 12 ? 'PM' : 'AM', shown = h % 12 || 12;
  return `${shown}:00 ${suffix}`;
}
hourInput.addEventListener('input', () => $('hour-value').textContent = formatHour(hourInput.value));
$('notice-close')?.addEventListener('click', () => $('demo-warning').classList.add('hidden'));

function setError(el, message) { el.textContent = message; el.classList.toggle('hidden', !message); }
function updateArrival(minutes) {
  const arrival = new Date(Date.now() + minutes * 60_000);
  $('arrival').textContent = arrival.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'});
  $('arrival-date').textContent = arrival.toLocaleDateString([], {month:'short', day:'numeric'});
}
function renderStatus(status) {
  currentMode = status.mode;
  const demo = status.mode === 'synthetic_demo';
  const active = new Set(status.features || Object.keys(featureControls));
  $('download-package').classList.toggle('hidden', !status.package_ready);
  const unavailable = [];
  for (const [feature, [fieldId, label]] of Object.entries(featureControls)) {
    const field = $(fieldId);
    if (field) { field.classList.toggle('hidden', !active.has(feature)); if (!active.has(feature)) unavailable.push(label); }
  }
  const unsupported = $('unsupported-inputs');
  unsupported.textContent = unavailable.length ? `Not present in this dataset: ${unavailable.join(', ')}. These inputs are hidden and are not used by the model.` : '';
  unsupported.classList.toggle('hidden', unavailable.length === 0);
  $('header-model').textContent = demo ? 'Demo model' : 'Dataset model';
  $('demo-warning').classList.toggle('hidden', !demo);
  $('real-notice').classList.toggle('hidden', demo);
  $('model-status-title').textContent = demo ? 'Synthetic preview active' : 'Dataset model is ready';
  $('model-status-detail').textContent = demo ? 'Real dataset metrics are not loaded' : `${Number(status.training_rows).toLocaleString()} deliveries used for training`;
  $('model-status-dot').classList.toggle('real', !demo);
  if (status.metrics) renderMetrics(status.metrics);
}
function renderMetrics(report) {
  $('metrics-section').classList.remove('hidden');
  $('metrics-rows').textContent = `${Number(report.rows_used).toLocaleString()} records · ${Number(report.test_rows).toLocaleString()} held out`;
  $('metrics-note').textContent = `${report.winner} selected by lowest held-out MAE. ${report.note || ''}`;
  $('metrics-body').innerHTML = report.metrics.map(row => `<tr class="${row.Model === report.winner ? 'winner' : ''}"><td>${row.Model}</td><td>${Number(row['CV MAE (min)']).toFixed(1)} min</td><td>${Number(row['MAE (min)']).toFixed(1)} min</td><td>${Number(row['RMSE (min)']).toFixed(1)} min</td><td>${Number(row['MAPE (%)']).toFixed(1)}%</td><td>${Number(row['R²']).toFixed(3)}</td></tr>`).join('');
}
async function loadStatus() {
  try { const response = await fetch('/api/status'); if (!response.ok) throw new Error(); renderStatus(await response.json()); }
  catch { $('header-model').textContent = 'Backend unavailable'; $('model-status-title').textContent = 'Start the local server'; $('model-status-detail').textContent = 'Run python app.py to connect the prediction service'; }
}
form.addEventListener('submit', async (event) => {
  event.preventDefault(); setError($('form-error'), '');
  const button = form.querySelector('button[type="submit"]'); button.disabled = true; button.querySelector('span').textContent = 'Calculating…';
  const payload = {Distance_km:Number($('distance').value), Hour:Number(hourInput.value), Traffic:$('traffic').value, Weather:$('weather').value, Vehicle:$('vehicle').value, Area:$('area').value, City:$('area').value, Agent_Rating:Number($('rating').value), Agent_Age:Number($('age').value), Multiple_Deliveries:Number($('deliveries').value), Festival:$('festival').value, Category:$('category').value};
  try {
    const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Could not calculate an estimate.');
    $('result-empty').classList.add('hidden'); $('result-value').classList.remove('hidden'); $('minutes').textContent = Math.round(result.minutes); updateArrival(result.minutes);
    $('result-panel').scrollIntoView({behavior:'smooth', block:'nearest'});
  } catch (error) { setError($('form-error'), error.message || 'The prediction service could not be reached.'); }
  finally { button.disabled = false; button.querySelector('span').textContent = 'Calculate ETA'; }
});

const fileInput = $('dataset-file'), trainButton = $('train-button');
function showFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) { setError($('train-error'), 'Please choose a CSV file.'); return; }
  if (file.size > 120 * 1024 * 1024) { setError($('train-error'), 'The file is larger than the 120 MB upload limit.'); return; }
  setError($('train-error'), ''); $('file-name').textContent = file.name; $('selected-file').classList.remove('hidden'); $('train-button').disabled = false;
}
fileInput.addEventListener('change', () => showFile(fileInput.files[0]));
$('clear-file').addEventListener('click', () => { fileInput.value=''; $('selected-file').classList.add('hidden'); trainButton.disabled=true; });
const dropzone = $('dropzone');
dropzone.addEventListener('dragover', e => {e.preventDefault(); dropzone.classList.add('dragover');});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {e.preventDefault(); dropzone.classList.remove('dragover'); const file=e.dataTransfer.files[0]; if(file){try{const transfer=new DataTransfer();transfer.items.add(file);fileInput.files=transfer.files;}catch{} showFile(file);}});
trainButton.addEventListener('click', async () => {
  const file = fileInput.files[0]; if (!file) return;
  setError($('train-error'), ''); $('train-progress').classList.remove('hidden'); trainButton.disabled=true;
  const data = new FormData(); data.append('dataset', file);
  try {
    const response = await fetch('/api/train', {method:'POST', body:data}); const result=await response.json();
    if(!response.ok) throw new Error(result.error || 'Training failed.');
    renderMetrics(result.report); renderStatus({mode:'real_dataset', winner:result.report.winner, training_rows:result.report.rows_used, metrics:result.report, package_ready:result.package_ready, features:result.features});
    $('real-notice-text').textContent=`${Number(result.report.rows_used).toLocaleString()} valid records used · ${result.report.winner} selected`;
    $('real-notice').scrollIntoView({behavior:'smooth',block:'center'});
  } catch(error) { setError($('train-error'), error.message || 'Training failed.'); }
  finally { $('train-progress').classList.add('hidden'); trainButton.disabled=false; }
});
loadStatus();
