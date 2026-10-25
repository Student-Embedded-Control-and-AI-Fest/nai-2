'use strict';

/*
 * NoodleAI Mode 2 — held-out evaluation add-on
 *
 * Loaded after app.js. It adds an independent Evaluate tab without changing
 * the existing training/validation logic. The test NPZ is never used to fit
 * the scaler, train the TF.js model, calibrate INT8 activations, or choose the
 * confidence threshold.
 */

(() => {
  const REP_NAME = {
    accel: 'Accelerometer',
    gyro: 'Gyroscope',
    'accel+gyro': 'Accel + Gyro',
    quaternion: 'Relative Quaternion',
    velocity: 'Estimated Velocity',
    'velocity+quaternion': 'Velocity + Quaternion',
  };

  const ev = {
    dataset: null,
    fileName: '',
    mappedY: null,
    lastResult: null,
  };

  const byId = id => document.getElementById(id);
  const fmtPct = v => Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—';
  
  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .evaluate-layout{display:grid;grid-template-columns:minmax(0,1fr);gap:15px}
      .evaluate-controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-top:14px}
      .evaluate-controls .file-button{margin:0}
      .eval-summary{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:9px;margin-top:14px}
      .eval-metric{background:linear-gradient(180deg,#f8fafc,#f3f6f9);border:1px solid #e4e9ef;border-radius:12px;padding:12px;min-width:0}
      .eval-metric span{display:block;font-size:.58rem;letter-spacing:.11em;color:#8c97a7;font-weight:850;text-transform:uppercase}
      .eval-metric strong{display:block;margin-top:4px;font-size:1.18rem;letter-spacing:-.03em;color:#263347;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      .eval-metric small{display:block;margin-top:2px;color:#8a95a5;font-size:.67rem}
      .eval-grid{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:15px;margin-top:15px}
      .eval-card{border:1px solid #e2e7ed;border-radius:15px;padding:14px;background:#fff;box-shadow:var(--shadow-soft);overflow:auto}
      .eval-card h3{margin:0 0 4px;color:#263347;font-size:.92rem}
      .eval-card p{margin:0 0 11px;color:#8893a3;font-size:.72rem}
      .eval-table{border-collapse:collapse;width:100%;font-size:.74rem;color:#3c485a}
      .eval-table th,.eval-table td{border-bottom:1px solid #e7ebf0;padding:7px 8px;text-align:right;white-space:nowrap}
      .eval-table th{color:#677487;background:#f7f9fb;font-weight:800}
      .eval-table th:first-child,.eval-table td:first-child{text-align:left}
      .eval-table .diag{font-weight:900;color:#177b5d;background:#ebf8f3}
      .eval-note{margin-top:10px;color:#697689;font-size:.72rem}
      .eval-ready{color:#167657!important;border-color:#caedde!important;background:#effaf6!important}
      .eval-warning{color:#84651a!important;border-color:#f0ddb2!important;background:#fff9eb!important}
      @media(max-width:1050px){.eval-summary{grid-template-columns:repeat(3,1fr)}.eval-grid{grid-template-columns:1fr}}
      @media(max-width:650px){.eval-summary{grid-template-columns:1fr 1fr}.evaluate-controls{align-items:stretch;flex-direction:column}.evaluate-controls>*{width:100%}}
    `;
    document.head.appendChild(style);
  }

  function injectTabAndPage() {
    const tabs = document.querySelector('.tabs');
    const deployTab = document.querySelector('.tab[data-tab="deploy"]');
    const deployPage = byId('tab-deploy');
    if (!tabs || !deployTab || !deployPage) throw new Error('Evaluate add-on: app layout not found.');

    const deployNum = deployTab.querySelector('span');
    if (deployNum) deployNum.textContent = '05';

    const tab = document.createElement('button');
    tab.className = 'tab';
    tab.dataset.tab = 'evaluate';
    tab.innerHTML = '<span>04</span>Evaluate';
    tabs.insertBefore(tab, deployTab);

    const page = document.createElement('section');
    page.id = 'tab-evaluate';
    page.className = 'tab-page';
    page.innerHTML = `
      <div class="evaluate-layout">
        <div class="panel">
          <div class="section-title">
            <span class="step">1</span>
            <div>
              <div class="section-kicker">HELD-OUT TEST</div>
              <h2>Separate dataset evaluation</h2>
              <p>Load a separate NoodleAI .npz after training. Evaluation never uses the loaded test samples for training, scaler fitting, INT8 calibration, or threshold selection. Use a truly separate session for independent testing.</p>
            </div>
          </div>

          <div class="evaluate-controls">
            <label class="file-button">Load test .npz<input id="evaluateDatasetFile" type="file" accept=".npz" hidden></label>
            <button id="evaluateBtn" class="primary" disabled>Evaluate frozen model</button>
            <button id="exportTestPaperBtn" class="secondary" disabled>Export paper results (.zip)</button>
            <button id="clearEvaluateBtn" class="ghost" disabled>Clear test set</button>
          </div>
          <div id="evaluateStatus" class="status-box">Load an separately collected test .npz. You may load the test set before or after training.</div>
        </div>

        <div class="panel">
          <div class="section-title">
            <span class="step">2</span>
            <div>
              <div class="section-kicker">TEST RESULTS</div>
              <h2>Frozen-model performance</h2>
              <p>Float32 and deployable INT8 predictions use the scaler, representation, and confidence threshold selected from the training dataset only.</p>
            </div>
          </div>

          <div class="eval-summary">
            <div class="eval-metric"><span>Test windows</span><strong id="evalCount">—</strong><small>user-supplied samples</small></div>
            <div class="eval-metric"><span>Float32 accuracy</span><strong id="evalFloatAcc">—</strong><small>TF.js frozen model</small></div>
            <div class="eval-metric"><span>INT8 accuracy</span><strong id="evalInt8Acc">—</strong><small>Noodle quantized path</small></div>
            <div class="eval-metric"><span>FP32 macro F1</span><strong id="evalFloatF1">—</strong><small>before rejection</small></div>
            <div class="eval-metric"><span>INT8 macro F1</span><strong id="evalMacroF1">—</strong><small>before rejection</small></div>
            <div class="eval-metric"><span>FP32/INT8 agreement</span><strong id="evalAgreement">—</strong><small>same test windows</small></div>
            <div class="eval-metric"><span>Coverage</span><strong id="evalCoverage">—</strong><small id="evalThresholdNote">at frozen threshold</small></div>
            <div class="eval-metric"><span>Accepted accuracy</span><strong id="evalAcceptedAcc">—</strong><small>after rejection</small></div>
          </div>

          <div id="evalFloatDetail"></div>
          <hr class="nai-report-divider">
          <div id="evalInt8Detail"></div>
          <p class="nai-report-warning">Results above are browser-side tests on a separately loaded .npz. Independence from training sessions and physical BL702 equivalence must be established separately. Both confusion matrices use argmax before confidence rejection.</p>
          <div id="evalProvenance" class="eval-note"></div>
        </div>
      </div>
    `;
    deployPage.parentNode.insertBefore(page, deployPage);

    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === 'evaluate'));
      document.querySelectorAll('.tab-page').forEach(p => p.classList.toggle('active', p.id === 'tab-evaluate'));
      updateReadyState();
    });
  }

  function resetDisplayedResults() {
    for (const id of ['evalCount','evalFloatAcc','evalInt8Acc','evalFloatF1','evalMacroF1','evalAgreement','evalCoverage','evalAcceptedAcc']) {
      byId(id).textContent = '—';
    }
    byId('evalThresholdNote').textContent = 'at frozen threshold';
    byId('evalFloatDetail').textContent = '';
    byId('evalInt8Detail').textContent = '';
    byId('exportTestPaperBtn').disabled = true;
    window.NoodleMode2Paper?.setTest(null);
    byId('evalProvenance').textContent = '';
    ev.lastResult = null;
  }

  function currentModelReady() {
    return !!(state.model && state.scaler && state.pkg && state.pkg.qmodel && state.trainedRep && state.modelLabels?.length);
  }

  function updateReadyState() {
    const btn = byId('evaluateBtn');
    if (!btn) return;
    const haveTest = !!ev.dataset;
    const haveModel = currentModelReady();
    btn.disabled = !(haveTest && haveModel);
    byId('clearEvaluateBtn').disabled = !haveTest;

    const status = byId('evaluateStatus');
    status.classList.remove('eval-ready','eval-warning');
    if (haveTest && haveModel) {
      status.classList.add('eval-ready');
      status.textContent = `${ev.fileName}: ${ev.dataset.y.length} separately supplied windows loaded. Frozen ${REP_NAME[state.trainedRep] || state.trainedRep} model is ready for separate-dataset evaluation.`;
    } else if (haveTest) {
      status.classList.add('eval-warning');
      status.textContent = `${ev.fileName}: ${ev.dataset.y.length} separately supplied windows loaded. Train a model in this browser session before evaluating.`;
    } else if (haveModel) {
      status.textContent = `Frozen ${REP_NAME[state.trainedRep] || state.trainedRep} model ready. Load an separately collected test .npz.`;
    } else {
      status.textContent = 'Load an separately collected test .npz. You may load the test set before or after training.';
    }
  }

  function remapTargets(ds, modelLabels) {
    const toModel = ds.labels.map(label => modelLabels.indexOf(label));
    const unknown = ds.labels.filter((_, i) => toModel[i] < 0);
    if (unknown.length) throw new Error(`Test dataset contains labels not present in the trained model: ${unknown.join(', ')}`);

    const absent = modelLabels.filter(label => !ds.labels.includes(label));
    if (absent.length) {
      console.warn('Held-out test set has no samples for model classes:', absent);
    }

    const y = new Int32Array(ds.y.length);
    for (let i=0;i<ds.y.length;i++) {
      const source = Number(ds.y[i]);
      if (!Number.isInteger(source) || source < 0 || source >= toModel.length) {
        throw new Error(`Invalid test label index ${source} at row ${i}`);
      }
      y[i] = toModel[source];
    }
    return y;
  }

  function thresholdMetrics(probs, y, threshold) {
    let accepted = 0, correct = 0;
    for (let i=0;i<y.length;i++) {
      const p = probs[i];
      let pi = 0, pv = p[0];
      for (let k=1;k<p.length;k++) if (p[k] > pv) { pv = p[k]; pi = k; }
      if (pv >= threshold) {
        accepted++;
        if (pi === y[i]) correct++;
      }
    }
    return {
      accepted,
      rejected: y.length - accepted,
      coverage: y.length ? accepted / y.length : 0,
      acceptedAccuracy: accepted ? correct / accepted : 0,
    };
  }

  async function loadTestFile(file) {
    const arrays = await NAI.loadNpz(file);
    const ds = NAI.parseDatasetArrays(arrays);
    if (ds.sampleRate !== NAI.SAMPLE_RATE_HZ) {
      throw new Error(`Test dataset is ${ds.sampleRate} Hz; this Mode 2 runtime expects ${NAI.SAMPLE_RATE_HZ} Hz.`);
    }
    ev.dataset = ds;
    ev.fileName = file.name;
    ev.mappedY = null;
    resetDisplayedResults();
    updateReadyState();
    if (typeof log === 'function') log(`Loaded held-out test set ${file.name}: ${ds.y.length} windows, N=${ds.N}, labels=${ds.labels.join(', ')}.`);
  }

  async function evaluateFrozenModel() {
    if (!ev.dataset) throw new Error('Load an separately collected test .npz first.');
    if (!currentModelReady()) throw new Error('Train a model in this browser session first. Evaluation never trains or refits the model.');

    const ds = ev.dataset;
    if (ds.N !== state.pkg.N) {
      throw new Error(`Window mismatch: test N=${ds.N}, trained model N=${state.pkg.N}.`);
    }

    const y = remapTargets(ds, state.modelLabels);
    const rows = NAI.buildRepresentationDataset(ds, state.trainedRep);
    if (!rows.length) throw new Error('Test dataset contains no samples.');
    if (rows[0].length !== state.scaler.mean.length) {
      throw new Error(`Input mismatch: test representation has ${rows[0].length} features; trained scaler expects ${state.scaler.mean.length}.`);
    }

    const all = Array.from({length:rows.length}, (_,i)=>i);
    const standardized = NAI.standardizeRows(rows, all, state.scaler);

    const floatPred = await NAI.predictTf(state.model, standardized, state.modelLabels.length);
    const qPred = NAI.predictQuantized(state.pkg.qmodel, standardized, state.modelLabels.length);
    const floatAcc = NAI.accuracy(floatPred.pred, Array.from(y));
    const int8Acc = NAI.accuracy(qPred.pred, Array.from(y));

    const floatMetrics = NoodleMetrics.evaluate(y, floatPred.pred, state.modelLabels);
    const int8Metrics = NoodleMetrics.evaluate(y, qPred.pred, state.modelLabels);
    const nAgree=Array.from(floatPred.pred).reduce((sum,p,i)=>sum+(p===qPred.pred[i]?1:0),0);
    const agreement={matches:nAgree,n:y.length,rate:nAgree/y.length};
    const threshold = Number(state.pkg.quant?.confidenceThreshold ?? state.pkg.sweep?.selected?.threshold ?? 0);
    const tm = thresholdMetrics(qPred.probs, y, threshold);
    const pairedPredictions=Array.from(y,(actual,i)=>({datasetIndex:i,actual:Number(actual),
      fp32Pred:Number(floatPred.pred[i]),int8Pred:Number(qPred.pred[i]),
      fp32Probs:Array.from(floatPred.probs[i]),int8Probs:Array.from(qPred.probs[i])}));

    byId('evalCount').textContent = String(y.length);
    byId('evalFloatAcc').textContent = fmtPct(floatAcc);
    byId('evalInt8Acc').textContent = fmtPct(int8Acc);
    byId('evalFloatF1').textContent = floatMetrics.macro.f1.toFixed(3);
    byId('evalMacroF1').textContent = int8Metrics.macro.f1.toFixed(3);
    byId('evalAgreement').textContent = `${nAgree} / ${y.length} (${fmtPct(agreement.rate)})`;
    byId('evalCoverage').textContent = fmtPct(tm.coverage);
    byId('evalAcceptedAcc').textContent = fmtPct(tm.acceptedAccuracy);
    byId('evalThresholdNote').textContent = `T=${threshold.toFixed(2)} · ${tm.rejected} rejected`;
    byId('evalFloatDetail').innerHTML = window.NoodleMode2Paper.metricsHTML(floatMetrics,'FP32 · separate test');
    byId('evalInt8Detail').innerHTML = window.NoodleMode2Paper.metricsHTML(int8Metrics,'INT8 · separate test');

    byId('evalProvenance').textContent =
      `${ev.fileName} · ${REP_NAME[state.trainedRep] || state.trainedRep} · topology ${state.pkg.dims.join(' → ')} · `+
      `scaler frozen from training split · confidence threshold frozen from validation split · test set not used for training/calibration/threshold selection.`;

    ev.mappedY = y;
    ev.lastResult = {fp32:floatMetrics,int8:int8Metrics,agreement,pairedPredictions,
      thresholdMetrics:tm,threshold,fileName:ev.fileName,createdUtc:new Date().toISOString(),
      modelSignature:state.pkg.dims.join(',')};
    window.NoodleMode2Paper.setTest(ev.lastResult);
    byId('exportTestPaperBtn').disabled = false;
    byId('evaluateStatus').classList.add('eval-ready');
    byId('evaluateStatus').textContent =
      `Separate-dataset evaluation complete: Float32 ${fmtPct(floatAcc)} · INT8 ${fmtPct(int8Acc)} · `+
      `INT8 macro F1 ${fmtPct(int8Metrics.macro.f1)} · agreement ${fmtPct(agreement.rate)} · coverage ${fmtPct(tm.coverage)} at T=${threshold.toFixed(2)}.`;

    if (typeof log === 'function') {
      log(`Separate test ${ev.fileName}: FP32 ${fmtPct(floatAcc)}, INT8 ${fmtPct(int8Acc)}, INT8 macro F1 ${fmtPct(int8Metrics.macro.f1)}, FP32/INT8 agreement ${fmtPct(agreement.rate)}, coverage ${fmtPct(tm.coverage)}, accepted accuracy ${fmtPct(tm.acceptedAccuracy)}, T=${threshold.toFixed(2)}.`);
    }
  }

  function bindEvents() {
    byId('exportTestPaperBtn').addEventListener('click', () => window.NoodleMode2Paper.exportReport());
    byId('evaluateDatasetFile').addEventListener('change', async e => {
      try {
        const file = e.target.files[0];
        if (file) await loadTestFile(file);
      } catch (err) {
        alert(err.message);
        if (typeof log === 'function') log(`Held-out dataset load ERROR: ${err.message}`);
      } finally {
        e.target.value = '';
      }
    });

    byId('evaluateBtn').addEventListener('click', async () => {
      const btn = byId('evaluateBtn');
      try {
        btn.disabled = true;
        btn.textContent = 'Evaluating…';
        await evaluateFrozenModel();
      } catch (err) {
        alert(err.message);
        byId('evaluateStatus').classList.remove('eval-ready');
        byId('evaluateStatus').classList.add('eval-warning');
        byId('evaluateStatus').textContent = err.message;
        if (typeof log === 'function') log(`Held-out evaluation ERROR: ${err.message}`);
      } finally {
        btn.textContent = 'Evaluate frozen model';
        updateReadyState();
      }
    });

    byId('clearEvaluateBtn').addEventListener('click', () => {
      ev.dataset = null;
      ev.fileName = '';
      ev.mappedY = null;
      resetDisplayedResults();
      updateReadyState();
      if (typeof log === 'function') log('Cleared held-out test dataset.');
    });

    // app.js invalidates and replaces the model during normal operation. Watch
    // the existing training status so the Evaluate button always reflects the
    // actual frozen-model state without modifying app.js.
    const trainResult = byId('trainResult');
    if (trainResult) {
      new MutationObserver(() => {
        resetDisplayedResults();
        updateReadyState();
      }).observe(trainResult, {childList:true,subtree:true,characterData:true});
    }
  }

  try {
    injectStyles();
    injectTabAndPage();
    bindEvents();
    resetDisplayedResults();
    updateReadyState();
    if (typeof log === 'function') log('Held-out Evaluate tab loaded.');
  } catch (err) {
    console.error(err);
  }
})();
