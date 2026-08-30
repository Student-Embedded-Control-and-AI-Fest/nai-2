
'use strict';

const $=id=>document.getElementById(id);
const MODE2_DEPLOY_SUPPORTED=true;
const REP_LABEL={accel:'Accelerometer',gyro:'Gyroscope','accel+gyro':'Accel + Gyro',quaternion:'Relative Quaternion',velocity:'Estimated Velocity','velocity+quaternion':'Velocity + Quaternion'};
const state={
  labels:[],samples:[],rawSamples:[],targets:[],rawLengths:[],durationsMs:[],N:75,setupLocked:false,
  pendingRaw:null,pendingWindow:null,pendingWindows:null,pendingLabelIndex:null,pendingDurationMs:0,recording:null,recordStartTms:null,
  model:null,scaler:null,pkg:null,history:null,thresholdSweep:null,trainedRep:null,modelLabels:[],deviceMode:'?',
  live:{t:[],accel:[[],[],[]],gyro:[[],[],[]]},
};

function log(msg){const t=new Date().toLocaleTimeString();$('log').textContent+=`[${t}] ${msg}\n`;$('log').scrollTop=$('log').scrollHeight;}
function downloadBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
function pct(v){return `${(100*v).toFixed(1)}%`;}
function switchTab(name){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.tab-page').forEach(p=>p.classList.toggle('active',p.id===`tab-${name}`));}

document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>switchTab(b.dataset.tab)));

async function initTf(){try{await tf.setBackend('cpu');await tf.ready();$('tfBadge').textContent=`TF.js ${tf.version.tfjs} · ${tf.getBackend()}`;log(`TensorFlow.js ${tf.version.tfjs} ready (${tf.getBackend()}).`);}catch(e){$('tfBadge').textContent='TensorFlow.js unavailable';log(`TensorFlow.js ERROR: ${e.message}`);}}

function datasetObject(){
  const n=state.targets.length;
  return {Xrows:state.samples,y:Int32Array.from(state.targets),labels:[...state.labels],N:state.N,sampleRate:NAI.SAMPLE_RATE_HZ,
    rawRows:state.rawSamples.length===n&&n?state.rawSamples:null,width:state.N*6,rawLengths:[...state.rawLengths],durationsMs:[...state.durationsMs]};
}

function parseHidden(){const h=$('hiddenLayers').value.split(',').map(s=>s.trim()).filter(Boolean).map(Number);if(!h.length||h.some(v=>!Number.isInteger(v)||v<1||v>512))throw new Error('Hidden layers must be comma-separated integers from 1 to 512.');return h;}

function selectedRep(){return $('representation').value;}
function updateInputAndTopology(){
  state.N=Math.max(10,Math.min(500,Number($('normalizedLength').value)||75));
  const stride=Math.max(1,Math.min(state.N,Number($('windowStride')?.value)||5));
  const rep=selectedRep(),D=state.N*NAI.REP[rep].channels;
  $('inputDimSummary').textContent=`${state.N} × ${NAI.REP[rep].channels} = ${D}`;
  if($('windowTimeSummary')) $('windowTimeSummary').textContent=`${state.N} @ ${NAI.SAMPLE_RATE_HZ} Hz = ${(state.N/NAI.SAMPLE_RATE_HZ).toFixed(2)} s · stride ${stride} = ${(stride/NAI.SAMPLE_RATE_HZ).toFixed(2)} s`;
  try{$('topology').textContent=`Topology: ${[D,...parseHidden(),Math.max(state.labels.length,0)].join(' → ')}  [${REP_LABEL[rep]}]`;}catch(e){$('topology').textContent=`Topology: ${e.message}`;}
}

function refreshLabels(){
  $('labelList').innerHTML='';$('recordLabel').innerHTML='';
  state.labels.forEach((label,i)=>{const a=document.createElement('option');a.value=String(i);a.textContent=label;$('labelList').appendChild(a);const b=a.cloneNode(true);$('recordLabel').appendChild(b);});
  $('classCount').textContent=String(state.labels.length);
  updateInputAndTopology();
}

function refreshDataset(){
  const n=state.targets.length;$('sampleCount').textContent=String(n);$('saveDatasetBtn').disabled=!n;
  $('rawDatasetStatus').textContent=n===0?'—':state.rawSamples.length===n?'available':'legacy only';
  const counts=Array(state.labels.length).fill(0);state.targets.forEach(y=>{if(y>=0&&y<counts.length)counts[y]++;});
  $('classCounts').innerHTML='';counts.forEach((c,i)=>{const s=document.createElement('span');s.className='class-pill';s.textContent=`${state.labels[i]}: ${c}`;$('classCounts').appendChild(s);});
  $('trainBtn').disabled=!(state.setupLocked&&n>0);
}

function invalidateModel(){if(state.model){try{state.model.dispose();}catch(_){}}state.model=null;state.scaler=null;state.pkg=null;state.history=null;state.thresholdSweep=null;state.trainedRep=null;state.modelLabels=[];$('saveModelBtn').disabled=true;$('deployBtn').disabled=true;$('trainResult').textContent='';$('curveSummary').textContent='Train a model to see loss and accuracy history.';$('quantResult').textContent='After float training: post-training INT8 quantization uses the training split for activation calibration, then sweeps confidence 0.30 → 0.90 on the held-out validation split.';drawTrainingCurves();drawThresholdSweep();}

function lockUi(locked){
  state.setupLocked=locked;
  const hasData=state.targets.length>0;

  /*
   * Existing windows must all have the same N, so once data exists the
   * window length is immutable.  Unlocking is still allowed for:
   *   - adding new labels/classes
   *   - recording more data
   *   - changing stride for future recordings
   */
  $('normalizedLength').disabled=locked||hasData;
  $('windowStride').disabled=locked;
  $('labelEntry').disabled=locked;
  $('addLabelBtn').disabled=locked;
  $('removeLabelBtn').disabled=locked;

  /* This is now a true Lock / Unlock toggle. */
  $('lockSetupBtn').disabled=false;
  $('lockSetupBtn').textContent=locked?'Unlock setup':'Lock setup';

  $('recordLabel').disabled=!locked;
  $('armRecordBtn').disabled=!(locked&&noodleBLE.connected);
  $('stopRecordBtn').disabled=true;
  $('trainBtn').disabled=!(locked&&state.targets.length);
}

function resetDataset(confirmFirst=true){
  if(confirmFirst&&(state.targets.length||state.setupLocked)&&!confirm('Clear all recorded samples and unlock the dataset setup?'))return;
  state.labels=[];state.samples=[];state.rawSamples=[];state.targets=[];state.rawLengths=[];state.durationsMs=[];state.pendingRaw=null;state.pendingWindow=null;state.pendingWindows=null;state.pendingLabelIndex=null;state.recording=null;state.recordStartTms=null;state.N=Number($('normalizedLength').value)||75;
  invalidateModel();lockUi(false);refreshLabels();refreshDataset();$('recordProgress').textContent='Define Mode 2 labels and lock setup first.';$('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;
}

$('addLabelBtn').addEventListener('click',()=>{
  if(state.setupLocked)return;
  const label=$('labelEntry').value.trim();
  if(!label)return;
  if(state.labels.includes(label)){alert('That label already exists.');return;}
  state.labels.push(label);
  $('labelEntry').value='';
  invalidateModel();
  refreshLabels();
  refreshDataset();
  log(`Added class “${label}”. Re-lock setup, record examples, then retrain.`);
});
$('labelEntry').addEventListener('keydown',e=>{if(e.key==='Enter'){$('addLabelBtn').click();e.preventDefault();}});
$('removeLabelBtn').addEventListener('click',()=>{
  if(state.setupLocked)return;
  const i=$('labelList').selectedIndex;
  if(i<0)return;

  const count=state.targets.reduce((n,y)=>n+(y===i?1:0),0);
  if(count>0){
    alert(`Cannot remove “${state.labels[i]}” because it already has ${count} saved windows. Add classes freely, but existing populated classes are protected.`);
    return;
  }

  const removed=state.labels[i];
  state.labels.splice(i,1);

  /* Removing an empty class shifts later label indices. */
  state.targets=state.targets.map(y=>y>i?y-1:y);

  invalidateModel();
  refreshLabels();
  refreshDataset();
  log(`Removed empty class “${removed}”.`);
});
$('normalizedLength').addEventListener('input',updateInputAndTopology);$('windowStride').addEventListener('input',updateInputAndTopology);$('representation').addEventListener('change',updateInputAndTopology);$('hiddenLayers').addEventListener('input',updateInputAndTopology);

$('lockSetupBtn').addEventListener('click',()=>{
  if(state.setupLocked){
    if(state.recording){
      alert('Stop the current recording before unlocking the dataset.');
      return;
    }

    lockUi(false);
    refreshDataset();

    if(state.targets.length){
      $('recordProgress').textContent=`Dataset unlocked. Existing ${state.targets.length} windows are preserved. Add classes or change stride, then Lock setup again to record more data. Window length stays fixed at ${state.N}.`;
    }else{
      $('recordProgress').textContent='Dataset setup unlocked. Edit labels/window settings, then Lock setup to record.';
    }

    log(`Dataset unlocked without clearing data (${state.targets.length} saved windows preserved).`);
    return;
  }

  const N=Number($('normalizedLength').value),stride=Number($('windowStride').value);
  if(!Number.isInteger(N)||N<10||N>500){alert('Use a Mode 2 window length from 10 to 500 samples.');return;}
  if(!Number.isInteger(stride)||stride<1||stride>N){alert('Window stride must be an integer from 1 to the window length.');return;}
  if(state.labels.length<2){alert('Define at least two labels first.');return;}

  if(state.targets.length&&N!==state.N){
    alert(`This dataset already contains ${state.N}-sample windows. Window length cannot change without resetting the dataset.`);
    $('normalizedLength').value=String(state.N);
    return;
  }

  state.N=N;
  lockUi(true);
  refreshDataset();
  $('recordProgress').textContent=`Ready. Select a label, press START, perform it repeatedly, then press STOP. Windows: ${N} samples, stride ${stride}.`;
  log(`Mode 2 dataset locked: window=${N} samples (${(N/NAI.SAMPLE_RATE_HZ).toFixed(2)} s), stride=${stride}, labels=${state.labels.length}.`);
});
$('resetDatasetBtn').addEventListener('click',()=>resetDataset(true));

function clearPendingRecording(){
  state.pendingRaw=null;state.pendingWindow=null;state.pendingWindows=null;state.pendingLabelIndex=null;state.pendingDurationMs=0;
  $('saveSampleBtn').disabled=true;$('discardSampleBtn').disabled=true;
}

$('armRecordBtn').addEventListener('click',()=>{
  try{
    if(!state.setupLocked)throw new Error('Lock the Mode 2 dataset setup first.');
    if(!noodleBLE.connected)throw new Error('Connect the device first.');
    const idx=$('recordLabel').selectedIndex;
    if(idx<0||idx>=state.labels.length)throw new Error('Choose a label.');
    clearPendingRecording();
    state.pendingLabelIndex=idx;
    state.recording=[];
    state.recordStartTms=null;
    $('recordLabel').disabled=true;
    $('armRecordBtn').disabled=true;
    $('stopRecordBtn').disabled=false;
    $('discardSampleBtn').disabled=false;
    $('recordProgress').textContent=`Recording “${state.labels[idx]}”… perform the gesture repeatedly, then press STOP.`;
    log(`Mode 2 START: ${state.labels[idx]}.`);
  }catch(e){alert(e.message);}
});

$('stopRecordBtn').addEventListener('click',()=>{
  if(!state.recording)return;
  const rows=state.recording;
  const idx=state.pendingLabelIndex;
  const N=state.N;
  const stride=Math.max(1,Math.min(N,Number($('windowStride').value)||5));
  const count=rows.length;
  const durationMs=count>1?(rows[count-1][6]-rows[0][6]):0;

  state.recording=null;
  state.recordStartTms=null;
  $('stopRecordBtn').disabled=true;
  $('armRecordBtn').disabled=!(state.setupLocked&&noodleBLE.connected);
  $('recordLabel').disabled=!state.setupLocked;

  if(count<N){
    clearPendingRecording();
    $('recordProgress').textContent=`Only ${count} samples captured. Need at least ${N} samples (${(N/NAI.SAMPLE_RATE_HZ).toFixed(2)} s). Try again.`;
    log(`Mode 2 STOP: too short (${count}/${N} samples).`);
    return;
  }

  const windows=[];
  for(let start=0;start+N<=count;start+=stride){
    const raw=new Float32Array(N*6);
    for(let r=0;r<N;r++){
      const row=rows[start+r];
      for(let c=0;c<6;c++)raw[r*6+c]=row[c];
    }
    const rawObj={data:raw,length:N};
    windows.push({raw:rawObj,window:NAI.normalizeRawSixAxis(rawObj,N)});
  }

  state.pendingWindows=windows;
  state.pendingDurationMs=durationMs;
  $('recordLabel').disabled=true;
  $('armRecordBtn').disabled=true;
  $('saveSampleBtn').disabled=false;
  $('discardSampleBtn').disabled=false;
  $('recordProgress').textContent=`Captured ${count} samples (${(durationMs/1000).toFixed(2)} s) → ${windows.length} overlapping windows (${N} samples, stride ${stride}). Save or discard.`;
  log(`Mode 2 STOP: ${state.labels[idx]} · raw=${count} · windows=${windows.length}.`);
});

$('saveSampleBtn').addEventListener('click',()=>{
  if(!state.pendingWindows?.length||state.pendingLabelIndex==null)return;
  const idx=state.pendingLabelIndex;
  const durationPerWindow=Math.round(1000*state.N/NAI.SAMPLE_RATE_HZ);
  for(const w of state.pendingWindows){
    state.samples.push(Float32Array.from(w.window));
    state.rawSamples.push({data:Float32Array.from(w.raw.data),length:w.raw.length});
    state.targets.push(idx);
    state.rawLengths.push(w.raw.length);
    state.durationsMs.push(durationPerWindow);
  }
  const added=state.pendingWindows.length;
  log(`Saved Mode 2 “${state.labels[idx]}”: ${added} windows, ${state.N}×6 raw samples each.`);
  clearPendingRecording();
  state.pendingLabelIndex=null;
  $('recordLabel').disabled=!state.setupLocked;
  $('armRecordBtn').disabled=!(state.setupLocked&&noodleBLE.connected);
  $('recordProgress').textContent=`Saved ${added} windows. Select a label and press START for another continuous recording.`;
  invalidateModel();refreshDataset();
});

$('discardSampleBtn').addEventListener('click',()=>{
  const wasRecording=!!state.recording;
  state.recording=null;state.recordStartTms=null;
  clearPendingRecording();state.pendingLabelIndex=null;
  $('stopRecordBtn').disabled=true;
  $('armRecordBtn').disabled=!(state.setupLocked&&noodleBLE.connected);
  $('recordLabel').disabled=!state.setupLocked;
  $('recordProgress').textContent=wasRecording?'Recording cancelled. Press START when ready.':'Discarded. Press START when ready.';
});

$('saveDatasetBtn').addEventListener('click',async()=>{try{const blob=await NAI.buildDatasetNpzBlob(datasetObject());downloadBlob(blob,'noodleai_dataset.npz');log('Saved NAI4 dataset (.npz).');}catch(e){alert(e.message);}});
$('loadDatasetBtn').addEventListener('click',()=>$('datasetFile').click());
$('datasetFile').addEventListener('change',async e=>{try{
  const file=e.target.files[0];
  if(!file)return;
  if((state.targets.length||state.setupLocked)&&!confirm('Replace the current dataset/setup?'))return;

  const arrays=await NAI.loadNpz(file);
  const ds=NAI.parseDatasetArrays(arrays);

  resetDataset(false);
  state.N=ds.N;
  $('normalizedLength').value=String(ds.N);
  state.labels=[...ds.labels];
  state.samples=ds.Xrows.map(r=>Float32Array.from(r));
  state.targets=Array.from(ds.y,Number);
  state.rawSamples=ds.rawRows?ds.rawRows.map(r=>({data:Float32Array.from(r.data),length:r.length})):[];
  state.rawLengths=arrays.raw_lengths?Array.from(arrays.raw_lengths.data,Number):Array(state.targets.length).fill(ds.N);
  state.durationsMs=arrays.durations_ms?Array.from(arrays.durations_ms.data,Number):Array(state.targets.length).fill(0);

  refreshLabels();
  lockUi(true);
  refreshDataset();

  $('recordProgress').textContent=`Loaded ${state.targets.length} windows from ${file.name}. Click “Unlock setup” to add classes or more data; the ${state.N}-sample window length is preserved.`;
  log(`Loaded ${file.name}: ${state.targets.length} windows; raw windows ${state.rawSamples.length===state.targets.length?'available':'not available'}. Dataset can be unlocked and extended.`);
}catch(err){
  alert(err.message);
  log(`Dataset load ERROR: ${err.message}`);
}finally{
  e.target.value='';
}});

async function trainModel(){
  try{
    if(!state.setupLocked||!state.targets.length)throw new Error('Create or load a dataset first.');
    const hidden=parseHidden(),epochs=Number($('epochs').value),rep=selectedRep();if(!Number.isInteger(epochs)||epochs<10||epochs>5000)throw new Error('Epochs must be 10..5000.');
    const ds=datasetObject();const K=state.labels.length;const counts=Array(K).fill(0);state.targets.forEach(y=>counts[y]++);if(Math.min(...counts)<2)throw new Error('Each class needs at least two samples for a stratified train/validation split.');
    $('trainBtn').disabled=true;$('trainResult').textContent='Preparing representation…';
    const rows=NAI.buildRepresentationDataset(ds,rep);const D=rows[0].length;const split=NAI.sklearnStratifiedSplit(ds.y,K,42);const scaler=NAI.fitScaler(rows,split.train);const Xtr=NAI.standardizeRows(rows,split.train,scaler),Xva=NAI.standardizeRows(rows,split.test,scaler);const ytr=Int32Array.from(split.train.map(i=>ds.y[i])),yva=Int32Array.from(split.test.map(i=>ds.y[i]));
    const model=NAI.makeModel(D,hidden,K,split.train.length);const tx=tf.tensor2d(NAI.flattenRows(Xtr),[Xtr.length,D],'float32'),vx=tf.tensor2d(NAI.flattenRows(Xva),[Xva.length,D],'float32');let ty,vy;if(K===2){ty=tf.tensor2d(Float32Array.from(ytr),[ytr.length,1]);vy=tf.tensor2d(Float32Array.from(yva),[yva.length,1]);}else{const ity=tf.tensor1d(ytr,'int32'),ivy=tf.tensor1d(yva,'int32');ty=tf.oneHot(ity,K);vy=tf.oneHot(ivy,K);ity.dispose();ivy.dispose();}
    const hist={epoch:[],loss:[],valLoss:[],acc:[],valAcc:[]};$('trainResult').textContent=`Training ${D} → ${hidden.join(' → ')} → ${K}…`;switchTab('curves');
    await model.fit(tx,ty,{epochs,batchSize:Math.min(200,split.train.length),shuffle:true,validationData:[vx,vy],verbose:0,callbacks:{onEpochEnd:async(epoch,l)=>{const acc=l.acc??l.accuracy??0,va=l.val_acc??l.val_accuracy??0;hist.epoch.push(epoch+1);hist.loss.push(l.loss);hist.valLoss.push(l.val_loss);hist.acc.push(acc);hist.valAcc.push(va);if(epoch===0||(epoch+1)%5===0||epoch+1===epochs){$('curveSummary').textContent=`Epoch ${epoch+1}/${epochs} · loss ${l.loss.toFixed(4)} · validation accuracy ${pct(va)}`;drawTrainingCurves(hist);await tf.nextFrame();}}}});
    tx.dispose();vx.dispose();ty.dispose();vy.dispose();
    const tr=await NAI.predictTf(model,Xtr,K),va=await NAI.predictTf(model,Xva,K);const trainAcc=NAI.accuracy(tr.pred,Array.from(ytr)),valAcc=NAI.accuracy(va.pred,Array.from(yva));
    $('trainResult').textContent=`Float validation ${pct(valAcc)} · quantizing INT8…`;
    const stride=Math.max(1,Math.min(state.N,Number($('windowStride').value)||5));
    const pkg=await NAI.exportNai4Int8(model,scaler,state.labels,state.N,rep,Xtr,Xva,yva,stride);
    if(state.model){try{state.model.dispose();}catch(_){}}state.model=model;state.scaler=scaler;state.pkg=pkg;state.history=hist;state.thresholdSweep=pkg.sweep;state.trainedRep=rep;state.modelLabels=[...state.labels];
    const q=pkg.sweep.selected;
    $('trainResult').textContent=`Float val ${pct(valAcc)} · INT8 val ${pct(pkg.qValAcc)} · T=${q.threshold.toFixed(2)} · ${(pkg.total/1024).toFixed(1)} KiB`;
    const f=pkg.filter;const fdesc=(f.flags&NAI.FILTER_HP)?`${f.highpassHz.toFixed(2)}–${f.lowpassHz.toFixed(1)} Hz band-pass`:`${f.lowpassHz.toFixed(1)} Hz low-pass`;
    $('quantResult').textContent=`INT8 PTQ complete · ${fdesc} · validation ${pct(pkg.qValAcc)} · threshold sweep 0.30–0.90 selected ${q.threshold.toFixed(2)} · accepted accuracy ${pct(q.acceptedAccuracy)} · coverage ${pct(q.coverage)} · stride ${stride} samples · raw package ${(pkg.total/1024).toFixed(1)} KiB.`;
    $('curveSummary').textContent=`Finished ${epochs} epochs · float validation ${pct(valAcc)} · INT8 validation ${pct(pkg.qValAcc)} · selected threshold ${q.threshold.toFixed(2)}`;$('saveModelBtn').disabled=false;$('deployBtn').disabled=!MODE2_DEPLOY_SUPPORTED||!noodleBLE.connected;drawTrainingCurves(hist);drawThresholdSweep(pkg.sweep);log(`Training + INT8 complete: ${REP_LABEL[rep]}, ${fdesc}, topology ${pkg.dims.join('→')}, float validation=${pct(valAcc)}, INT8 validation=${pct(pkg.qValAcc)}, threshold=${q.threshold.toFixed(2)}, coverage=${pct(q.coverage)}, NAI4 INT8=${(pkg.total/1024).toFixed(1)} KiB.`);switchTab('dataset');
  }catch(e){alert(e.message);log(`Training ERROR: ${e.message}`);}finally{$('trainBtn').disabled=!(state.setupLocked&&state.targets.length);}
}
$('trainBtn').addEventListener('click',trainModel);

$('deployNaiFile').addEventListener('change',async e=>{
  try{
    const file=e.target.files[0];
    if(!file)return;

    const pkg=await NAI.loadNaiPackage(file);

    if(!pkg.int8)throw new Error('This is a Float32 NAI4 package. Mode 2 v0.4.5 firmware expects the new INT8 NAI4 package; load the dataset and Train + INT8 calibrate first.');
    state.pkg=pkg;
    state.trainedRep=pkg.rep;
    state.modelLabels=[...pkg.labels];

    $('saveModelBtn').disabled=false;
    $('deployBtn').disabled=!noodleBLE.connected;

    $('deployText').textContent=
      `Loaded ${file.name}: ${(pkg.total/1024).toFixed(1)} KiB · `+
      `${pkg.N} samples · ${pkg.inputDim} inputs · ${pkg.labels.length} classes · INT8 · threshold ${pkg.quant.confidenceThreshold.toFixed(2)} · stride ${pkg.quant.strideSamples}`+
      `${pkg.filter?` · filter ${(pkg.filter.flags&NAI.FILTER_HP)?`${pkg.filter.highpassHz.toFixed(2)}–${pkg.filter.lowpassHz.toFixed(1)} Hz BP`:`${pkg.filter.lowpassHz.toFixed(1)} Hz LP`}`:' · legacy preprocessing'}. Ready to deploy.`;

    log(`Loaded deployable NAI4 INT8 package ${file.name}: ${pkg.dims.join('→')}, threshold=${pkg.quant.confidenceThreshold.toFixed(2)}, stride=${pkg.quant.strideSamples}, ${(pkg.total/1024).toFixed(1)} KiB.`);
  }catch(err){
    alert(err.message);
    log(`NAI load ERROR: ${err.message}`);
  }finally{
    e.target.value='';
  }
});

$('saveModelBtn').addEventListener('click',()=>{if(!state.pkg)return;downloadBlob(state.pkg.blob,`noodleai_${state.trainedRep||'model'}.nai`);});
$('deployBtn').addEventListener('click',async()=>{if(!state.pkg)return;try{$('deployBtn').disabled=true;$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;const chunk=Number($('chunkSize').value);await noodleBLE.deployPackage(state.pkg.files,{chunkSize:chunk,onProgress:p=>{const prog=$('deployProgress');prog.max=Math.max(1,p.total);prog.value=p.sent;const percent=p.total?Math.round(100*p.sent/p.total):0;const msg={begin:'Erasing single model region…','file-begin':`Preparing ${p.file}…`,sending:`${p.file}: ${percent}% total`,commit:'Files CRC-verified; committing single model slot…',done:'MODEL_STORED — files verified and committed.',error:`Deployment failed: ${p.error||'unknown error'}`}[p.stage]||p.stage;$('deployText').textContent=msg;}});log('Deployment complete: MODEL_STORED.');}catch(e){alert(e.message);log(`Deployment ERROR: ${e.message}`);}finally{$('deployBtn').disabled=!MODE2_DEPLOY_SUPPORTED||!(noodleBLE.connected&&state.pkg);$('trainingModeBtn').disabled=!noodleBLE.connected;$('inferenceModeBtn').disabled=!noodleBLE.connected;}});
$('trainingModeBtn').addEventListener('click',async()=>{try{await noodleBLE.setTraining();}catch(e){alert(e.message);}});$('inferenceModeBtn').addEventListener('click',async()=>{try{await noodleBLE.setInference();}catch(e){alert(e.message);}});

$('connectBtn').addEventListener('click',async()=>{try{if(noodleBLE.connected)await noodleBLE.disconnect();else await noodleBLE.connect();}catch(e){alert(e.message);log(`BLE ERROR: ${e.message}`);}});
noodleBLE.addEventListener('connected',e=>{$('connectBtn').textContent='Disconnect';$('bleBadge').textContent='Connected';$('bleBadge').classList.remove('badge-muted');$('deviceStatus').textContent=`Connected: ${e.detail.name}`;$('supportNote').textContent='Mode 2 continuous six-axis stream active.';$('trainingModeBtn').disabled=false;$('inferenceModeBtn').disabled=false;$('modeStatus').textContent='Current mode: waiting for device status';$('armRecordBtn').disabled=!state.setupLocked;$('stopRecordBtn').disabled=true;$('deployBtn').disabled=!MODE2_DEPLOY_SUPPORTED||!state.pkg;log(`BLE connected to ${e.detail.name} · Mode 2 continuous stream.`);});
noodleBLE.addEventListener('disconnected',()=>{$('connectBtn').textContent='Connect';$('bleBadge').textContent='Disconnected';$('bleBadge').classList.add('badge-muted');$('deviceStatus').textContent='Disconnected';$('supportNote').textContent=NoodleAIBLE.supportMessage();$('trainingModeBtn').disabled=true;$('inferenceModeBtn').disabled=true;$('modeStatus').textContent='Current mode: MODE 2 STREAMING';$('armRecordBtn').disabled=true;$('stopRecordBtn').disabled=true;state.recording=null;$('deployBtn').disabled=true;log('BLE disconnected.');});
noodleBLE.addEventListener('warning',e=>log(`BLE warning: ${e.detail.text}`));noodleBLE.addEventListener('deploy-log',e=>log(e.detail.text));

noodleBLE.addEventListener('imu',e=>{const s=e.detail.sample;$('accelStatus').textContent=`ax ${s.ax.toFixed(3)} g   ay ${s.ay.toFixed(3)} g   az ${s.az.toFixed(3)} g`;$('gyroStatus').textContent=`gx ${s.gx.toFixed(1)} °/s   gy ${s.gy.toFixed(1)} °/s   gz ${s.gz.toFixed(1)} °/s`;pushLive(s);if(state.recording){if(state.recordStartTms==null)state.recordStartTms=s.t_ms;state.recording.push([s.ax,s.ay,s.az,s.gx,s.gy,s.gz,s.t_ms]);const secs=(state.recording.length/NAI.SAMPLE_RATE_HZ).toFixed(1);$('recordProgress').textContent=`Recording “${state.labels[state.pendingLabelIndex]}”: ${state.recording.length} samples · ${secs} s — perform it repeatedly, then STOP.`;}});

noodleBLE.addEventListener('status',e=>{const text=e.detail.text;if(e.detail.notify)log(`Device: ${text}`);if(text==='MODE:T'){state.deviceMode='T';$('modeStatus').textContent='Current mode: TRAINING';$('deviceStatus').textContent='Training / streaming mode';}else if(text==='MODE:I'){state.deviceMode='I';$('modeStatus').textContent='Current mode: INFERENCE';$('deviceStatus').textContent='Inference mode — filling rolling window';}else if(text==='NO_MODEL'){$('deviceStatus').textContent='No committed model loaded';$('modeStatus').textContent='Current mode: TRAINING';state.deviceMode='T';}else if(text.startsWith('MODEL:READY:')){const p=text.split(':');const N=Number(p[2]),D=Number(p[3]),K=Number(p[4]),stride=Number(p[5]||5),thr=Number(p[6]||500)/1000;$('deployText').textContent=`Device INT8 model ready: ${N} samples · ${D} inputs · ${K} classes · threshold ${thr.toFixed(2)}.`;$('predictionMeta').textContent=`Device rolling window: ${N} samples (${(N/NAI.SAMPLE_RATE_HZ).toFixed(2)} s) · stride ${stride} samples (${(stride/NAI.SAMPLE_RATE_HZ).toFixed(2)} s) · accept confidence ≥ ${pct(thr)}`;}else if(text==='MODEL_STORED'){switchTab('deploy');$('deployText').textContent='MODEL_STORED — flash verified and committed. Loading model into Noodle…';}else if(text==='ERR:MODEL_LOAD'){$('deployText').textContent='Model was stored, but device-side Noodle loading failed.';}else if(text.startsWith('L:')){const p=text.split(':');if(p.length>=3){const idx=Number(p[1]);state.modelLabels[idx]=p.slice(2).join(':');}}else if(text.startsWith('P:')){const p=text.split(':');if(p.length>=3){const idx=Number(p[1]),conf=Number(p[2]);const label=state.modelLabels[idx]??state.labels[idx]??String(idx);$('predictionLabel').textContent=label;$('predictionConfidence').textContent=`Confidence ${pct(conf)} · accepted`;$('deviceStatus').textContent=`Inference: ${label} (${pct(conf)})`;}}else if(text.startsWith('U:')){const conf=Number(text.split(':')[1]);$('predictionLabel').textContent='UNKNOWN';$('predictionConfidence').textContent=`Confidence ${pct(conf)} · below model threshold`;$('deviceStatus').textContent=`Inference: UNKNOWN (${pct(conf)})`;}});

function pushLive(s){const L=250;state.live.t.push(s.t_ms/1000);const av=[s.ax,s.ay,s.az],gv=[s.gx,s.gy,s.gz];for(let c=0;c<3;c++){state.live.accel[c].push(av[c]);state.live.gyro[c].push(gv[c]);}if(state.live.t.length>L){state.live.t.shift();for(const a of state.live.accel)a.shift();for(const a of state.live.gyro)a.shift();}drawLineChart($('accelCanvas'),state.live.accel,['ax','ay','az']);drawLineChart($('gyroCanvas'),state.live.gyro,['gx','gy','gz']);}
$('clearPlotBtn').addEventListener('click',()=>{state.live={t:[],accel:[[],[],[]],gyro:[[],[],[]]};drawLiveEmpty();});

function drawLineChart(canvas,series,labels,{fixedY=null}={}){const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,pad={l:42,r:14,t:18,b:28};ctx.clearRect(0,0,W,H);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,W,H);const all=series.flat().filter(Number.isFinite);let lo=fixedY?fixedY[0]:(all.length?Math.min(...all):-1),hi=fixedY?fixedY[1]:(all.length?Math.max(...all):1);if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1;}if(!fixedY){const p=.12*(hi-lo);lo-=p;hi+=p;}ctx.strokeStyle='#e3e8ef';ctx.lineWidth=1;for(let k=0;k<=4;k++){const y=pad.t+k*(H-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}ctx.fillStyle='#758092';ctx.font='12px system-ui';ctx.textAlign='right';ctx.fillText(hi.toFixed(2),pad.l-6,pad.t+4);ctx.fillText(lo.toFixed(2),pad.l-6,H-pad.b);const colors=['#2869df','#00a37a','#d88a14','#8f5bd6'];series.forEach((a,c)=>{if(a.length<2)return;ctx.strokeStyle=colors[c%colors.length];ctx.lineWidth=1.8;ctx.beginPath();for(let i=0;i<a.length;i++){const x=pad.l+i*(W-pad.l-pad.r)/Math.max(1,a.length-1),y=pad.t+(hi-a[i])*(H-pad.t-pad.b)/(hi-lo);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();});ctx.textAlign='left';labels.forEach((l,i)=>{ctx.fillStyle=colors[i%colors.length];ctx.fillRect(pad.l+i*82,H-15,12,3);ctx.fillStyle='#596476';ctx.fillText(l,pad.l+17+i*82,H-10);});}
function drawLiveEmpty(){drawLineChart($('accelCanvas'),[[],[],[]],['ax','ay','az']);drawLineChart($('gyroCanvas'),[[],[],[]],['gx','gy','gz']);}

function drawTrainingCurves(h=state.history){if(!h||!h.epoch?.length){drawLineChart($('lossCanvas'),[[],[]],['train','validation']);drawLineChart($('accuracyCanvas'),[[],[]],['train','validation'],{fixedY:[0,1]});return;}drawLineChart($('lossCanvas'),[h.loss,h.valLoss],['train loss','validation loss']);drawLineChart($('accuracyCanvas'),[h.acc,h.valAcc],['train accuracy','validation accuracy'],{fixedY:[0,1]});}
function drawThresholdSweep(sw=state.thresholdSweep){if(!sw||!sw.rows?.length){drawLineChart($('thresholdCanvas'),[[],[]],['accepted accuracy','coverage'],{fixedY:[0,1]});return;}drawLineChart($('thresholdCanvas'),[sw.rows.map(r=>r.acceptedAccuracy),sw.rows.map(r=>r.coverage)],['accepted accuracy','coverage'],{fixedY:[0,1]});}

$('clearLogBtn').addEventListener('click',()=>$('log').textContent='');
$('supportNote').textContent=NoodleAIBLE.supportMessage();
refreshLabels();refreshDataset();lockUi(false);updateInputAndTopology();drawLiveEmpty();drawTrainingCurves();drawThresholdSweep();initTf();
