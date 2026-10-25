'use strict';
/* Paper-ready FP32/INT8 metrics for NoodleAI Mode 2.
 * Validation stays distinct from the separate, user-supplied Evaluate test set.
 */
(function(){
 const $=id=>document.getElementById(id);
 const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
 const score=x=>Number.isFinite(x)?Number(x).toFixed(3):'—';
 const percent=x=>Number.isFinite(x)?(100*x).toFixed(1)+'%':'—';
 let lastTest=null;
 function metricsHTML(m, title){
   const macro=m.macro, weighted=m.weighted;
   const cards=[['Accuracy',percent(m.accuracy),`${m.correct} / ${m.n} correct`],
      ['Macro precision',score(macro.precision),'unweighted'],['Macro recall',score(macro.recall),'unweighted'],
      ['Macro F1',score(macro.f1),'unweighted'],['Weighted F1',score(weighted.f1),'support-weighted']];
   let html=`<div class="nai-report-title">${esc(title)}</div><div class="nai-report-cards">`;
   for(const [label,value,desc] of cards)html+=`<div class="nai-metric"><span>${esc(label)}</span><strong>${value}</strong><small>${esc(desc)}</small></div>`;
   html+='</div><div class="nai-report-pair"><div class="nai-report-card"><h3>Per-class metrics</h3><table class="nai-report-table"><thead><tr><th>Class</th><th>Samples</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>';
   for(const r of m.perClass)html+=`<tr><td>${esc(r.label)}</td><td>${r.support}</td><td>${score(r.precision)}</td><td>${score(r.recall)}</td><td>${score(r.f1)}</td></tr>`;
   html+='</tbody></table></div><div class="nai-report-card"><h3>Confusion matrix</h3><p>Actual rows × predicted columns; before confidence rejection.</p><table class="nai-report-table nai-confusion"><thead><tr><th>Actual ↓ / Predicted →</th>';
   for(const l of m.labels)html+=`<th>${esc(l)}</th>`;
   html+='</tr></thead><tbody>';
   m.matrix.forEach((row,i)=>{html+=`<tr><th>${esc(m.labels[i])}</th>`;row.forEach((v,j)=>{html+=`<td class="${v?(i===j?'nai-true':'nai-false'):''}">${v}</td>`});html+='</tr>'});
   return html+'</tbody></table></div></div>';
 }
 function init(){
   const css=document.createElement('style');css.textContent=`
    .nai-paper-panel{margin-top:16px}.nai-report-title{font-size:1rem;font-weight:850;color:#253348;margin:10px 0 9px}
    .nai-report-cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}
    .nai-metric{background:#f6f8fb;border:1px solid #e1e7f1;border-radius:13px;padding:13px;min-width:0}
    .nai-metric span,.nai-metric small{display:block;font-size:.67rem;color:#65748b}
    .nai-metric strong{display:block;color:#233048;font-size:1.35rem;margin:6px 0}
    .nai-report-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0 20px}
    .nai-report-card{border:1px solid #e0e6ee;border-radius:14px;background:#fff;padding:13px;overflow:auto}
    .nai-report-card h3{font-size:.91rem;color:#293750;margin:0 0 10px}
    .nai-report-card p{font-size:.7rem;color:#65748b;margin:0 0 9px}
    .nai-report-table{width:100%;border-collapse:collapse;font-size:.72rem;color:#36435a}
    .nai-report-table th{background:#f2f5f9;color:#4c5870}
    .nai-report-table td,.nai-report-table th{padding:8px;border-bottom:1px solid #e9edf2;text-align:right;white-space:nowrap}
    .nai-report-table th:first-child,.nai-report-table td:first-child{text-align:left}
    .nai-report-table .nai-true{background:#e3f6ee;color:#167a56;font-weight:900}
    .nai-report-table .nai-false{background:#fff0ee;color:#ad3b37;font-weight:850}
    .nai-report-warning{font-size:.73rem;color:#77601d;margin:10px 0;line-height:1.65}
    .nai-report-divider{border:0;border-top:1px solid #e5eaf0;margin:24px 0 14px}
    @media(max-width:1080px){.nai-report-cards{grid-template-columns:repeat(3,1fr)}.nai-report-pair{grid-template-columns:1fr}}
    @media(max-width:650px){.nai-report-cards{grid-template-columns:repeat(2,1fr)}}`;
   document.head.appendChild(css);
   const tab=$('tab-curves');
   const p=document.createElement('section');p.className='panel nai-paper-panel';p.id='mode2ValidationPanel';p.hidden=true;
   p.innerHTML=`<div class="section-head"><div><div class="section-kicker">QUANTIZATION VALIDATION</div>
     <h2>Final-epoch FP32 and INT8 results</h2><p>Same held-out window-level validation split; INT8 calibration uses training windows only.</p></div>
     <button id="exportValidationPaperBtn" class="secondary">Export paper results (.zip)</button></div>
     <div id="mode2ValidationResults"></div><div id="mode2ValidationCaution" class="nai-report-warning"></div>`;
   tab.appendChild(p);
   $('exportValidationPaperBtn').addEventListener('click',()=>exportReport());
 }
 function renderValidation(report){
   const p=$('mode2ValidationPanel');if(!p)return;
   p.hidden=false;
   const agreement=report.fp32Int8Agreement;
   $('mode2ValidationResults').innerHTML=metricsHTML(report.validation,'FP32 · validation')+
    `<hr class="nai-report-divider">`+metricsHTML(report.int8Validation,'INT8 · post-training quantized validation')+
    `<p class="nai-report-warning"><strong>FP32/INT8 prediction agreement:</strong> ${agreement.matches} / ${agreement.n} (${percent(agreement.rate)}). These are browser-side validation measurements, not on-device accuracy.</p>`;
   const smallest=Math.min(...report.validation.perClass.map(x=>x.support));
   $('mode2ValidationCaution').textContent=`Window-level stratified validation (seed 42): ${report.train.n} train / ${report.validation.n} validation windows. Rolling windows may overlap across partitions; therefore these numbers should not be interpreted as session-independent test performance.${smallest<5?' Smallest class has only '+smallest+' validation windows.':''}`;
 }
 function reset(){lastTest=null;const p=$('mode2ValidationPanel');if(p){p.hidden=true;$('mode2ValidationResults').innerHTML='';}}
 function setTest(report){lastTest=report;}
 function getTest(){return lastTest;}
 function csv(rows){const cell=x=>{const s=String(x??'');return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s};return rows.map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n';}
 function rowsMetrics(split,m){return [
   [split,'samples',m.n],[split,'correct',m.correct],[split,'accuracy',m.accuracy],
   ...['macro','weighted','micro'].flatMap(avg=>['precision','recall','f1'].map(k=>[split,avg+'_'+k,m[avg][k]]))
 ];}
 function perClassCsv(m){return csv([['class_index','class','support','predicted','tp','precision','recall','f1'],...m.perClass.map(r=>[r.index,r.label,r.support,r.predicted,r.tp,r.precision,r.recall,r.f1])]);}
 function confusionCsv(m){return csv([['actual\\predicted',...m.labels],...m.matrix.map((row,i)=>[m.labels[i],...row])]);}
 function predictionsCsv(records,labels){return csv([
   ['dataset_index','actual_index','actual_class','fp32_predicted_index','fp32_predicted_class','int8_predicted_index','int8_predicted_class','agreement','fp32_correct','int8_correct',...labels.map((_,i)=>'fp32_probability_'+i),...labels.map((_,i)=>'int8_probability_'+i)],
   ...records.map(r=>[r.datasetIndex,r.actual,labels[r.actual],r.fp32Pred,labels[r.fp32Pred],r.int8Pred,labels[r.int8Pred],Number(r.fp32Pred===r.int8Pred),Number(r.actual===r.fp32Pred),Number(r.actual===r.int8Pred),...r.fp32Probs,...r.int8Probs])
 ]);}
 async function exportReport(){
   if(!state.report){alert('Train the model before exporting paper results.');return;}
   const validation=state.report, test=getTest();
   const activeModel=state.pkg?.dims?.join(',');
   if(test&&test.modelSignature!==activeModel)throw new Error('Loaded test results belong to a different model. Re-evaluate before exporting.');
   const buttons=[$('exportValidationPaperBtn'),$('exportTestPaperBtn')].filter(Boolean);
   try{
     buttons.forEach(b=>b.disabled=true);
     const zip=new JSZip();
     const files=NoodleMetrics.buildReportFiles(validation);
     for(const [name,content] of Object.entries(files))zip.file('validation/'+name,content);
     if(validation.thresholdSweep){} // threshold data below is from the selected model
     if(state.pkg?.sweep?.rows?.length)zip.file('validation/threshold_sweep.csv',csv([['threshold','coverage','accepted_accuracy','accepted','correct'],...state.pkg.sweep.rows.map(r=>[r.threshold,r.coverage,r.acceptedAccuracy,r.accepted,r.correct])]));
     const valRecords=validation.validationPredictions.map((p,i)=>({datasetIndex:p.datasetIndex,actual:p.actual,fp32Pred:p.predicted,int8Pred:validation.int8ValidationPredictions[i].predicted,fp32Probs:p.probabilities,int8Probs:validation.int8ValidationPredictions[i].probabilities}));
     zip.file('validation/paired_predictions_fp32_int8.csv',predictionsCsv(valRecords,validation.metadata.labels));
     if(test){
       zip.file('test/metrics_summary.csv',csv([['split','metric','value'],...rowsMetrics('test_fp32',test.fp32),...rowsMetrics('test_int8',test.int8),['agreement','matches',test.agreement.matches],['agreement','rate',test.agreement.rate],['threshold','value',test.threshold],['threshold','coverage',test.thresholdMetrics.coverage],['threshold','accepted_accuracy',test.thresholdMetrics.acceptedAccuracy],['threshold','accepted',test.thresholdMetrics.accepted],['threshold','rejected',test.thresholdMetrics.rejected]]));
       zip.file('test/per_class_fp32.csv',perClassCsv(test.fp32));zip.file('test/per_class_int8.csv',perClassCsv(test.int8));
       zip.file('test/confusion_matrix_fp32.csv',confusionCsv(test.fp32));zip.file('test/confusion_matrix_int8.csv',confusionCsv(test.int8));
       zip.file('test/paired_predictions_fp32_int8.csv',predictionsCsv(test.pairedPredictions,test.int8.labels));
       zip.file('test/test_metadata.json',JSON.stringify({dataset:test.fileName,n:test.fp32.n,models:test.modelSignature,classes:test.int8.labels,threshold:test.threshold,createdUtc:test.createdUtc,independence:'User-supplied test set: separation from training is not programmatically established'},null,2)+'\n');
     }
     for(const [id,name] of [['lossCanvas','loss_vs_epoch.png'],['accuracyCanvas','accuracy_vs_epoch.png'],['thresholdCanvas','threshold_sweep.png']]){
       const canvas=$(id);if(canvas&&typeof canvas.toDataURL==='function'){
         const data=canvas.toDataURL('image/png');if(data.startsWith('data:image/png;base64,'))zip.file('figures/'+name,data.split(',')[1],{base64:true});
       }
     }
     zip.file('README.txt',`NoodleAI Mode 2 browser-generated research package\n\nvalidation/: final epoch, window-level stratified holdout and PTQ comparison.\ntest/: optional user-loaded separately collected test .npz, evaluated by the frozen model.\nfigures/: training and threshold curves, rendered in the browser.\n\nBoth FP32 and INT8 probabilities/predictions are from browser inference, not the physical BL702.\nINT8 activation calibration used training inputs; model threshold selected on validation.\nTest metrics use the frozen threshold and are not used to tune it.\nConfusion matrices count unrejected argmax predictions; coverage/accepted accuracy apply the confidence threshold separately.\nOverlapping rolling windows can leak temporal neighbors across a random train/validation split. Prefer session-disjoint or subject-disjoint testing for generalization claims.\nUser-loaded test-set independence is not automatically verified.\n`);
     const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE'});
     downloadBlob(blob,'noodleai_mode2_paper_results.zip');
     if(typeof log==='function')log(`Exported Mode 2 paper report: validation${test?' + separate test':''}.`);
   }catch(err){alert('Report export failed: '+err.message);if(typeof log==='function')log('Report export error: '+err.message);}
   finally{buttons.forEach(b=>b.disabled=false);}
 }
 init();window.NoodleMode2Paper=Object.freeze({metricsHTML,renderValidation,reset,setTest,getTest,exportReport});
})();
