'use strict';
// Classification metrics for held-out gesture evaluation. No TensorFlow dependency.
// Rows in the confusion matrix are true labels; columns are predicted labels.
(function () {
  function evaluate(yTrue, yPred, labels) {
    const K=labels.length;
    if (!K || yTrue.length!==yPred.length) throw new Error('Metrics require equal-length predictions and targets and at least one class.');
    const matrix=Array.from({length:K},()=>Array(K).fill(0));
    let correct=0;
    for(let i=0;i<yTrue.length;i++) {
      const a=Number(yTrue[i]),p=Number(yPred[i]);
      if(!Number.isInteger(a)||!Number.isInteger(p)||a<0||p<0||a>=K||p>=K)throw new Error('Invalid class index in metrics.');
      matrix[a][p]++;
      if(a===p)correct++;
    }
    const perClass=labels.map((label,c)=>{
      const support=matrix[c].reduce((a,b)=>a+b,0), tp=matrix[c][c];
      const predicted=matrix.reduce((s,row)=>s+row[c],0);
      const precision=predicted?tp/predicted:0;
      const recall=support?tp/support:0;
      const f1=(precision+recall)>0?2*precision*recall/(precision+recall):0;
      return {label:String(label),index:c,support,predicted,tp,precision,recall,f1};
    });
    const n=yTrue.length;
    const macro={precision:0,recall:0,f1:0};
    const weighted={precision:0,recall:0,f1:0};
    for(const row of perClass)for(const k of ['precision','recall','f1']){
      macro[k]+=row[k]/K;
      weighted[k]+=n?row[k]*row.support/n:0;
    }
    // For single-label multiclass classification, micro P/R/F1 equals accuracy.
    const accuracy=n?correct/n:0;
    const micro={precision:accuracy,recall:accuracy,f1:accuracy};
    return {n,correct,accuracy,macro,weighted,micro,perClass,matrix,labels:[...labels]};
  }

  // Match the existing seeded stratified partition whenever possible. Repair a
  // rare class with zero examples in either partition by swapping sample indices,
  // preserving the total split sizes and avoiding undefined per-class recall.
  function ensureClassCoverage(original,targets,K){
    const train=[...original.train],test=[...original.test];
    const trainCounts=Array(K).fill(0),testCounts=Array(K).fill(0);
    train.forEach(i=>trainCounts[targets[i]]++);
    test.forEach(i=>testCounts[targets[i]]++);
    let coverageAdjusted=false;
    for(let c=0;c<K;c++){
      if(testCounts[c]===0){
        const ti=train.findIndex(i=>targets[i]===c && trainCounts[c]>1);
        const vi=test.findIndex(i=>testCounts[targets[i]]>1);
        if(ti<0||vi<0)throw new Error('Unable to reserve a validation gesture for every class. Collect more examples per class.');
        const donor=targets[test[vi]];
        [train[ti],test[vi]]=[test[vi],train[ti]];
        trainCounts[c]--;testCounts[c]++;trainCounts[donor]++;testCounts[donor]--;
        coverageAdjusted=true;
      }
    }
    for(let c=0;c<K;c++){
      if(trainCounts[c]===0){
        const vi=test.findIndex(i=>targets[i]===c && testCounts[c]>1);
        const ti=train.findIndex(i=>trainCounts[targets[i]]>1);
        if(ti<0||vi<0)throw new Error('Unable to reserve a training gesture for every class. Collect more examples per class.');
        const donor=targets[train[ti]];
        [train[ti],test[vi]]=[test[vi],train[ti]];
        trainCounts[c]++;testCounts[c]--;trainCounts[donor]--;testCounts[donor]++;
        coverageAdjusted=true;
      }
    }
    if(trainCounts.some(n=>!n)||testCounts.some(n=>!n))throw new Error('Each class must be represented in both data partitions.');
    return {train,test,coverageAdjusted};
  }

  function cell(v) {
    const t=String(v??'');
    return /[",\r\n]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t;
  }
  function csv(rows){return rows.map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n';}
  function f(v){return Number(v).toFixed(6);}
  function perClassCsv(metrics){return csv([
    ['class_index','class','support','predicted','true_positive','precision','recall','f1'],
    ...metrics.perClass.map(r=>[r.index,r.label,r.support,r.predicted,r.tp,f(r.precision),f(r.recall),f(r.f1)])
  ]);}
  function confusionCsv(metrics){return csv([
    ['actual\\predicted',...metrics.labels],
    ...metrics.matrix.map((row,i)=>[metrics.labels[i],...row])
  ]);}
  function summaryRows(split,metrics){return [
    [split,'samples',metrics.n],
    [split,'correct',metrics.correct],
    [split,'accuracy',f(metrics.accuracy)],
    ...['macro','weighted','micro'].flatMap(avg=>['precision','recall','f1'].map(metric=>[split,`${avg}_${metric}`,f(metrics[avg][metric])]))
  ];}
  function buildReportFiles(report){
    const {train,validation,int8Validation,fp32Int8Agreement,int8ValidationPredictions,history,metadata,split,validationPredictions}=report;
    const paired=int8Validation&&int8ValidationPredictions&&fp32Int8Agreement;
    const int8Files=paired?{
      'per_class_int8_validation.csv':perClassCsv(int8Validation),
      'confusion_matrix_int8_validation.csv':confusionCsv(int8Validation),
      'validation_predictions_int8.csv':csv([['dataset_index','actual_index','actual_class','predicted_index','predicted_class','correct',...metadata.labels.map((_,i)=>`prob_class_${i}`)],
        ...int8ValidationPredictions.map(p=>[p.datasetIndex,p.actual,metadata.labels[p.actual],p.predicted,metadata.labels[p.predicted],p.actual===p.predicted?1:0,...p.probabilities.map(f)])]),
      'fp32_int8_comparison.csv':csv([['dataset_index','actual_index','fp32_predicted_index','int8_predicted_index','agree','fp32_correct','int8_correct'],
        ...validationPredictions.map((p,i)=>[p.datasetIndex,p.actual,p.predicted,int8ValidationPredictions[i].predicted,p.predicted===int8ValidationPredictions[i].predicted?1:0,p.actual===p.predicted?1:0,p.actual===int8ValidationPredictions[i].predicted?1:0])]),
    }:{};
    return {
      'metrics_summary.csv':csv([['split','metric','value'],...summaryRows('train',train),...summaryRows('validation',validation),
        ...(paired?[...summaryRows('validation_int8',int8Validation),['comparison_fp32_int8','class_agreement',f(fp32Int8Agreement.rate)],['comparison_fp32_int8','class_agreement_count',fp32Int8Agreement.matches]]:[])]),
      ...int8Files,
      'per_class_validation.csv':perClassCsv(validation),
      'per_class_train.csv':perClassCsv(train),
      'confusion_matrix_validation.csv':confusionCsv(validation),
      'confusion_matrix_train.csv':confusionCsv(train),
      'training_history.csv':csv([['epoch','train_loss','validation_loss','train_accuracy','validation_accuracy'],
        ...history.epoch.map((epoch,i)=>[epoch,f(history.loss[i]),f(history.valLoss[i]),f(history.acc[i]),f(history.valAcc[i])])]),
      'validation_predictions.csv':csv([['dataset_index','actual_index','actual_class','predicted_index','predicted_class','correct',...metadata.labels.map((_,i)=>`prob_class_${i}`)],
        ...validationPredictions.map(p=>[p.datasetIndex,p.actual,metadata.labels[p.actual],p.predicted,metadata.labels[p.predicted],p.actual===p.predicted?1:0,...p.probabilities.map(f)])]),
      'data_split.csv':csv([['dataset_index','split','class_index','class'],
        ...split.train.map(i=>[i,'train',metadata.targets[i],metadata.labels[metadata.targets[i]]]),
        ...split.test.map(i=>[i,'validation',metadata.targets[i],metadata.labels[metadata.targets[i]]])]),
      'run_metadata.json':JSON.stringify({...metadata,targets:undefined},null,2)+'\n',
      'README.txt':[
        'NoodleAI gesture classification report',
        'FP32 metrics computed from TensorFlow.js predictions after the final epoch.',
        'INT8 metrics, when included, come from post-training quantized Noodle-compatible JS prediction on the same validation samples.',
        'PTQ calibration uses only training-partition inputs. This is browser-side INT8 validation, not device-measured accuracy.',
        'Validation is a stratified holdout from the captured gesture dataset, NOT an independent test set.',
        'Scaler is fitted only on training samples. Split seed: 42.',
        'Accuracy = correct predictions / total.',
        'Precision(c) = TP(c)/(TP(c)+FP(c)); Recall(c) = TP(c)/(TP(c)+FN(c)).',
        'F1(c) = 2 * Precision(c) * Recall(c) / (Precision(c)+Recall(c)).',
        'Zero denominator produces 0. Macro is the unweighted mean over ALL defined classes.',
        'Weighted scores weight each class by its true-label support.',
        'In this single-label task, micro precision/recall/F1 all equal accuracy.',
        'Confusion matrix: rows = actual label, columns = predicted label.',
        'Missing validation support for a class makes its per-class recall unmeasurable: check supports.',
        'Reported results are the LAST epoch, not a best-epoch selection.',
        'Repeatedly tuning on the validation partition biases performance estimates.',
        'For a paper, use subject/session-independent test data or subject-wise cross-validation as applicable.',
      ].join('\n')+'\n'
    };
  }
  window.NoodleMetrics=Object.freeze({evaluate,ensureClassCoverage,buildReportFiles});
})();
