'use strict';

const SAMPLE_RATE_HZ = 50;
const MADGWICK_BETA = 0.10;
const GRAVITY_MPS2 = 9.80665;
const MAX_PACKAGE_BYTES = 550 * 1024;
const FLAG_BINARY_CLASSIFIER = 0x0001;
const FLAG_INT8 = 0x0002;
const QCFG_MAGIC = 'Q8M1';
const FILTER_MAGIC = 'FLT1';
const FILTER_HP = 0x0001;
const FILTER_LP = 0x0002;
const DEFAULT_HP_HZ = 0.10;
const DEFAULT_LP_HZ = 8.00;

const REP = {
  'accel': { code: 1, channels: 3 },
  'gyro': { code: 2, channels: 3 },
  'accel+gyro': { code: 3, channels: 6 },
  'quaternion': { code: 4, channels: 4 },
  'velocity': { code: 5, channels: 3 },
  'velocity+quaternion': { code: 6, channels: 7 },
};
const CODE_TO_REP = Object.fromEntries(Object.entries(REP).map(([k,v]) => [v.code, k]));
// ---------------------------------------------------------------------------
// NPY / NPZ reader: enough for NoodleAI .npz datasets
// ---------------------------------------------------------------------------

function product(shape) {
  return shape.length ? shape.reduce((a,b) => a*b, 1) : 1;
}

function parseShape(header) {
  const m = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
  if (!m) throw new Error('NPY shape not found');
  const text = m[1].trim();
  if (!text) return [];
  return text.split(',').map(s => s.trim()).filter(Boolean).map(Number);
}

function parseNpy(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  if (u8.length < 10 || u8[0] !== 0x93 ||
      String.fromCharCode(...u8.slice(1,6)) !== 'NUMPY') {
    throw new Error('Not an NPY file');
  }
  const major = u8[6], minor = u8[7];
  const dv = new DataView(arrayBuffer);
  let headerLen, pos;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    pos = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    pos = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.${minor}`);
  }
  const header = new TextDecoder('latin1').decode(u8.slice(pos, pos + headerLen));
  const descrM = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  const fortM = header.match(/['"]fortran_order['"]\s*:\s*(True|False)/);
  if (!descrM || !fortM) throw new Error('Malformed NPY header');
  if (fortM[1] === 'True') throw new Error('Fortran-order NPY arrays are not supported');
  const descr = descrM[1];
  const shape = parseShape(header);
  const count = product(shape);
  const dataOffset = pos + headerLen;
  const d = new DataView(arrayBuffer, dataOffset);

  let data;
  const endian = descr[0];
  const body = ['<','>','|','='].includes(endian) ? descr.slice(1) : descr;
  const little = endian !== '>';

  if (body === 'f4') {
    data = new Float32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getFloat32(i*4, little);
  } else if (body === 'f8') {
    data = new Float64Array(count);
    for (let i=0;i<count;i++) data[i] = d.getFloat64(i*8, little);
  } else if (body === 'i4') {
    data = new Int32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getInt32(i*4, little);
  } else if (body === 'u4') {
    data = new Uint32Array(count);
    for (let i=0;i<count;i++) data[i] = d.getUint32(i*4, little);
  } else if (body === 'i8') {
    data = new Array(count);
    for (let i=0;i<count;i++) data[i] = Number(d.getBigInt64(i*8, little));
  } else if (body === 'u8') {
    data = new Array(count);
    for (let i=0;i<count;i++) data[i] = Number(d.getBigUint64(i*8, little));
  } else if (body === 'i2') {
    data = new Int16Array(count);
    for (let i=0;i<count;i++) data[i] = d.getInt16(i*2, little);
  } else if (body === 'u2') {
    data = new Uint16Array(count);
    for (let i=0;i<count;i++) data[i] = d.getUint16(i*2, little);
  } else if (body === 'i1') {
    data = new Int8Array(arrayBuffer, dataOffset, count).slice();
  } else if (body === 'u1' || body === 'b1') {
    data = new Uint8Array(arrayBuffer, dataOffset, count).slice();
  } else if (body.startsWith('U')) {
    const chars = Number(body.slice(1));
    data = new Array(count);
    let off = 0;
    for (let i=0;i<count;i++) {
      let s = '';
      for (let j=0;j<chars;j++) {
        const cp = d.getUint32(off, little);
        off += 4;
        if (cp) s += String.fromCodePoint(cp);
      }
      data[i] = s;
    }
  } else if (body.startsWith('S')) {
    const width = Number(body.slice(1));
    data = new Array(count);
    let off = 0;
    for (let i=0;i<count;i++) {
      const bytes = new Uint8Array(arrayBuffer, dataOffset + off, width);
      off += width;
      let n = bytes.indexOf(0);
      if (n < 0) n = width;
      data[i] = new TextDecoder('utf-8').decode(bytes.slice(0,n));
    }
  } else {
    throw new Error(`Unsupported NPY dtype ${descr}`);
  }

  return { data, shape, descr };
}

async function loadNpz(file) {
  const zip = await JSZip.loadAsync(await file.arrayBuffer());
  const arrays = {};
  for (const [name, entry] of Object.entries(zip.files)) {
    if (entry.dir || !name.endsWith('.npy')) continue;
    const key = name.replace(/\.npy$/, '');
    arrays[key] = parseNpy(await entry.async('arraybuffer'));
  }
  return arrays;
}

function scalar(arrays, key, fallback=null) {
  if (!arrays[key]) return fallback;
  const d = arrays[key].data;
  return Array.isArray(d) || ArrayBuffer.isView(d) ? d[0] : d;
}

function rowFromFlat(flat, row, width) {
  return Float32Array.from(flat.subarray(row*width, (row+1)*width));
}

function parseDatasetArrays(a) {
  if (!a.X || !a.y || !a.labels) throw new Error('Dataset must contain X, y, and labels');
  const Xshape = a.X.shape;
  if (Xshape.length !== 2) throw new Error(`X must be 2-D; got shape ${Xshape}`);
  const n = Xshape[0], width = Xshape[1];

  const y = Int32Array.from(a.y.data, Number);
  if (y.length !== n) throw new Error('X/y row count mismatch');

  const labels = Array.from(a.labels.data, String);
  const N = Number(scalar(a, 'normalized_length', scalar(a, 'window_length', 0)));
  if (!N) throw new Error('Dataset lacks normalized_length/window_length');
  if (width !== N*6) throw new Error(`Expected stored six-axis X width ${N*6}; got ${width}`);

  const Xrows = Array.from({length:n}, (_,i) => rowFromFlat(a.X.data, i, width));

  let rawRows = null;
  if (a.raw_data && a.raw_offsets) {
    const rs = a.raw_data.shape;
    if (rs.length !== 2 || rs[1] !== 6) throw new Error('raw_data must have shape [T,6]');
    const offsets = Array.from(a.raw_offsets.data, Number);
    if (offsets.length !== n+1) throw new Error('raw_offsets must have n_samples+1 entries');
    rawRows = [];
    for (let i=0;i<n;i++) {
      const start = offsets[i], end = offsets[i+1];
      if (end-start < 2) throw new Error(`Raw gesture ${i} has fewer than 2 samples`);
      const row = new Float32Array((end-start)*6);
      for (let k=start;k<end;k++) {
        for (let c=0;c<6;c++) row[(k-start)*6+c] = a.raw_data.data[k*6+c];
      }
      rawRows.push({ data: row, length: end-start });
    }
  }

  const sampleRate = Number(scalar(a, 'sample_rate_hz', SAMPLE_RATE_HZ));
  return { Xrows, y, labels, N, sampleRate, rawRows, width, fileName: '' };
}


// ---------------------------------------------------------------------------
// Exact legacy NumPy RandomState MT19937 permutation and sklearn-style split
// ---------------------------------------------------------------------------

class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.index = 624;
    this.mt[0] = seed >>> 0;
    for (let i=1;i<624;i++) {
      const x = this.mt[i-1] ^ (this.mt[i-1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, x) + i) >>> 0;
    }
  }
  twist() {
    for (let i=0;i<624;i++) {
      const y = (this.mt[i] & 0x80000000) | (this.mt[(i+1)%624] & 0x7fffffff);
      this.mt[i] = this.mt[(i+397)%624] ^ (y >>> 1) ^ ((y & 1) ? 0x9908b0df : 0);
    }
    this.index = 0;
  }
  uint32() {
    if (this.index >= 624) this.twist();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  interval(max) {
    max >>>= 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4;
    mask |= mask >>> 8; mask |= mask >>> 16;
    let v;
    do { v = this.uint32() & mask; } while (v > max);
    return v >>> 0;
  }
  permutationArray(values) {
    const a = Array.from(values);
    for (let i=a.length-1;i>0;i--) {
      const j = this.interval(i);
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
}

function approximateMode(classCounts, nDraws, rng) {
  const total = classCounts.reduce((a,b)=>a+b,0);
  const continuous = classCounts.map(c => c/total*nDraws);
  const floored = continuous.map(Math.floor);
  let need = nDraws - floored.reduce((a,b)=>a+b,0);
  if (need > 0) {
    const remainder = continuous.map((v,i)=>v-floored[i]);
    const values = Array.from(new Set(remainder)).sort((a,b)=>b-a);
    for (const value of values) {
      const inds = [];
      for (let i=0;i<remainder.length;i++) {
        if (Math.abs(remainder[i]-value) < 1e-15) inds.push(i);
      }
      const addNow = Math.min(inds.length, need);
      const chosen = addNow === inds.length ? inds : rng.permutationArray(inds).slice(0, addNow);
      for (const i of chosen) floored[i] += 1;
      need -= addNow;
      if (need === 0) break;
    }
  }
  return floored;
}

function pythonRound(x) {
  const f = Math.floor(x), frac = x-f;
  if (frac < 0.5) return f;
  if (frac > 0.5) return f+1;
  return (f % 2 === 0) ? f : f+1;
}

function sklearnStratifiedSplit(y, nClasses, seed=42) {
  const n = y.length;
  const nTest = Math.max(nClasses, pythonRound(0.20*n));
  const nTrain = n - nTest;
  const byClass = Array.from({length:nClasses}, ()=>[]);
  for (let i=0;i<n;i++) byClass[y[i]].push(i);
  const counts = byClass.map(v=>v.length);
  if (Math.min(...counts) < 2) throw new Error('Every class needs at least 2 samples for stratification');

  const rng = new MT19937(seed);
  const n_i = approximateMode(counts, nTrain, rng);
  const remain = counts.map((c,i)=>c-n_i[i]);
  const t_i = approximateMode(remain, nTest, rng);

  let train = [], test = [];
  for (let c=0;c<nClasses;c++) {
    const perm = rng.permutationArray(Array.from({length:counts[c]}, (_,i)=>i));
    const idxs = perm.map(p=>byClass[c][p]);
    train.push(...idxs.slice(0, n_i[c]));
    test.push(...idxs.slice(n_i[c], n_i[c]+t_i[c]));
  }
  train = rng.permutationArray(train);
  test = rng.permutationArray(test);
  return { train, test };
}


// ---------------------------------------------------------------------------
// NAI4 motion representations — mirror the current Python/C++ implementation
// ---------------------------------------------------------------------------

function resample(data, rows, cols, targetN) {
  const out = new Float32Array(targetN*cols);
  const scale = (rows-1)/(targetN-1);
  for (let i=0;i<targetN;i++) {
    const pos = i*scale;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0+1, rows-1);
    const a = pos-i0;
    for (let c=0;c<cols;c++) {
      out[i*cols+c] = data[i0*cols+c] + a*(data[i1*cols+c]-data[i0*cols+c]);
    }
  }
  return out;
}

function centerColumns(data, rows, cols) {
  const out = Float32Array.from(data);
  for (let c=0;c<cols;c++) {
    let s=0;
    for (let r=0;r<rows;r++) s += out[r*cols+c];
    const m=s/rows;
    for (let r=0;r<rows;r++) out[r*cols+c] -= m;
  }
  return out;
}

function filterConfigForRepresentation(rep) {
  if (rep==='accel'||rep==='gyro'||rep==='accel+gyro') {
    return {flags:FILTER_HP|FILTER_LP, highpassHz:DEFAULT_HP_HZ, lowpassHz:DEFAULT_LP_HZ};
  }
  return {flags:FILTER_LP, highpassHz:0.0, lowpassHz:DEFAULT_LP_HZ};
}

function firstOrderFilterColumns(data, rows, cols, sampleRate, filter) {
  const out=new Float32Array(rows*cols);
  if (!rows || !cols) return out;
  const dt=1/sampleRate;
  const hpOn=!!(filter.flags&FILTER_HP);
  const lpOn=!!(filter.flags&FILTER_LP);
  const hpAlpha=hpOn ? (1/(2*Math.PI*filter.highpassHz))/((1/(2*Math.PI*filter.highpassHz))+dt) : 0;
  const lpBeta=lpOn ? dt/((1/(2*Math.PI*filter.lowpassHz))+dt) : 0;
  for(let c=0;c<cols;c++){
    let xPrev=data[c], hpPrev=0, lpPrev=0;
    let v=hpOn?0:xPrev;
    if(lpOn){lpPrev=v;v=lpPrev;}
    out[c]=v;
    for(let r=1;r<rows;r++){
      const x=data[r*cols+c];
      if(hpOn){const hp=hpAlpha*(hpPrev+x-xPrev);xPrev=x;hpPrev=hp;v=hp;}else v=x;
      if(lpOn){lpPrev += lpBeta*(v-lpPrev);v=lpPrev;}
      out[r*cols+c]=v;
    }
  }
  return out;
}

function qNorm(q) {
  const n = Math.hypot(q[0],q[1],q[2],q[3]);
  if (!Number.isFinite(n) || n < 1e-12) return [1,0,0,0];
  return q.map(v=>v/n);
}
function qConj(q) { return [q[0],-q[1],-q[2],-q[3]]; }
function qMul(a,b) {
  const [aw,ax,ay,az]=a, [bw,bx,by,bz]=b;
  return [
    aw*bw-ax*bx-ay*by-az*bz,
    aw*bx+ax*bw+ay*bz-az*by,
    aw*by-ax*bz+ay*bw+az*bx,
    aw*bz+ax*by-ay*bx+az*bw,
  ];
}
function qRotate(q,v) {
  return qMul(qMul(q,[0,v[0],v[1],v[2]]),qConj(q)).slice(1);
}
function cross(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a,b) { return a.reduce((s,v,i)=>s+v*b[i],0); }

function quatAlignAccelToWorldZ(acc) {
  let u = Array.from(acc, Number);
  let n = Math.hypot(...u);
  if (!Number.isFinite(n) || n < 1e-8) return [1,0,0,0];
  u=u.map(v=>v/n);
  const v=[0,0,1], d=dot(u,v);
  if (d < -0.999999) {
    const base=Math.abs(u[0])<0.9?[1,0,0]:[0,1,0];
    let axis=cross(u,base); n=Math.hypot(...axis); axis=axis.map(x=>x/n);
    return [0,...axis];
  }
  return qNorm([1+d,...cross(u,v)]);
}

function madgwickUpdateImu(q, accel, gyroDps, dt, beta=MADGWICK_BETA) {
  let [q1,q2,q3,q4]=q;
  const rad=Math.PI/180;
  const [gx,gy,gz]=gyroDps.map(v=>v*rad);
  let [ax,ay,az]=accel;
  let qdot=[
    0.5*(-q2*gx-q3*gy-q4*gz),
    0.5*( q1*gx+q3*gz-q4*gy),
    0.5*( q1*gy-q2*gz+q4*gx),
    0.5*( q1*gz+q2*gy-q3*gx),
  ];
  const an=Math.hypot(ax,ay,az);
  if (Number.isFinite(an) && an>1e-8) {
    ax/=an; ay/=an; az/=an;
    const _2q1=2*q1,_2q2=2*q2,_2q3=2*q3,_2q4=2*q4;
    const _4q1=4*q1,_4q2=4*q2,_4q3=4*q3,_8q2=8*q2,_8q3=8*q3;
    const q1q1=q1*q1,q2q2=q2*q2,q3q3=q3*q3,q4q4=q4*q4;
    let s=[
      _4q1*q3q3+_2q3*ax+_4q1*q2q2-_2q2*ay,
      _4q2*q4q4-_2q4*ax+4*q1q1*q2-_2q1*ay-_4q2+_8q2*q2q2+_8q2*q3q3+_4q2*az,
      4*q1q1*q3+_2q1*ax+_4q3*q4q4-_2q4*ay-_4q3+_8q3*q2q2+_8q3*q3q3+_4q3*az,
      4*q2q2*q4-_2q2*ax+4*q3q3*q4-_2q3*ay,
    ];
    const sn=Math.hypot(...s);
    if (Number.isFinite(sn) && sn>1e-12) {
      s=s.map(v=>v/sn);
      qdot=qdot.map((v,i)=>v-beta*s[i]);
    }
  }
  return qNorm(q.map((v,i)=>v+qdot[i]*dt));
}

function deriveQuatVelocity(rawObj, sampleRate=SAMPLE_RATE_HZ, filter=null) {
  const T=rawObj.length, dt=1/sampleRate;
  const raw=filter ? firstOrderFilterColumns(rawObj.data,T,6,sampleRate,filter) : rawObj.data;
  const get3=(k,o)=>[raw[k*6+o],raw[k*6+o+1],raw[k*6+o+2]];
  let qAbs=quatAlignAccelToWorldZ(get3(0,0));
  const qRef=qAbs.slice();
  const quat=new Float32Array(T*4);
  const vel=new Float32Array(T*3);
  quat.set([1,0,0,0],0);
  let aPrev=qRotate(qAbs,get3(0,0)).map((v,i)=>(v-(i===2?1:0))*GRAVITY_MPS2);

  for (let k=1;k<T;k++) {
    qAbs=madgwickUpdateImu(qAbs,get3(k,0),get3(k,3),dt);
    let qr=qNorm(qMul(qConj(qRef),qAbs));
    const prev=[quat[(k-1)*4],quat[(k-1)*4+1],quat[(k-1)*4+2],quat[(k-1)*4+3]];
    if (dot(qr,prev)<0) qr=qr.map(v=>-v);
    quat.set(qr,k*4);

    const aw=qRotate(qAbs,get3(k,0));
    const a=aw.map((v,i)=>(v-(i===2?1:0))*GRAVITY_MPS2);
    for (let c=0;c<3;c++) vel[k*3+c]=vel[(k-1)*3+c]+0.5*(aPrev[c]+a[c])*dt;
    aPrev=a;
  }
  const end=[vel[(T-1)*3],vel[(T-1)*3+1],vel[(T-1)*3+2]];
  for (let k=0;k<T;k++) {
    const a=k/(T-1);
    for (let c=0;c<3;c++) vel[k*3+c]-=a*end[c];
  }
  vel[0]=vel[1]=vel[2]=0;
  vel[(T-1)*3]=vel[(T-1)*3+1]=vel[(T-1)*3+2]=0;
  return {quat,vel,T};
}

function normalizeQuatRows(data, rows, offset=0, stride=4) {
  const out=Float32Array.from(data);
  for (let r=0;r<rows;r++) {
    const b=r*stride+offset;
    const q=qNorm([out[b],out[b+1],out[b+2],out[b+3]]);
    out[b]=q[0];out[b+1]=q[1];out[b+2]=q[2];out[b+3]=q[3];
  }
  return out;
}

function buildRepresentationOne(ds, sampleIndex, rep) {
  const N=ds.N;
  if (!REP[rep]) throw new Error(`Unknown representation ${rep}`);
  const filter=filterConfigForRepresentation(rep);

  if (ds.rawRows) {
    const rawObj=ds.rawRows[sampleIndex];
    /*
     * Device inference always owns exactly N chronological 50-Hz raw samples.
     * Normalize the browser raw window to the same geometry before filtering so
     * training and deployment execute the same signal path.
     */
    const rawN={data:resample(rawObj.data,rawObj.length,6,N),length:N};
    if (rep==='accel'||rep==='gyro'||rep==='accel+gyro') {
      const idx = rep==='accel' ? [0,1,2] : rep==='gyro' ? [3,4,5] : [0,1,2,3,4,5];
      const selected=new Float32Array(N*idx.length);
      for (let r=0;r<N;r++) for (let c=0;c<idx.length;c++) selected[r*idx.length+c]=rawN.data[r*6+idx[c]];
      return firstOrderFilterColumns(selected,N,idx.length,SAMPLE_RATE_HZ,filter);
    }
    const d=deriveQuatVelocity(rawN,SAMPLE_RATE_HZ,filter);
    if (rep==='quaternion') return normalizeQuatRows(d.quat,N);
    if (rep==='velocity') return d.vel;
    const both=new Float32Array(N*7);
    for (let r=0;r<N;r++) {
      both[r*7]=d.vel[r*3]; both[r*7+1]=d.vel[r*3+1]; both[r*7+2]=d.vel[r*3+2];
      both[r*7+3]=d.quat[r*4]; both[r*7+4]=d.quat[r*4+1];
      both[r*7+5]=d.quat[r*4+2]; both[r*7+6]=d.quat[r*4+3];
    }
    return normalizeQuatRows(both,N,3,7);
  }

  // Legacy datasets have only fixed N x 6 rows. We can still apply the same
  // new direct band-pass path; derived motion representations require raw data.
  if (rep==='quaternion'||rep==='velocity'||rep==='velocity+quaternion') {
    throw new Error('This dataset has no raw_data/raw_offsets. Derived quaternion/velocity representations require a fresh NAI4 raw dataset.');
  }
  const src=ds.Xrows[sampleIndex];
  const idx = rep==='accel' ? [0,1,2] : rep==='gyro' ? [3,4,5] : [0,1,2,3,4,5];
  const selected=new Float32Array(N*idx.length);
  for (let r=0;r<N;r++) for (let c=0;c<idx.length;c++) selected[r*idx.length+c]=src[r*6+idx[c]];
  return firstOrderFilterColumns(selected,N,idx.length,SAMPLE_RATE_HZ,filter);
}

function buildRepresentationDataset(ds, rep) {
  return Array.from({length:ds.y.length},(_,i)=>buildRepresentationOne(ds,i,rep));
}


// ---------------------------------------------------------------------------
// StandardScaler exactly in the spirit of sklearn: train only, ddof=0
// ---------------------------------------------------------------------------

function fitScaler(rows, trainIdx) {
  const D=rows[0].length, n=trainIdx.length;
  const mean=new Float64Array(D), variance=new Float64Array(D);
  for (const i of trainIdx) {
    const x=rows[i];
    for (let j=0;j<D;j++) mean[j]+=x[j];
  }
  for (let j=0;j<D;j++) mean[j]/=n;
  for (const i of trainIdx) {
    const x=rows[i];
    for (let j=0;j<D;j++) {
      const d=x[j]-mean[j];
      variance[j]+=d*d;
    }
  }
  const scale=new Float64Array(D);
  for (let j=0;j<D;j++) {
    variance[j]/=n;
    const s=Math.sqrt(variance[j]);
    scale[j]=(Number.isFinite(s)&&s>1e-12)?s:1.0;
  }
  return {mean,scale};
}

function standardizeRows(rows, indices, scaler) {
  const D=rows[0].length;
  return indices.map(i=>{
    const out=new Float32Array(D), x=rows[i];
    for (let j=0;j<D;j++) out[j]=(x[j]-scaler.mean[j])/scaler.scale[j];
    return out;
  });
}

function flattenRows(rows) {
  const n=rows.length,D=rows[0].length,out=new Float32Array(n*D);
  rows.forEach((r,i)=>out.set(r,i*D));
  return out;
}



function makeModel(inputDim, hidden, nClasses, nTrain) {
  const model=tf.sequential();
  const alpha=0.0001; // sklearn MLPClassifier default
  const l2=alpha/(2*nTrain); // match sklearn's alpha/(2*n_samples) loss term

  hidden.forEach((units,i)=>{
    model.add(tf.layers.dense({
      inputShape: i===0 ? [inputDim] : undefined,
      units,
      activation:'relu',
      kernelInitializer: tf.initializers.glorotUniform({seed:42+i}),
      biasInitializer:'zeros',
      kernelRegularizer: tf.regularizers.l2({l2}),
    }));
  });

  if (nClasses===2) {
    model.add(tf.layers.dense({
      units:1, activation:'sigmoid',
      kernelInitializer:tf.initializers.glorotUniform({seed:42+hidden.length}),
      biasInitializer:'zeros',
      kernelRegularizer:tf.regularizers.l2({l2}),
    }));
    model.compile({
      optimizer:tf.train.adam(0.001,0.9,0.999,1e-8),
      loss:'binaryCrossentropy',
      metrics:['accuracy'],
    });
  } else {
    model.add(tf.layers.dense({
      units:nClasses, activation:'softmax',
      kernelInitializer:tf.initializers.glorotUniform({seed:42+hidden.length}),
      biasInitializer:'zeros',
      kernelRegularizer:tf.regularizers.l2({l2}),
    }));
    model.compile({
      optimizer:tf.train.adam(0.001,0.9,0.999,1e-8),
      loss:'categoricalCrossentropy',
      metrics:['accuracy'],
    });
  }
  return model;
}

function argmaxRow(prob) {
  let bi=0,bv=prob[0];
  for (let i=1;i<prob.length;i++) if (prob[i]>bv){bv=prob[i];bi=i;}
  return bi;
}
function accuracy(pred,y){let n=0;for(let i=0;i<y.length;i++)if(pred[i]===y[i])n++;return n/y.length;}

async function predictTf(model, standardizedRows, nClasses) {
  const n=standardizedRows.length,D=standardizedRows[0].length;
  const xs=tf.tensor2d(flattenRows(standardizedRows),[n,D],'float32');
  const out=model.predict(xs);
  const raw=await out.data();
  xs.dispose(); out.dispose();
  const probs=[], pred=[];
  if (nClasses===2) {
    for(let i=0;i<n;i++){const p=raw[i];probs.push([1-p,p]);pred.push(p>=0.5?1:0);}
  } else {
    for(let i=0;i<n;i++){const p=Array.from(raw.slice(i*nClasses,(i+1)*nClasses));probs.push(p);pred.push(argmaxRow(p));}
  }
  return {probs,pred};
}


// ---------------------------------------------------------------------------
// Binary NAI4 writer / reader + Mode 2 INT8 PTQ
// ---------------------------------------------------------------------------

function float32LE(values) {
  const out=new ArrayBuffer(values.length*4),dv=new DataView(out);
  for(let i=0;i<values.length;i++)dv.setFloat32(i*4,Number(values[i]),true);
  return new Uint8Array(out);
}
function readFloat32LE(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), out=new Float32Array(bytes.byteLength/4);
  for(let i=0;i<out.length;i++)out[i]=dv.getFloat32(i*4,true);
  return out;
}
function int32LE(values) {
  const out=new ArrayBuffer(values.length*4),dv=new DataView(out);
  for(let i=0;i<values.length;i++)dv.setInt32(i*4,Number(values[i]),true);
  return new Uint8Array(out);
}
function readInt32LE(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength), out=new Int32Array(bytes.byteLength/4);
  for(let i=0;i<out.length;i++)out[i]=dv.getInt32(i*4,true);
  return out;
}
function int8Bytes(values) {
  const out=new Uint8Array(values.length);
  for(let i=0;i<values.length;i++)out[i]=Number(values[i])&0xff;
  return out;
}
function readInt8(bytes) {
  const out=new Int8Array(bytes.length);
  for(let i=0;i<bytes.length;i++)out[i]=bytes[i]>127?bytes[i]-256:bytes[i];
  return out;
}

function roundEven(x) {
  if(!Number.isFinite(x)) return 0;
  const f=Math.floor(x), r=x-f;
  if(r<0.5)return f;
  if(r>0.5)return f+1;
  return (f%2===0)?f:f+1;
}
function clampI8(x){return x>127?127:(x<-128?-128:x);}
function clampI32(x){return x>2147483647?2147483647:(x<-2147483648?-2147483648:Math.trunc(x));}
function qFloat(x,scale,zp){return clampI8(roundEven(x/scale)+zp);}

// Mirrors noodle_quantize_multiplier() in noodle_int8_math.cpp.
function quantizeMultiplier(realMultiplier) {
  if(!(realMultiplier>=0)||!Number.isFinite(realMultiplier))throw new Error('Invalid quantized multiplier');
  if(realMultiplier===0)return {multiplier:0,shift:0};
  let exp=Math.floor(Math.log2(realMultiplier))+1;
  let q=realMultiplier/Math.pow(2,exp); // [0.5,1)
  let qFixed=Math.round(q*2147483648);
  if(qFixed===2147483648){qFixed/=2;exp+=1;}
  if(exp<-31)return {multiplier:0,shift:0};
  if(exp>30)throw new Error('Quantized multiplier exponent overflow');
  return {multiplier:qFixed,shift:exp};
}
function satRoundHighMul(a,b) {
  const ab=BigInt(Math.trunc(a))*BigInt(Math.trunc(b));
  const one=1n<<30n;
  const nudge=ab>=0n?one:(1n-one);
  return Number((ab+nudge)/(1n<<31n));
}
function roundingDivideByPot(x,exp) {
  if(exp<=0)return Math.trunc(x);
  if(exp>31)return x<0?-1:0;
  const xb=BigInt(Math.trunc(x));
  const mask=(1n<<BigInt(exp))-1n;
  const remainder=xb&mask;
  const threshold=(mask>>1n)+(x<0?1n:0n);
  return Number((xb>>BigInt(exp))+(remainder>threshold?1n:0n));
}
function multiplyByQuantizedMultiplier(x,multiplier,shift) {
  const left=shift>0?shift:0, right=shift>0?0:-shift;
  let shifted;
  if(left>=31)shifted=x>=0?2147483647:-2147483648;
  else shifted=clampI32(Math.trunc(x)*Math.pow(2,left));
  const high=satRoundHighMul(shifted,multiplier);
  return roundingDivideByPot(high,right);
}
function applyRequant(acc,mult,shift,zp,amin=-128,amax=127) {
  let q=multiplyByQuantizedMultiplier(acc,mult,shift)+zp;
  if(q<amin)q=amin;if(q>amax)q=amax;
  return clampI8(q);
}

function makeCfg(labels,N,inputDim,dims,rep,binary,quant=null,filter=null) {
  const enc=new TextEncoder();
  const labelBytes=labels.map(s=>enc.encode(String(s)));
  for(const b of labelBytes)if(!b.length||b.length>31)throw new Error('NAI labels must be 1..31 UTF-8 bytes');
  const quantBytes=quant ? (28 + 8*(dims.length-1)) : 0;
  const filterBytes=filter ? 16 : 0;
  const total=20 + 2*dims.length + labelBytes.reduce((s,b)=>s+1+b.length,0) + quantBytes + filterBytes;
  const out=new Uint8Array(total),dv=new DataView(out.buffer);
  out.set([0x4e,0x41,0x49,0x34],0); // NAI4
  let o=4;
  const flags=(binary?FLAG_BINARY_CLASSIFIER:0)|(quant?FLAG_INT8:0);
  const vals=[4,flags,N,SAMPLE_RATE_HZ,dims.length-1,labels.length,inputDim,REP[rep].code];
  vals.forEach(v=>{dv.setUint16(o,v,true);o+=2;});
  dims.forEach(v=>{dv.setUint16(o,v,true);o+=2;});
  for(const b of labelBytes){out[o++]=b.length;out.set(b,o);o+=b.length;}
  if(quant){
    out.set(new TextEncoder().encode(QCFG_MAGIC),o);o+=4;
    dv.setUint16(o,quant.strideSamples,true);o+=2;
    dv.setUint16(o,0,true);o+=2;
    dv.setFloat32(o,quant.thresholdMin,true);o+=4;
    dv.setFloat32(o,quant.thresholdMax,true);o+=4;
    dv.setFloat32(o,quant.confidenceThreshold,true);o+=4;
    dv.setFloat32(o,quant.inputScale,true);o+=4;
    dv.setInt32(o,quant.inputZeroPoint,true);o+=4;
    quant.layers.forEach(q=>{
      dv.setFloat32(o,q.outputScale,true);o+=4;
      dv.setInt32(o,q.outputZeroPoint,true);o+=4;
    });
  }
  if(filter){
    out.set(new TextEncoder().encode(FILTER_MAGIC),o);o+=4;
    dv.setUint16(o,filter.flags,true);o+=2;
    dv.setUint16(o,0,true);o+=2;
    dv.setFloat32(o,filter.highpassHz,true);o+=4;
    dv.setFloat32(o,filter.lowpassHz,true);o+=4;
  }
  return out;
}

async function extractTfWeightsOI(model) {
  const layers=[];
  for(const layer of model.layers) {
    const wb=layer.getWeights();
    if(wb.length!==2)throw new Error('Expected Dense layer kernel+bias');
    const [kernel,bias]=wb;
    const ks=kernel.shape; // [I,O]
    const I=ks[0],O=ks[1];
    const kd=await kernel.data(),bd=await bias.data();
    const oi=new Float32Array(O*I);
    for(let i=0;i<I;i++)for(let o=0;o<O;o++)oi[o*I+i]=kd[i*O+o];
    layers.push({I,O,w:oi,b:Float32Array.from(bd)});
  }
  return layers;
}

function maxAbsRows(rows){
  let m=0;
  for(const r of rows)for(let i=0;i<r.length;i++){const a=Math.abs(r[i]);if(a>m)m=a;}
  return m;
}
function maxRows(rows){
  let m=0;
  for(const r of rows)for(let i=0;i<r.length;i++)if(r[i]>m)m=r[i];
  return m;
}
function denseFloatRows(rows,layer,relu){
  const out=new Array(rows.length);
  for(let n=0;n<rows.length;n++){
    const a=rows[n],z=new Float32Array(layer.O);
    for(let o=0;o<layer.O;o++){
      let s=layer.b[o],base=o*layer.I;
      for(let i=0;i<layer.I;i++)s+=layer.w[base+i]*a[i];
      z[o]=relu&&s<0?0:s;
    }
    out[n]=z;
  }
  return out;
}

function calibrateActivationQParams(floatLayers,calibrationRows){
  if(!calibrationRows.length)throw new Error('INT8 calibration needs training rows');
  const activations=[];
  const inMax=maxAbsRows(calibrationRows);
  activations.push({scale:Math.max(inMax/127,1e-12),zeroPoint:0});
  let rows=calibrationRows;
  for(let li=0;li<floatLayers.length;li++){
    const hidden=li<floatLayers.length-1;
    rows=denseFloatRows(rows,floatLayers[li],hidden);
    if(hidden){
      const hi=maxRows(rows);
      activations.push({scale:Math.max(hi/255,1e-12),zeroPoint:-128});
    }else{
      const ma=maxAbsRows(rows);
      activations.push({scale:Math.max(ma/127,1e-12),zeroPoint:0});
    }
  }
  return activations;
}

function quantizeDenseLayers(floatLayers,activations){
  return floatLayers.map((l,li)=>{
    const qW=new Int8Array(l.O*l.I),qB=new Int32Array(l.O),mult=new Int32Array(l.O),shift=new Int32Array(l.O);
    const weightScale=new Float64Array(l.O);
    const inScale=activations[li].scale,outScale=activations[li+1].scale;
    for(let o=0;o<l.O;o++){
      let ma=0,base=o*l.I;
      for(let i=0;i<l.I;i++){const a=Math.abs(l.w[base+i]);if(a>ma)ma=a;}
      const ws=Math.max(ma/127,1e-12);weightScale[o]=ws;
      for(let i=0;i<l.I;i++)qW[base+i]=clampI8(roundEven(l.w[base+i]/ws));
      qB[o]=clampI32(roundEven(l.b[o]/(inScale*ws)));
      const qm=quantizeMultiplier(inScale*ws/outScale);
      mult[o]=qm.multiplier;shift[o]=qm.shift;
    }
    return {I:l.I,O:l.O,w:qW,b:qB,mult,shift,
      inputScale:activations[li].scale,inputZeroPoint:activations[li].zeroPoint,
      outputScale:activations[li+1].scale,outputZeroPoint:activations[li+1].zeroPoint};
  });
}

function noodleBinaryProbability(q,scale,zp){
  const x=(q-zp)*scale;
  const y=x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const pi=Math.max(0,Math.min(255,Math.round(y*256)));
  return pi/256;
}
function noodleSoftmaxProbabilities(qrow,scale){
  let maxq=qrow[0];for(let i=1;i<qrow.length;i++)if(qrow[i]>maxq)maxq=qrow[i];
  const lut=new Uint16Array(256);
  for(let d=0;d<256;d++)lut[d]=Math.max(0,Math.min(32767,Math.round(Math.exp(-d*scale)*32767)));
  let sum=0;const e=new Uint16Array(qrow.length);
  for(let i=0;i<qrow.length;i++){const d=(maxq-qrow[i])&255;e[i]=lut[d];sum+=e[i];}
  const p=new Array(qrow.length);
  for(let i=0;i<qrow.length;i++){let pi=Math.floor((e[i]*256+Math.floor(sum/2))/sum);if(pi>255)pi=255;p[i]=pi/256;}
  return p;
}

function predictQuantized(qmodel,standardizedRows,nClasses){
  const probs=[],pred=[];
  for(const row of standardizedRows){
    let a=new Int16Array(row.length);
    const aq=qmodel.activations[0];
    for(let i=0;i<row.length;i++)a[i]=qFloat(row[i],aq.scale,aq.zeroPoint);
    qmodel.layers.forEach((l,li)=>{
      const z=new Int16Array(l.O),hidden=li<qmodel.layers.length-1;
      const amin=hidden?l.outputZeroPoint:-128;
      for(let o=0;o<l.O;o++){
        let acc=l.b[o],base=o*l.I;
        for(let i=0;i<l.I;i++)acc+=(a[i]-l.inputZeroPoint)*l.w[base+i];
        z[o]=applyRequant(clampI32(acc),l.mult[o],l.shift[o],l.outputZeroPoint,amin,127);
      }
      a=z;
    });
    let p;
    const final=qmodel.activations[qmodel.activations.length-1];
    if(nClasses===2){const p1=noodleBinaryProbability(a[0],final.scale,final.zeroPoint);p=[1-p1,p1];}
    else p=noodleSoftmaxProbabilities(a,final.scale);
    probs.push(p);pred.push(argmaxRow(p));
  }
  return {probs,pred};
}

function sweepConfidenceThreshold(probs,y,{min=0.30,max=0.90,step=0.01,minCoverage=0.90}={}){
  const rows=[];
  const pred=probs.map(argmaxRow),conf=probs.map(p=>Math.max(...p));
  const n=y.length;
  for(let k=0;;k++){
    const t=Number((min+k*step).toFixed(6));if(t>max+1e-9)break;
    let accepted=0,correct=0;
    for(let i=0;i<n;i++)if(conf[i]>=t){accepted++;if(pred[i]===y[i])correct++;}
    rows.push({threshold:t,coverage:accepted/n,acceptedAccuracy:accepted?correct/accepted:0,accepted,correct});
  }
  const eligible=rows.filter(r=>r.coverage+1e-12>=minCoverage&&r.accepted>0);
  const pool=eligible.length?eligible:rows.filter(r=>r.accepted>0);
  let selected=pool[0];
  for(const r of pool.slice(1)){
    if(r.acceptedAccuracy>selected.acceptedAccuracy+1e-12 ||
       (Math.abs(r.acceptedAccuracy-selected.acceptedAccuracy)<=1e-12 && r.threshold>selected.threshold))selected=r;
  }
  return {rows,selected,min,max,step,minCoverage};
}

async function exportNai4Int8(model,scaler,labels,N,rep,calibrationRows,validationRows,validationY,strideSamples=5) {
  const floatLayers=await extractTfWeightsOI(model);
  if(floatLayers.length>4)throw new Error('Current single-slot INT8 deployment supports at most 4 Dense layers (3 hidden + output): 3 base files + 4 files per layer must fit the 19-file slot header.');
  const dims=[floatLayers[0].I,...floatLayers.map(l=>l.O)];
  const binary=labels.length===2;
  const activations=calibrateActivationQParams(floatLayers,calibrationRows);
  const qLayers=quantizeDenseLayers(floatLayers,activations);
  const qmodel={activations,layers:qLayers};
  const qva=predictQuantized(qmodel,validationRows,labels.length);
  const qValAcc=accuracy(qva.pred,Array.from(validationY));
  const sweep=sweepConfidenceThreshold(qva.probs,Array.from(validationY),{min:0.30,max:0.90,step:0.01,minCoverage:0.90});
  const filter=filterConfigForRepresentation(rep);
  const quant={
    strideSamples,
    thresholdMin:sweep.min,
    thresholdMax:sweep.max,
    confidenceThreshold:sweep.selected.threshold,
    inputScale:activations[0].scale,
    inputZeroPoint:activations[0].zeroPoint,
    layers:qLayers.map(l=>({outputScale:l.outputScale,outputZeroPoint:l.outputZeroPoint})),
  };
  const files={
    'cfg.bin':makeCfg(labels,N,dims[0],dims,rep,binary,quant,filter),
    'mean.bin':float32LE(scaler.mean),
    'scale.bin':float32LE(scaler.scale),
  };
  qLayers.forEach((l,i)=>{
    const n=String(i).padStart(2,'0');
    files[`w${n}.bin`]=int8Bytes(l.w);
    files[`b${n}.bin`]=int32LE(l.b);
    files[`m${n}.bin`]=int32LE(l.mult);
    files[`s${n}.bin`]=int32LE(l.shift);
  });
  const total=Object.values(files).reduce((s,b)=>s+b.byteLength,0);
  if(total>MAX_PACKAGE_BYTES)throw new Error(`Raw NAI4 package ${(total/1024).toFixed(1)} KiB exceeds 550 KiB`);
  const zip=new JSZip();Object.entries(files).forEach(([name,bytes])=>zip.file(name,bytes));
  const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE'});
  return {files,total,blob,dims,binary,rep,labels,N,scaler,int8:true,quant,filter,qmodel,qValAcc,sweep};
}

function parseCfg(bytes) {
  const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const magic=String.fromCharCode(...bytes.slice(0,4));
  let o=4;
  const v=[];for(let i=0;i<8;i++){v.push(dv.getUint16(o,true));o+=2;}
  const [version,flags,N,rate,nLayers,nClasses,inputDim,reserved]=v;
  if(!['NAI2','NAI3','NAI4'].includes(magic))throw new Error(`Unsupported NAI magic ${magic}`);
  let rep;if(magic==='NAI2')rep='accel';else rep=CODE_TO_REP[reserved]||null;
  if(!rep)throw new Error(`Unknown representation code ${reserved}`);
  const dims=[];for(let i=0;i<nLayers+1;i++){dims.push(dv.getUint16(o,true));o+=2;}
  const dec=new TextDecoder('utf-8'),labels=[];
  for(let i=0;i<nClasses;i++){const n=bytes[o++];labels.push(dec.decode(bytes.slice(o,o+n)));o+=n;}
  const int8=!!(flags&FLAG_INT8);let quant=null;
  if(int8){
    if(o+28+8*nLayers>bytes.length)throw new Error('Truncated NAI4 INT8 config');
    const qmagic=new TextDecoder().decode(bytes.slice(o,o+4));o+=4;
    if(qmagic!==QCFG_MAGIC)throw new Error(`Unknown INT8 config ${qmagic}`);
    const strideSamples=dv.getUint16(o,true);o+=2;o+=2;
    const thresholdMin=dv.getFloat32(o,true);o+=4;
    const thresholdMax=dv.getFloat32(o,true);o+=4;
    const confidenceThreshold=dv.getFloat32(o,true);o+=4;
    const inputScale=dv.getFloat32(o,true);o+=4;
    const inputZeroPoint=dv.getInt32(o,true);o+=4;
    const layers=[];for(let i=0;i<nLayers;i++){const outputScale=dv.getFloat32(o,true);o+=4;const outputZeroPoint=dv.getInt32(o,true);o+=4;layers.push({outputScale,outputZeroPoint});}
    quant={strideSamples,thresholdMin,thresholdMax,confidenceThreshold,inputScale,inputZeroPoint,layers};
  }
  let filter=null;
  if(o<bytes.length){
    if(o+16!==bytes.length)throw new Error(`Unexpected cfg.bin trailing bytes (${bytes.length-o})`);
    const fmagic=new TextDecoder().decode(bytes.slice(o,o+4));o+=4;
    if(fmagic!==FILTER_MAGIC)throw new Error(`Unknown filter config ${fmagic}`);
    const fflags=dv.getUint16(o,true);o+=2;o+=2;
    const highpassHz=dv.getFloat32(o,true);o+=4;
    const lowpassHz=dv.getFloat32(o,true);o+=4;
    filter={flags:fflags,highpassHz,lowpassHz};
  }
  if(o!==bytes.length)throw new Error(`Unexpected cfg.bin trailing bytes (${bytes.length-o})`);
  return {magic,version,flags,N,rate,nLayers,nClasses,inputDim,rep,dims,labels,binary:!!(flags&FLAG_BINARY_CLASSIFIER),int8,quant,filter};
}

async function loadNai(file) {
  const zip=await JSZip.loadAsync(await file.arrayBuffer());
  const get=async name=>{const e=zip.file(name);if(!e)throw new Error(`NAI missing ${name}`);return new Uint8Array(await e.async('arraybuffer'));};
  const cfg=parseCfg(await get('cfg.bin'));
  if(cfg.int8)throw new Error('Browser loadNai() float predictor does not open INT8 packages; use loadNaiPackage() for deployment.');
  const mean=readFloat32LE(await get('mean.bin')),scale=readFloat32LE(await get('scale.bin'));
  const layers=[];
  for(let i=0;i<cfg.nLayers;i++){
    const I=cfg.dims[i],O=cfg.dims[i+1];
    const w=readFloat32LE(await get(`w${String(i).padStart(2,'0')}.bin`));
    const b=readFloat32LE(await get(`b${String(i).padStart(2,'0')}.bin`));
    if(w.length!==I*O||b.length!==O)throw new Error(`Layer ${i} byte count mismatch`);
    layers.push({I,O,w,b});
  }
  return {...cfg,mean,scale,layers,fileName:file.name};
}

async function loadNaiPackage(file) {
  const zip=await JSZip.loadAsync(await file.arrayBuffer());
  const get=async name=>{const e=zip.file(name);if(!e)throw new Error(`NAI missing ${name}`);return new Uint8Array(await e.async('arraybuffer'));};
  const cfgBytes=await get('cfg.bin'),cfg=parseCfg(cfgBytes);
  const names=['cfg.bin','mean.bin','scale.bin'];
  for(let i=0;i<cfg.nLayers;i++){
    const n=String(i).padStart(2,'0');names.push(`w${n}.bin`,`b${n}.bin`);if(cfg.int8)names.push(`m${n}.bin`,`s${n}.bin`);
  }
  const files={};for(const name of names)files[name]=await get(name);
  const total=Object.values(files).reduce((s,b)=>s+b.byteLength,0);
  return {files,total,blob:file,rep:cfg.rep,labels:[...cfg.labels],N:cfg.N,inputDim:cfg.inputDim,dims:[...cfg.dims],binary:cfg.binary,int8:cfg.int8,quant:cfg.quant,filter:cfg.filter,cfg};
}

function predictNai(nai, rows) {
  const probs=[],pred=[];
  for(const raw of rows){
    let a=new Float64Array(nai.inputDim);for(let i=0;i<a.length;i++)a[i]=(raw[i]-nai.mean[i])/nai.scale[i];
    nai.layers.forEach((l,li)=>{const z=new Float64Array(l.O);for(let o=0;o<l.O;o++){let s=l.b[o],base=o*l.I;for(let i=0;i<l.I;i++)s+=l.w[base+i]*a[i];z[o]=s;}if(li<nai.layers.length-1){for(let o=0;o<z.length;o++)if(z[o]<0)z[o]=0;}a=z;});
    let p;if(nai.binary){const p1=a[0]>=0?1/(1+Math.exp(-a[0])):Math.exp(a[0])/(1+Math.exp(a[0]));p=[1-p1,p1];}else{const m=Math.max(...a),e=Array.from(a,v=>Math.exp(v-m)),s=e.reduce((x,y)=>x+y,0);p=e.map(v=>v/s);}
    probs.push(p);pred.push(argmaxRow(p));
  }
  return {probs,pred};
}

// ---------------------------------------------------------------------------
// NPY / NPZ writer for Python-compatible NAI4 datasets
// ---------------------------------------------------------------------------

function makeNpyHeader(descr, shape) {
  const shapeText = shape.length === 0 ? '()' : `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeText}, }`;
  // NPY v1: magic(6)+version(2)+header_len(2)+header must align to 16 bytes.
  const preamble = 10;
  let pad = 16 - ((preamble + header.length + 1) % 16);
  if (pad === 16) pad = 0;
  header += ' '.repeat(pad) + '\n';
  const hb = new TextEncoder().encode(header);
  if (hb.length > 65535) throw new Error('NPY header too large');
  const out = new Uint8Array(10 + hb.length);
  out.set([0x93,0x4e,0x55,0x4d,0x50,0x59,0x01,0x00], 0);
  new DataView(out.buffer).setUint16(8, hb.length, true);
  out.set(hb, 10);
  return out;
}

function concatBytes(a,b) {
  const out = new Uint8Array(a.length+b.length);
  out.set(a,0); out.set(b,a.length); return out;
}

function npyF32(values, shape) {
  const header = makeNpyHeader('<f4', shape);
  const payload = new Uint8Array(values.length*4);
  const dv = new DataView(payload.buffer);
  for (let i=0;i<values.length;i++) dv.setFloat32(i*4, Number(values[i]), true);
  return concatBytes(header,payload);
}

function npyI32(values, shape) {
  const arr = Array.from(values, Number);
  const header = makeNpyHeader('<i4', shape);
  const payload = new Uint8Array(arr.length*4);
  const dv = new DataView(payload.buffer);
  for (let i=0;i<arr.length;i++) dv.setInt32(i*4, arr[i], true);
  return concatBytes(header,payload);
}

function npyUnicode(values, shape) {
  const arr = Array.from(values, v=>String(v));
  const maxChars = Math.max(1, ...arr.map(s=>Array.from(s).length));
  const header = makeNpyHeader(`<U${maxChars}`, shape);
  const payload = new Uint8Array(arr.length*maxChars*4);
  const dv = new DataView(payload.buffer);
  let off=0;
  for (const s of arr) {
    const cps = Array.from(s, ch=>ch.codePointAt(0));
    for (let j=0;j<maxChars;j++) { dv.setUint32(off, cps[j] || 0, true); off += 4; }
  }
  return concatBytes(header,payload);
}

function flattenRawRows(rawRows) {
  let total=0;
  for (const r of rawRows) total += r.length;
  const out = new Float32Array(total*6);
  const offsets = new Int32Array(rawRows.length+1);
  let row=0;
  rawRows.forEach((r,i)=>{
    offsets[i]=row;
    out.set(r.data, row*6);
    row += r.length;
  });
  offsets[rawRows.length]=row;
  return {data:out, offsets, totalRows:row};
}

async function buildDatasetNpzBlob(ds) {
  if (!ds || !ds.Xrows?.length) throw new Error('Dataset is empty');
  const zip = new JSZip();
  const n=ds.Xrows.length, D=ds.N*6;
  const X=flattenRows(ds.Xrows);
  zip.file('X.npy', npyF32(X,[n,D]));
  zip.file('y.npy', npyI32(ds.y,[n]));
  zip.file('labels.npy', npyUnicode(ds.labels,[ds.labels.length]));
  zip.file('normalized_length.npy', npyI32([ds.N],[]));
  zip.file('window_length.npy', npyI32([ds.N],[]));
  zip.file('raw_lengths.npy', npyI32(ds.rawLengths || Array(n).fill(ds.N),[n]));
  zip.file('durations_ms.npy', npyI32(ds.durationsMs || Array(n).fill(0),[n]));
  zip.file('sample_rate_hz.npy', npyI32([ds.sampleRate || SAMPLE_RATE_HZ],[]));
  zip.file('channels.npy', npyUnicode(['ax','ay','az','gx','gy','gz'],[6]));
  zip.file('channel_count.npy', npyI32([6],[]));
  zip.file('preprocess.npy', npyUnicode(['nai4_raw6_plus_motion_repr_v1'],[]));
  if (ds.rawRows && ds.rawRows.length === n) {
    const raw=flattenRawRows(ds.rawRows);
    zip.file('raw_offsets.npy', npyI32(raw.offsets,[n+1]));
    zip.file('raw_data.npy', npyF32(raw.data,[raw.totalRows,6]));
  }
  return await zip.generateAsync({type:'blob', compression:'DEFLATE'});
}

function normalizeRawSixAxis(rawObj, targetN) {
  return centerColumns(resample(rawObj.data, rawObj.length, 6, targetN), targetN, 6);
}


window.NAI = Object.freeze({
  SAMPLE_RATE_HZ, MADGWICK_BETA, GRAVITY_MPS2, MAX_PACKAGE_BYTES,
  FILTER_HP, FILTER_LP, DEFAULT_HP_HZ, DEFAULT_LP_HZ,
  REP, CODE_TO_REP,
  loadNpz, parseDatasetArrays, buildDatasetNpzBlob,
  resample, centerColumns, normalizeRawSixAxis, firstOrderFilterColumns, filterConfigForRepresentation,
  sklearnStratifiedSplit, buildRepresentationDataset,
  fitScaler, standardizeRows, flattenRows,
  makeModel, predictTf, accuracy,
  exportNai4Int8, loadNai, loadNaiPackage, predictNai,
  predictQuantized, sweepConfidenceThreshold,
  _internals:Object.freeze({makeNpyHeader,npyF32,npyI32,npyUnicode,parseNpy,MT19937,makeCfg,parseCfg,quantizeMultiplier,multiplyByQuantizedMultiplier}),
});
