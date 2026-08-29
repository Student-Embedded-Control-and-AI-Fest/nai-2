
'use strict';

const NAI_MODEL_PAYLOAD_CAPACITY = 192512; // 188 KiB single-slot payload

const NAI_BLE = Object.freeze({
  DEVICE_PREFIX: 'NoodleAI',
  SERVICE_UUID: '7f8b0001-5f5b-4f4a-a5d5-2e889aa10001',
  IMU_UUID:     '7f8b0002-5f5b-4f4a-a5d5-2e889aa10001',
  CONTROL_UUID: '7f8b0003-5f5b-4f4a-a5d5-2e889aa10001',
  MODEL_UUID:   '7f8b0004-5f5b-4f4a-a5d5-2e889aa10001',
  STATUS_UUID:  '7f8b0005-5f5b-4f4a-a5d5-2e889aa10001',
  OP_SET_TRAINING: 0x01,
  OP_SET_INFERENCE: 0x02,
  OP_DEPLOY_BEGIN: 0x20,
  OP_FILE_BEGIN: 0x21,
  OP_FILE_END: 0x22,
  OP_DEPLOY_COMMIT: 0x23,
  OP_DEPLOY_ABORT: 0x24,
  SAMPLE_PERIOD_MS: 20,
});

class NoodleAIBLE extends EventTarget {
  constructor() {
    super();
    this.device=null; this.server=null; this.service=null;
    this.imuChar=null; this.controlChar=null; this.modelChar=null; this.statusChar=null;
    this.connected=false; this.mode='?'; this.packetCount=0; this.statusWaiters=[];
    this._onImu=this._onImu.bind(this);
    this._onStatus=this._onStatus.bind(this);
    this._onDisconnected=this._onDisconnected.bind(this);
  }

  emit(type, detail={}) { this.dispatchEvent(new CustomEvent(type,{detail})); }

  static supportMessage() {
    if (!window.isSecureContext) return 'Web Bluetooth needs HTTPS (or localhost while developing).';
    if (!navigator.bluetooth) return 'Web Bluetooth is not enabled. Chrome/Edge is recommended. On Chrome/Linux, enable Experimental Web Platform Features if navigator.bluetooth is unavailable.';
    return 'Web Bluetooth is available.';
  }

  async connect() {
    if (!navigator.bluetooth) throw new Error(NoodleAIBLE.supportMessage());
    this.device=await navigator.bluetooth.requestDevice({
      filters:[{namePrefix:NAI_BLE.DEVICE_PREFIX}], optionalServices:[NAI_BLE.SERVICE_UUID],
    });
    this.device.addEventListener('gattserverdisconnected',this._onDisconnected);
    this.server=await this.device.gatt.connect();
    this.service=await this.server.getPrimaryService(NAI_BLE.SERVICE_UUID);
    [this.imuChar,this.controlChar,this.modelChar,this.statusChar]=await Promise.all([
      this.service.getCharacteristic(NAI_BLE.IMU_UUID),
      this.service.getCharacteristic(NAI_BLE.CONTROL_UUID),
      this.service.getCharacteristic(NAI_BLE.MODEL_UUID),
      this.service.getCharacteristic(NAI_BLE.STATUS_UUID),
    ]);
    await this.statusChar.startNotifications();
    this.statusChar.addEventListener('characteristicvaluechanged',this._onStatus);
    await this.imuChar.startNotifications();
    this.imuChar.addEventListener('characteristicvaluechanged',this._onImu);
    this.connected=true; this.packetCount=0;
    this.emit('connected',{name:this.device.name||'NoodleAI'});
    try {
      const v=await this.statusChar.readValue();
      const text=this.decodeAscii(v); if(text) this._handleStatus(text,false);
    } catch(_) {}
  }

  async disconnect() {
    if(this.device?.gatt?.connected) this.device.gatt.disconnect();
    else this._onDisconnected();
  }

  _onDisconnected() {
    this.connected=false; this.mode='?';
    this.rejectAllWaiters(new Error('BLE disconnected'));
    this.emit('disconnected');
  }

  decodeAscii(v) {
    return new TextDecoder('utf-8').decode(new Uint8Array(v.buffer,v.byteOffset,v.byteLength)).replace(/\\0+$/g,'').trim();
  }

  _onStatus(e) {
    const text=this.decodeAscii(e.target.value); if(text) this._handleStatus(text,true);
  }

  _handleStatus(text, notify=true) {
    if(text==='MODE:T') this.mode='T';
    else if(text==='MODE:I') this.mode='I';
    this.emit('status',{text,notify,mode:this.mode});
    for(let i=0;i<this.statusWaiters.length;i++) {
      const w=this.statusWaiters[i];
      if(text.startsWith('ERR:')) { clearTimeout(w.timer); this.statusWaiters.splice(i,1); w.reject(new Error(text)); break; }
      const match=w.exact ? text===w.prefix : text.startsWith(w.prefix);
      if(match) { clearTimeout(w.timer); this.statusWaiters.splice(i,1); w.resolve(text); break; }
    }
  }

  waitStatus(prefix,timeoutMs=8000,exact=false) {
    return new Promise((resolve,reject)=>{
      const w={prefix,exact,resolve,reject,timer:null};
      w.timer=setTimeout(()=>{const i=this.statusWaiters.indexOf(w);if(i>=0)this.statusWaiters.splice(i,1);reject(new Error(`Timed out waiting for ${prefix}`));},timeoutMs);
      this.statusWaiters.push(w);
    });
  }

  rejectAllWaiters(err) { for(const w of this.statusWaiters){clearTimeout(w.timer);w.reject(err);} this.statusWaiters=[]; }

  _onImu(e) {
    const v=e.target.value;
    if(v.byteLength!==128){this.emit('warning',{text:`IMU packet is ${v.byteLength} bytes; expected 128.`});return;}
    const t0=v.getUint32(0,true), count=v.getUint8(4);
    if(count<1||count>5){this.emit('warning',{text:`Invalid IMU sample count ${count}.`});return;}
    this.packetCount++;
    for(let s=0;s<count;s++) {
      const off=8+s*24;
      this.emit('imu',{sample:{
        t_ms:t0+s*NAI_BLE.SAMPLE_PERIOD_MS,
        ax:v.getFloat32(off+0,true), ay:v.getFloat32(off+4,true), az:v.getFloat32(off+8,true),
        gx:v.getFloat32(off+12,true), gy:v.getFloat32(off+16,true), gz:v.getFloat32(off+20,true),
      },packetCount:this.packetCount});
    }
  }

  async writeControl(bytes) {
    if(!this.connected||!this.controlChar) throw new Error('NoodleAI is not connected');
    if(this.controlChar.writeValueWithResponse) await this.controlChar.writeValueWithResponse(bytes);
    else await this.controlChar.writeValue(bytes);
  }
  async writeModel(bytes) {
    if(this.modelChar.writeValueWithResponse) await this.modelChar.writeValueWithResponse(bytes);
    else await this.modelChar.writeValue(bytes);
  }
  async setTraining(){await this.writeControl(Uint8Array.of(NAI_BLE.OP_SET_TRAINING));}
  async setInference(){await this.writeControl(Uint8Array.of(NAI_BLE.OP_SET_INFERENCE));}

  crc32(bytes) {
    let crc=0xffffffff;
    for(let i=0;i<bytes.length;i++){crc^=bytes[i];for(let k=0;k<8;k++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
    return (crc^0xffffffff)>>>0;
  }
  fileBegin(name,bytes) {
    const nb=new TextEncoder().encode(name); if(!nb.length||nb.length>15) throw new Error(`Invalid NAI filename ${name}`);
    const out=new Uint8Array(10+nb.length),dv=new DataView(out.buffer);
    out[0]=NAI_BLE.OP_FILE_BEGIN;out[1]=nb.length;dv.setUint32(2,bytes.length,true);dv.setUint32(6,this.crc32(bytes),true);out.set(nb,10);return out;
  }

  async deployPackage(files,{chunkSize=160,onProgress=()=>{}}={}) {
    if(!this.connected) throw new Error('Connect NoodleAI first');
    const entries=Object.entries(files||{}); if(!entries.length) throw new Error('No NAI4 package to deploy');
    const total=entries.reduce((s,[,b])=>s+b.byteLength,0);
    if(total>NAI_MODEL_PAYLOAD_CAPACITY) {
      throw new Error(`NAI4 raw files ${(total/1024).toFixed(1)} KiB exceed the current M0Sense single-slot payload capacity ${(NAI_MODEL_PAYLOAD_CAPACITY/1024).toFixed(0)} KiB`);
    }
    let sent=0;
    try {
      onProgress({sent,total,file:'',stage:'begin'});
      await this.writeControl(Uint8Array.of(NAI_BLE.OP_DEPLOY_BEGIN));
      const slot=await this.waitStatus('DEPLOY:SINGLE',15000,true);
      this.emit('deploy-log',{text:`${slot} · ${(total/1024).toFixed(1)} KiB`});
      for(const [name,b0] of entries) {
        const bytes=b0 instanceof Uint8Array?b0:new Uint8Array(b0);
        onProgress({sent,total,file:name,stage:'file-begin'});
        await this.writeControl(this.fileBegin(name,bytes));
        await this.waitStatus(`FILE_READY:${name}`,5000,true);
        for(let pos=0;pos<bytes.length;pos+=chunkSize) {
          const chunk=bytes.subarray(pos,Math.min(pos+chunkSize,bytes.length));
          await this.writeModel(chunk); sent+=chunk.length;
          onProgress({sent,total,file:name,stage:'sending'});
          if((Math.floor(pos/chunkSize)%25)===0) await new Promise(r=>setTimeout(r,0));
        }
        await this.writeControl(Uint8Array.of(NAI_BLE.OP_FILE_END));
        await this.waitStatus(`FILE_OK:${name}`,8000,true);
        this.emit('deploy-log',{text:`FILE_OK:${name}`});
      }
      onProgress({sent,total,file:'',stage:'commit'});
      await this.writeControl(Uint8Array.of(NAI_BLE.OP_DEPLOY_COMMIT));
      await this.waitStatus('MODEL_STORED',15000,true);
      onProgress({sent:total,total,file:'',stage:'done'});
    } catch(err) {
      try{await this.writeControl(Uint8Array.of(NAI_BLE.OP_DEPLOY_ABORT));}catch(_){}
      onProgress({sent,total,file:'',stage:'error',error:err.message});
      throw err;
    }
  }
}

window.noodleBLE=new NoodleAIBLE();
window.NAI_BLE=NAI_BLE;
