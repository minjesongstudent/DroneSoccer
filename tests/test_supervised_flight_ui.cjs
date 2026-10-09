// Exercise the real single-click UI handlers with mocked DOM/HTTP. No hardware.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../experiments/faisal/lab-tools/Mac-Lab-Recorder/supervised_flight.py'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const ids = ['start', 'ready', 'land', 'cut', 'status', 'camera', 'error', 'connection', 'countdown'];
  const nodes = Object.fromEntries(ids.map(id => [id, {textContent:'', disabled:id==='start', checked:false}]));
  const h = {nodes, requests:[], state:{state:'ground', readiness:''}, failed:false, rejectStart:false, deferStart:false, intervals:new Map(), listeners:{}, documentListeners:{}};
  let timerId=0;
  const sandbox = {location:{hash:'#token'}, document:{hidden:false,querySelector:s=>nodes[s.slice(1)],addEventListener:(event,fn)=>{h.documentListeners[event]=fn}},
    addEventListener:(event,fn)=>{h.listeners[event]=fn},
    setInterval:fn=>{const id=++timerId;h.intervals.set(id,fn);return id},clearInterval:id=>h.intervals.delete(id),setTimeout(){},
    fetch:async(url, options)=>{
      if (url === '/state') {
        if (h.failed) throw Error('connection lost');
        return {ok:true,json:async()=>({control:h.state,preflight_remaining_s:h.state.state==='ground'?170:null})};
      }
      const action = JSON.parse(options.body).action;h.requests.push(action);
      if (action === 'start' && h.deferStart) return new Promise(resolve=>{h.finishStart=()=>resolve({ok:true,json:async()=>({})})});
      if (action === 'start' && h.rejectStart) return {ok:false,json:async()=>({error:'stale camera',control_state:'ground'})};
      return {ok:true,json:async()=>({})};
    }};
  h.context=vm.createContext(sandbox);vm.runInContext(script,h.context);
  h.click=()=>nodes.start.onclick();
  h.poll=()=>vm.runInContext('poll()',h.context);
  h.tick=async()=>{for (const fn of [...h.intervals.values()]) await fn()};
  return h;
}
(async()=>{
  // 1. Checkbox remains required; an unchecked click sends no commands.
  let h=harness();await flush();await h.click();
  assert.deepEqual(h.requests,[]);assert.match(h.nodes.error.textContent,/supervision/);
  // 2. One normal click starts once and keeps liveness pulses without pointer holds.
  h=harness();await flush();h.nodes.ready.checked=true;await h.click();
  assert.deepEqual(h.requests,['heartbeat','start']);
  assert.equal(h.nodes.start.onpointerdown,undefined);assert.equal(h.nodes.start.onpointerup,undefined);
  await h.tick();await h.tick();assert.deepEqual(h.requests,['heartbeat','start','heartbeat','heartbeat']);
  await h.click();assert.equal(h.requests.filter(a=>a==='start').length,1);
  assert.equal(h.nodes.start.disabled,true);assert.equal(h.requests.includes('land'),false);
  // 3. LAND stops heartbeat and prevents a second start even before the next poll.
  await h.nodes.land.onclick();const count=h.requests.length;await h.tick();await h.click();
  assert.equal(h.requests.at(-1),'land');assert.equal(h.requests.length,count);
  // 4. Preflight rejection permits a deliberate retry without a landing command.
  h=harness();await flush();h.nodes.ready.checked=true;h.rejectStart=true;
  await h.click();assert.deepEqual(h.requests,['heartbeat','start']);
  assert.match(h.nodes.error.textContent,/stale camera/);await h.poll();
  h.rejectStart=false;await h.click();assert.deepEqual(h.requests,['heartbeat','start','heartbeat','start']);
  // 5. A transient idle polling failure clears upon reconnection without starting anything.
  h=harness();await flush();h.failed=true;await h.poll();assert.equal(h.nodes.start.disabled,true);
  assert.match(h.nodes.connection.textContent,/unreachable/);
  h.failed=false;await h.poll();assert.equal(h.nodes.connection.textContent,'Connected to local recorder');
  assert.equal(h.nodes.start.disabled,false);assert.match(h.nodes.countdown.textContent,/170/);
  assert.deepEqual(h.requests,[]);
  // 6. LAND during pending takeoff is immediate and repeated after the late response.
  h=harness();await flush();h.nodes.ready.checked=true;h.deferStart=true;
  const pending=h.click();await flush();await h.click();await h.nodes.land.onclick();
  assert.deepEqual(h.requests,['heartbeat','start','land']);
  h.finishStart();await pending;assert.deepEqual(h.requests,['heartbeat','start','land','land']);
  assert.equal(h.intervals.size,0);
  // 7. Disconnect while running requests landing and stops browser heartbeats.
  h=harness();await flush();h.nodes.ready.checked=true;await h.click();
  h.failed=true;await h.poll();assert.equal(h.requests.at(-1),'land');assert.equal(h.intervals.size,0);
  h.failed=false;h.state={state:'stopped',readiness:''};await h.poll();
  assert.equal(h.nodes.start.disabled,true);
  // 8. Server safety-triggered landing stops UI pulses, with no restart.
  h=harness();await flush();h.nodes.ready.checked=true;await h.click();
  h.state={state:'active',readiness:'',policy_seconds:360};await h.poll();
  assert.equal(h.intervals.size,1);assert.match(h.nodes.countdown.textContent,/Recording continues/);
  h.state={state:'landing',readiness:''};await h.poll();assert.equal(h.intervals.size,0);
  h.state={state:'stopped',readiness:''};await h.poll();assert.equal(h.nodes.start.disabled,true);
  // 9. Escape and losing focus each stop an active single-click session.
  for(const event of ['keydown','blur']){
    h=harness();await flush();h.nodes.ready.checked=true;await h.click();
    h.listeners[event]({key:'Escape',preventDefault(){}});await flush();
    assert.equal(h.requests.at(-1),'land');assert.equal(h.intervals.size,0);
  }
  // 10. Emergency is distinct from landing and permanently stops the UI session.
  h=harness();await flush();h.nodes.ready.checked=true;await h.click();
  h.nodes.cut.onclick();await flush();await h.click();
  assert.equal(h.requests.at(-1),'emergency');assert.equal(h.intervals.size,0);
  assert.equal(h.nodes.start.disabled,true);
  console.log('10 single-click flight UI regression scenarios passed (mock transport only)');
})().catch(error=>{console.error(error);process.exitCode=1});
