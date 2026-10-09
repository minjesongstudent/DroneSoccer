// Run the real inline UI handlers against mocked DOM/HTTP. No drone or server.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../experiments/faisal/lab-tools/Mac-Lab-Recorder/supervised_flight.py'), 'utf8');
const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const ids = ['hold', 'ready', 'land', 'cut', 'status', 'camera', 'error', 'connection', 'countdown'];
  const nodes = Object.fromEntries(ids.map(id => [id, {textContent:'', disabled:id==='hold', checked:false, setPointerCapture(){}}]));
  const h = {nodes, requests:[], state:{state:'ground', readiness:''}, failed:false, rejectStart:false, deferStart:false};
  const sandbox = {location:{hash:'#token'}, document:{querySelector:s=>nodes[s.slice(1)],addEventListener(){}},
    addEventListener(){}, setInterval(){return 1},clearInterval(){},setTimeout(){},
    fetch:async(url, options)=>{
      if (url === '/state') {
        if (h.failed) throw Error('connection lost');
        return {ok:true,json:async()=>({control:h.state,capture_remaining_s:170})};
      }
      const action = JSON.parse(options.body).action;h.requests.push(action);
      if (action === 'start' && h.deferStart) return new Promise(resolve=>{h.finishStart=()=>resolve({ok:true,json:async()=>({})})});
      if (action === 'start' && h.rejectStart) return {ok:false,json:async()=>({error:'stale camera',control_state:'ground'})};
      return {ok:true,json:async()=>({})};
    }};
  h.context=vm.createContext(sandbox);vm.runInContext(script,h.context);
  h.press=()=>nodes.hold.onpointerdown({target:nodes.hold,pointerId:1,preventDefault(){}});
  h.poll=()=>vm.runInContext('poll()',h.context);
  return h;
}
(async()=>{
  // A click without supervision must never stop the ground session or send commands.
  let h=harness();await flush();await h.press();await h.nodes.hold.onpointerup();
  assert.deepEqual(h.requests,[]);assert.match(h.nodes.error.textContent,/supervision/);
  // A server-confirmed preflight rejection must allow a deliberate retry, with no land.
  h=harness();await flush();h.nodes.ready.checked=true;h.rejectStart=true;
  await h.press();await h.nodes.hold.onpointerup();assert.deepEqual(h.requests,['heartbeat','start']);
  assert.match(h.nodes.error.textContent,/stale camera/);
  h.rejectStart=false;await h.press();await h.nodes.hold.onpointerup();
  assert.deepEqual(h.requests,['heartbeat','start','heartbeat','start','land']);
  // A transient poll error clears on recovery; never preserve a false disconnect warning.
  h=harness();await flush();h.failed=true;await h.poll();assert.equal(h.nodes.hold.disabled,true);
  assert.match(h.nodes.connection.textContent,/unreachable/);
  h.failed=false;await h.poll();assert.equal(h.nodes.connection.textContent,'Connected to local recorder');
  assert.equal(h.nodes.hold.disabled,false);assert.match(h.nodes.countdown.textContent,/170/);
  // Release while takeoff HTTP is pending must request landing immediately AND after reply.
  h=harness();await flush();h.nodes.ready.checked=true;h.deferStart=true;
  const pending=h.press();await flush();await h.nodes.hold.onpointerup();
  assert.deepEqual(h.requests,['heartbeat','start','land']);
  h.finishStart();await pending;assert.deepEqual(h.requests,['heartbeat','start','land','land']);
  // Loss of UI while held latches a land request and stops heartbeats.
  h=harness();await flush();h.nodes.ready.checked=true;await h.press();
  h.failed=true;await h.poll();assert.equal(h.requests.at(-1),'land');
  assert.equal(vm.runInContext('held',h.context),false);
  // Reconnection to stopped session never permits restart.
  h.failed=false;h.state={state:'stopped',readiness:''};await h.poll();
  assert.equal(h.nodes.hold.disabled,true);
  console.log('6 supervised flight UI regression scenarios passed (mock transport only)');
})().catch(error=>{console.error(error);process.exitCode=1});
