// Read-only comparison panel; only the C viewer's own controls change its run.
const experimentPanel = document.createElement('section');
experimentPanel.className = 'panel';
experimentPanel.style.margin = '24px 32px 0';
const heading = document.createElement('h2');
heading.textContent = 'Guided experiment C · successful examples + easier starts';
const links = document.createElement('p');
for (const [label, port] of [['Open original A', 8771], ['Open corrected B', 8772]]) {
  const link = document.createElement('a');
  link.href = 'http://127.0.0.1:' + port;
  link.target = '_blank';
  link.rel = 'noopener';
  link.style.color = '#8bc6ff';
  link.textContent = label;
  links.append(link, '  ·  ');
}
const runStatus = document.createElement('p');
const bootstrapStatus = document.createElement('p');
const benchmarkStatus = document.createElement('p');
const targetStatus = document.createElement('p');
const comparisonRows = document.createElement('div');
const comparisonNote = document.createElement('p');
comparisonNote.className = 'note';
comparisonNote.textContent = 'Teacher demonstrations, policy training passes and policy evaluation successes are separate. C learns from demonstrations before training and an imitation loss that fades to zero by 50,000 online steps. The policy chooses every online and evaluation action. Equal online steps exclude demonstrations and pretraining; they are not equal total compute. Fixed evaluation layouts are repeated validation cases, not proof of performance on unseen flights.';
experimentPanel.append(heading, links, runStatus, bootstrapStatus, benchmarkStatus, targetStatus, comparisonRows, comparisonNote);
document.querySelector('main').before(experimentPanel);
document.title = 'Drone Soccer · Guided experiment C';
document.querySelector('h1').textContent = 'Drone Soccer · Guided experiment C';

async function compareRuns() {
  try {
    const [comparisonResponse, stateResponse] = await Promise.all([fetch('/comparison'), fetch('/state')]);
    if (!comparisonResponse.ok || !stateResponse.ok) throw Error('Status unavailable');
    const [data, current] = await Promise.all([comparisonResponse.json(), stateResponse.json()]);
    runStatus.textContent = [['A', data.original], ['B', data.corrected], ['C', current]].map(([label, run]) =>
      label + ': ' + run.mode + (run.steps == null ? '' : ' · ' + run.steps.toLocaleString() + ' online steps')).join(' | ');
    const bootstrap = current.bootstrap;
    bootstrapStatus.textContent = bootstrap && bootstrap.successful_episodes != null
      ? 'Teacher demonstrations: ' + bootstrap.successful_episodes + '/' + bootstrap.episodes + ' successful episodes · ' + (bootstrap.transitions || 0).toLocaleString() + ' replay transitions · ' + (bootstrap.pretrain_updates || 0).toLocaleString() + ' pretraining updates. Policy training passes: ' + (current.total_passes || 0).toLocaleString() + '.'
      : 'Preparing teacher demonstrations and GPU pretraining. No policy results yet.';
    const latest = data.latest_original_centered;
    benchmarkStatus.textContent = latest
      ? 'Policy on the original centered-pass test: ' + latest.successful_episodes + '/' + latest.episodes + ' successful passes at ' + latest.step.toLocaleString() + ' online steps' + (latest.step === 0 ? ' (after pretraining, before online training)' : '') + '.'
      : 'Original centered-pass benchmark: waiting for the first policy evaluation.';
    const target = data.target;
    targetStatus.textContent = 'Target: at least 20/25 original centered passes at two consecutive online checkpoints. ' +
      (target.met ? 'Reached at the latest checkpoints; full-task results still matter.' : target.consecutive_checkpoints + '/2 qualifying checkpoints so far.');
    const matched = data.matched;
    comparisonRows.replaceChildren();
    if (data.warning) {
      comparisonRows.textContent = data.warning;
    } else if (!matched) {
      comparisonRows.textContent = 'Waiting for an A/B/C checkpoint at the same online-step count. First possible comparison: 10,000 steps.';
    } else {
      const title = document.createElement('p');
      title.textContent = 'Same online checkpoint: ' + matched.step.toLocaleString() + ' steps · 25 identical test seeds per suite';
      comparisonRows.append(title);
      const labels = {original_centered: 'Original centered pass', full_task: 'Full task', full_task_delay_200ms: 'Full task + 200 ms'};
      for (const [name, values] of Object.entries(matched.suites)) {
        const row = document.createElement('p');
        row.textContent = labels[name] + ': ' + [['A', 'original'], ['B', 'corrected'], ['C', 'guided']].map(([label, key]) =>
          label + ' ' + Math.round(values[key].success_rate * 25) + '/25').join(' · ') + ' successful episodes';
        comparisonRows.append(row);
      }
      if (matched.omitted.length) {
        const note = document.createElement('p');
        note.className = 'note';
        note.textContent = 'Not compared: ' + matched.omitted.map(name => labels[name]).join(', ') + '. A/B must still use centered lesson 0 for its basic result to match C’s fixed benchmark; other missing or differing protocols are also excluded.';
        comparisonRows.append(note);
      }
    }
  } catch (error) {
    runStatus.textContent = 'Comparison temporarily unavailable';
  }
  setTimeout(compareRuns, 10000);
}
compareRuns();
