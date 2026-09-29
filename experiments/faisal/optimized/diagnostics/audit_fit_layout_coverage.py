"""Describe FIT-only collection coverage without inspecting reserved trajectories."""
from pathlib import Path
import json
import sys
import numpy as np

OUT = Path(__file__).parent
BANK = Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3')
sys.path.insert(0, 'D:/DroneSoccerData/runs/reset-coverage-20260928/source_snapshot')
from src.guided_curriculum import GuidedLessonEnv

records = [r for r in json.loads((BANK/'records.json').read_text()) if r['bank_split'] == 'fit']
rows = []
for r in records:
    assert r['seed'] % 4 != 3
    env = GuidedLessonEnv(stage=4)
    env.reset(seed=r['seed'])
    if r['usable_repeated_lap']:
        with np.load(BANK/'trajectories'/f"{r['seed']}.npz") as saved:
            np.testing.assert_allclose(env.pos, saved['positions'][0], atol=1e-12, rtol=0)
    delta = env.pos-env.hoop_centre
    bearing = np.arctan2(env.hoop_normal[1], env.hoop_normal[0])
    row = dict(seed=r['seed'], success=r['usable_repeated_lap'],
        height=float(env.hoop_centre[2]), distance=float(-delta@env.hoop_normal),
        bearing=float(bearing), lateral=float(delta@env._axis_h),
        vertical=float(delta[2]), yaw_error=float(env.heading-bearing),
        diameter_h=float(2*np.linalg.norm(env._hoop_h)),
        diameter_v=float(2*np.linalg.norm(env._hoop_v)))
    rows.append(row)
    env.close()
keys = [k for k in rows[0] if k not in ('seed','success')]
success = [r for r in rows if r['success']]
summary = {}
for key in keys:
    all_values = np.array([r[key] for r in rows])
    good = np.array([r[key] for r in success])
    edges = np.quantile(all_values, [0,.25,.5,.75,1])
    bins = np.minimum(np.searchsorted(edges[1:-1], all_values, side='right'), 3)
    summary[key] = dict(all_range=[float(all_values.min()),float(all_values.max())],
        successful_range=[float(good.min()),float(good.max())],
        successful_percentiles=[float(np.mean(all_values<=v)) for v in good],
        quartiles=[dict(episodes=int(sum(bins==i)), successes=sum(r['success'] for j,r in enumerate(rows) if bins[j]==i)) for i in range(4)])
result = dict(episodes=len(rows), successes=len(success), success_rows=success,
    dimensions=summary, reserved_trajectories_read=0, training_updates=0,
    limitations='Nine successes are descriptive only; this does not establish causal effects. All collection layouts are front-start lessons, not the general corner-start task.')
(OUT/'fit_layout_coverage.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
