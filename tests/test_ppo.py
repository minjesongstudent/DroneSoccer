import numpy as np
import pytest
import torch

from src import actor as actor_module
from src import contract, evaluate, ppo, train
from src.params import Drone
from src.sim import PRIVILEGED_DIM, HoopEnv


def test_privileged_state_has_one_length_in_every_draw():
    """The critic's input is fixed in size whatever the episode's command delay."""
    env = HoopEnv(randomise=True)
    for seed in range(5):
        env.reset(seed=seed)
        assert env.privileged().shape == (PRIVILEGED_DIM,)
        env.step(np.zeros(contract.ACTION_DIM, np.float32))
        assert np.isfinite(env.privileged()).all()


def test_privileged_state_with_no_command_delay():
    """evaluate --delay 0 flies with nothing queued."""
    env = HoopEnv(drone=Drone(delay_steps=0))
    env.reset(seed=0)
    assert env.privileged().shape == (PRIVILEGED_DIM,)


def test_privileged_state_says_where_the_drone_is_round_the_hoop():
    """Its first number is how far past the hoop's plane the drone is: what tells a
    drone just through the hoop from one in front, when both see nothing."""
    env = HoopEnv()
    env.reset(seed=0)
    env.place([1.0, 1.0, 1.5], 0.0, [1.5, 1.0, 1.5], [1.0, 0.0, 0.0])
    assert env.privileged()[0] == pytest.approx(-0.5)
    env.place([2.0, 1.0, 1.5], 0.0, [1.5, 1.0, 1.5], [1.0, 0.0, 0.0])
    assert env.privileged()[0] == pytest.approx(0.5)


def _random_actor():
    torch.manual_seed(0)
    actor = ppo.Actor()
    torch.nn.init.normal_(actor.head[-1].weight, 0.0, 0.1)   # the real init is near zero
    return actor


def test_actor_steps_as_it_replays_sequences():
    """PPO recomputes the rollout's log-probabilities from sequences, so a sequence must
    give what step-by-step flying gave, with the memory cleared where episodes start."""
    actor = _random_actor()
    T, B = 20, 3
    x = torch.randn(T, B, contract.POLICY_INPUT_DIM)
    starts = torch.zeros(T, B)
    starts[0, :] = 1.0
    starts[7, 1] = 1.0
    starts[13, 2] = 1.0
    h0 = torch.randn(B, actor.memory)
    with torch.no_grad():
        seq, h_seq = actor(x, h0, starts)
        h, steps = h0, []
        for t in range(T):
            mean, h = actor.step(x[t], h * (1.0 - starts[t])[:, None])
            steps.append(mean)
    assert torch.allclose(seq, torch.stack(steps), atol=1e-5)
    assert torch.allclose(h_seq, h, atol=1e-5)
    assert seq.abs().max() > 1e-3   # not trivially zero


def test_memory_is_cleared_where_an_episode_starts():
    actor = _random_actor()
    x = torch.randn(5, 1, contract.POLICY_INPUT_DIM)
    starts = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0]])
    with torch.no_grad():
        mean, _ = actor(x, torch.randn(1, actor.memory), starts)
        fresh, _ = actor(x[3:], actor.initial(1), starts[3:])
    assert torch.allclose(mean[3:], fresh, atol=1e-5)


def test_advantages_stop_at_a_crash_and_bootstrap_past_the_rollout():
    roll = {"values": np.array([[1.0], [2.0], [3.0]], np.float32),
            "rewards": np.array([[0.0], [-3.0], [0.5]], np.float32),
            "dones": np.array([[0.0], [1.0], [0.0]], np.float32),
            "last_value": np.array([4.0], np.float32)}
    adv, ret = ppo.advantages(roll, gamma=0.9, lam=1.0)
    # Step 1 crashed: nothing after it counts.
    assert adv[1, 0] == pytest.approx(-3.0 - 2.0)
    # Step 2 runs on past the rollout's end, valued by the critic.
    assert adv[2, 0] == pytest.approx(0.5 + 0.9 * 4.0 - 3.0)
    assert ret[0, 0] == pytest.approx(0.0 + 0.9 * (-3.0))


def test_a_timeout_bootstraps_and_a_crash_does_not():
    """Fleet.step adds GAMMA times the critic's value of the state a timeout cut off."""
    rng = np.random.default_rng(0)
    fleet = ppo.Fleet(1, 1.0, False, 0.0, 0, rng)
    env = fleet.envs[0]
    env.place([1.0, 1.0, 1.5], 0.0, [1.5, 1.0, 1.5], [1.0, 0.0, 0.0])   # resets env.steps
    env.steps = contract.LAPS_MAX_STEPS - 1
    rewards, dones, ended = fleet.step(np.zeros((1, contract.ACTION_DIM), np.float32), lambda o, p: 10.0)
    assert dones[0] == 1.0 and ended[0][0]["outcome"] == "timeout"
    assert rewards[0] == pytest.approx(ppo.REWARD_SCALE * -contract.R_TIME + contract.GAMMA * 10.0, abs=0.05)


def test_saved_actor_flies_like_the_trained_one_and_evaluate_loads_it(tmp_path):
    actor = _random_actor()
    path = tmp_path / "model.zip"
    actor_module.save_actor(actor, path)
    assert actor_module.is_ppo(path)
    assert actor_module.is_ppo(tmp_path / "model")   # SB3's load adds the .zip itself
    policy = evaluate.load_policy(str(tmp_path / "model"))
    x = np.random.default_rng(0).normal(size=contract.POLICY_INPUT_DIM).astype(np.float32)
    first = policy(x)
    with torch.no_grad():
        expected = actor.step(torch.as_tensor(x)[None], actor.initial(1))[0][0].numpy().clip(-1, 1)
    assert np.allclose(first, expected, atol=1e-6)
    assert not np.allclose(policy(x), first)   # its memory carried on
    policy.reset()
    assert np.allclose(policy(x), first)


def test_sac_models_are_not_taken_for_ppo(tmp_path):
    from stable_baselines3 import SAC
    model = SAC("MlpPolicy", HoopEnv(), device="cpu", buffer_size=10)
    model.save(tmp_path / "sac.zip")
    assert not actor_module.is_ppo(tmp_path / "sac.zip")
    assert not actor_module.is_ppo(tmp_path / "sac")
    (tmp_path / "junk.zip").write_text("not a zip")
    assert not actor_module.is_ppo(tmp_path / "junk.zip")   # left for SB3 to report


def test_tracker_counts_the_students_flights_only():
    """Teacher-flown DAgger episodes and run-in starts say nothing of the student's start."""
    tracker = ppo.Tracker(100, 1.0, writer=None, start_steps=0)
    flight = {"outcome": "hit", "passes": 1, "steps": 80, "run_in": False}
    tracker.add([(flight, True, 0.0), (dict(flight, run_in=True), False, 0.0), (flight, False, 0.0)],
                fleet=None, steps=0)
    assert len(tracker.episodes.recent) == 1 and len(tracker.episodes.run_ins) == 1


def _labels(T=40, B=3, start_at=None):
    rng = np.random.default_rng(0)
    roll = {"obs": rng.normal(size=(T, B, contract.POLICY_INPUT_DIM)).astype(np.float32),
            "labels": rng.uniform(-1, 1, (T, B, contract.ACTION_DIM)).astype(np.float32),
            "starts": np.zeros((T, B), np.float32)}
    if start_at is not None:
        roll["starts"][start_at, 0] = 1.0
    labels = ppo.Labels()
    labels.add(roll)
    return labels, roll


def test_labels_score_only_steps_with_a_true_memory(monkeypatch):
    """Mid-episode sequences replay from a clear memory, so their first BC_BURN_IN steps
    are not scored, except from where an episode starts in them."""
    monkeypatch.setattr(ppo, "BC_BURN_IN", 10)
    actor = _random_actor()
    labels, roll = _labels(start_at=4)
    with torch.no_grad():
        mean, _ = actor(torch.as_tensor(roll["obs"]), actor.initial(3), torch.as_tensor(roll["starts"]))
    error = ((mean.numpy() - roll["labels"]) ** 2).mean(-1)
    scored = np.zeros_like(error, bool)
    scored[10:] = True
    scored[4:, 0] = True
    expected = error[scored].mean()
    with torch.no_grad():
        assert labels.loss(actor, np.arange(3)).item() == pytest.approx(expected, rel=1e-4)


def test_labels_round_trip(tmp_path):
    labels, _ = _labels()
    labels.save(tmp_path / "l.npz")
    back = ppo.Labels.load(tmp_path / "l.npz")
    assert len(back) == 3
    assert np.array_equal(back.joined["labels"], labels.joined["labels"])


def test_normaliser_fits_each_part_of_the_input():
    critic = ppo.Critic(obs_dim=2, priv_dim=1)
    obs = np.array([[0.0, 10.0], [2.0, 10.0]])
    priv = np.array([[1.0], [5.0]])
    ppo.fit_normaliser(critic, obs, priv)
    assert critic.mean.tolist() == pytest.approx([1.0, 10.0, 3.0])
    assert critic.std.tolist() == pytest.approx([1.0, ppo.STD_FLOOR, 2.0])


def test_too_few_simulators_for_the_minibatches_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="MINIBATCHES"):
        ppo.train_ppo(100, "x", 0, 1.0, False, rollout_envs=ppo.MINIBATCHES - 1)


def test_train_resume_and_pick_best(monkeypatch, tmp_path, capsys):
    """DAgger, the critic warm-up and PPO run end to end; a cut-off run resumes from
    its snapshot, with the teacher's labels; the clone is a candidate for model.zip."""
    monkeypatch.setattr(train, "RUNS", tmp_path)
    monkeypatch.setattr(ppo, "DAGGER_BETA", (1.0, 0.0))
    monkeypatch.setattr(ppo, "DAGGER_STEPS", 256)
    monkeypatch.setattr(ppo, "ROLLOUT_STEPS", 32)
    monkeypatch.setattr(ppo, "CRITIC_WARMUP", 256)
    monkeypatch.setattr(ppo, "CHECKPOINT_EVERY", 512)
    ppo.train_ppo(1200, "p", seed=0, level=1.0, randomise=True, select_episodes=0, rollout_envs=4)
    out = tmp_path / "p"
    assert (out / "resume" / "labels.npz").exists()
    assert (out / "checkpoints" / "model_512_steps.zip").exists()   # the clone, after 2 x 256
    assert '"steps": 1024' in (out / "resume" / "state.json").read_text()

    path = ppo.train_ppo(2000, "p", seed=0, level=1.0, randomise=True, resume=True, select_episodes=1,
                         rollout_envs=4)
    text = capsys.readouterr().out
    assert "resumed  from step 1024" in text and "dagger   round" in text
    assert text.count("dagger   round") == 2   # not again on resume
    assert actor_module.is_ppo(path)
    assert "model_512_steps.zip" in (out / "checkpoints.json").read_text()


def _tiny(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "RUNS", tmp_path)
    monkeypatch.setattr(ppo, "DAGGER_BETA", (1.0, 0.0))
    monkeypatch.setattr(ppo, "DAGGER_STEPS", 256)
    monkeypatch.setattr(ppo, "ROLLOUT_STEPS", 32)
    monkeypatch.setattr(ppo, "CRITIC_WARMUP", 256)
    monkeypatch.setattr(ppo, "CHECKPOINT_EVERY", 10_000)


def test_a_run_cut_off_after_dagger_resumes_without_redoing_it(monkeypatch, tmp_path, capsys):
    """The snapshot written as DAgger ends is enough to carry on, long before the first
    CHECKPOINT_EVERY."""
    _tiny(monkeypatch, tmp_path)
    ppo.train_ppo(512, "c", seed=0, level=1.0, randomise=False, select_episodes=0, rollout_envs=4)
    assert '"steps": 512' in (tmp_path / "c" / "resume" / "state.json").read_text()
    ppo.train_ppo(768, "c", seed=0, level=1.0, randomise=False, resume=True, select_episodes=0, rollout_envs=4)
    text = capsys.readouterr().out
    assert "resumed  from step 512" in text and text.count("dagger   round") == 2


def test_without_dagger_the_scale_flights_count(monkeypatch, tmp_path, capsys):
    """--dagger 0 flies DAgger's first round's worth to set the input scales, and those
    steps count towards --steps like every other."""
    _tiny(monkeypatch, tmp_path)
    ppo.train_ppo(256, "d", seed=0, level=1.0, randomise=False, dagger=0, select_episodes=0, rollout_envs=4)
    assert '"steps": 256' in (tmp_path / "d" / "resume" / "state.json").read_text()
    assert "dagger   round" not in capsys.readouterr().out


def test_dagger_round_line_reports_ended_flights():
    """The tiny runs above end no flights, so this line was never built with any."""
    flights = [({"passes": 3, "outcome": "timeout"}, True, 0.0), ({"passes": 1, "outcome": "hit"}, True, 0.0)]
    assert ppo._flights(flights, True) == "2 flights 2.00 passes/flight, 2+ passes 50%, crashed 50%"
    assert ppo._flights(flights, False) == "no flights"


def test_a_run_cut_off_mid_dagger_carries_on_from_the_next_round(monkeypatch, tmp_path, capsys):
    """A snapshot after every round: a job killed in round 1 resumes there, with round 0's labels."""
    _tiny(monkeypatch, tmp_path)
    real_clone, calls = ppo.clone, []

    def dies_in_round_1(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("walltime")
        return real_clone(*args, **kwargs)

    monkeypatch.setattr(ppo, "clone", dies_in_round_1)
    with pytest.raises(KeyboardInterrupt):
        ppo.train_ppo(768, "m", seed=0, level=1.0, randomise=False, select_episodes=0, rollout_envs=4)
    assert '"steps": 256' in (tmp_path / "m" / "resume" / "state.json").read_text()
    monkeypatch.setattr(ppo, "clone", real_clone)
    capsys.readouterr()
    ppo.train_ppo(768, "m", seed=0, level=1.0, randomise=False, resume=True, select_episodes=0, rollout_envs=4)
    text = capsys.readouterr().out
    assert "resumed  from step 256" in text
    assert "dagger   round 0" not in text and "dagger   round 1" in text
    assert "on 16 sequences" in text   # round 0's 8 and round 1's 8


def test_resume_follows_the_run_not_the_resume_options(monkeypatch, tmp_path):
    """Warm-up and labels are the run's: a --dagger 0 run resumed without --dagger keeps no
    warm-up, and ignores labels an older run of the same name left behind."""
    _tiny(monkeypatch, tmp_path)
    ppo.train_ppo(512, "r", seed=0, level=1.0, randomise=False, select_episodes=0, rollout_envs=4)
    assert (tmp_path / "r" / "resume" / "labels.npz").exists()   # the older run's
    ppo.train_ppo(256, "r", seed=0, level=1.0, randomise=False, dagger=0, select_episodes=0, rollout_envs=4)
    ppo.train_ppo(512, "r", seed=0, level=1.0, randomise=False, resume=True, select_episodes=0, rollout_envs=4)
    run = torch.load(tmp_path / "r" / "resume" / "state.pt", weights_only=False)["run"]
    assert run["warmup"] == 0 and not run["labels"] and run["phase"] == "ppo"


def test_restart_puts_every_simulator_on_a_new_episode_at_the_current_beta():
    fleet = ppo.Fleet(4, 1.0, False, 0.0, 0, np.random.default_rng(0))
    fleet.starts[:] = 0.0
    fleet.beta = 1.0
    fleet.restart()
    assert fleet.starts.tolist() == [1.0] * 4 and fleet.expert.all()
    assert all(env.steps == 0 for env in fleet.envs)


def test_log_dirs_are_numbered_by_the_exact_run_name(monkeypatch, tmp_path):
    """foo_s0_1 belongs to foo_s0, not to foo: the numbering plots.find_log_dir reads back."""
    monkeypatch.setattr(train, "RUNS", tmp_path)
    (tmp_path / "foo_s0_1").mkdir()
    assert ppo._log_dir("foo", resume=False) == tmp_path / "foo_1"
    (tmp_path / "foo_1").mkdir()
    assert ppo._log_dir("foo", resume=True) == tmp_path / "foo_1"
    assert ppo._log_dir("foo", resume=False) == tmp_path / "foo_2"


def test_teacher_then_student_keeps_the_true_state_out_of_the_student(monkeypatch, tmp_path):
    """A privileged teacher trains, then labels a student's DAgger: the student's saved
    actor reads the policy input alone and is not flagged privileged."""
    _tiny(monkeypatch, tmp_path)
    ppo.train_ppo(512, "t", seed=0, level=1.0, randomise=True, select_episodes=0, rollout_envs=4,
                  privileged_actor=True)
    teacher = tmp_path / "t" / "final.zip"
    assert actor_module.is_privileged(teacher)
    assert actor_module.load_actor(teacher).obs_dim == contract.POLICY_INPUT_DIM + PRIVILEGED_DIM
    ppo.train_ppo(512, "s", seed=0, level=1.0, randomise=True, select_episodes=0, rollout_envs=4,
                  teacher=str(teacher))
    student = tmp_path / "s" / "final.zip"
    assert not actor_module.is_privileged(student)
    assert actor_module.load_actor(student).obs_dim == contract.POLICY_INPUT_DIM
