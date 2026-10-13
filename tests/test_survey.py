"""labcheck survey without a drone: the rim maths, the rate summary, and the hoop
survey's stages flown against a fake hoop that moves with the commands."""
import numpy as np

from src import contract, labcheck
from src.detect import Detection


def test_rim_fraction():
    e = ((480, 360), (300, 300), 0.0)
    assert labcheck.rim_fraction(e, 960, 720) == (1.0, "")
    rim, edge = labcheck.rim_fraction(e, 960, 720, dx=-480)   # centre on the left edge
    assert abs(rim - 0.5) < 0.02 and edge == "left"


def test_hold_rates():
    t = np.arange(0, 6, 0.1)
    on = (t >= 1) & (t < 4)
    cmds = {"t": t, "phase": np.where(on, "hold ud +50", "rest"),
            "ud": np.where(on, 50.0, 0.0), "yaw": np.zeros_like(t)}
    # Climbs at 0.33 m/s once moving, as the policy flights did at rc 40-48.
    tof = 80 + 33 * np.clip(t - 1.3, 0, 2.7)
    state = {"t": t, "tof": tof, "yaw": np.zeros_like(t)}
    text = "\n".join(labcheck.summarise_holds(state, cmds))
    assert "+0.33 m/s" in text and "+0.40 m/s" in text   # measured, and the sim's at rc 50 (RC_FULL ud 50)


class FakeSession:
    def __init__(self, folder):
        self.dir, self.t, self.moved, self.tof = folder, 0.0, np.zeros(2), 150.0
        self.frame_bgr, self.decorate, self.events, self.cmd = None, None, [], (0, 0, 0, 0)

    def now(self):
        return self.t

    def state(self):
        return {"tof": self.tof, "vgz": 0}

    def event(self, text):
        self.events.append(text)


class FakeHoop:
    """A hoop in the image that the commands move: yaw slides it sideways (the whole
    image with it), up/down slides it vertically by the climb over its distance (the
    cage behind, further away, by half that: image shift alone gets it wrong), forward
    grows it, strafing turns it (the ratio)."""

    def __init__(self, min_rim=0.6):
        self.cx, self.cy, self.size, self.angle, self.min_rim = 480.0, 151.0, 414.0, 0.0, min_rim

    def move(self, rc, session):
        lr, fb, ud, yaw = rc
        dx = -2.0 * yaw                        # clockwise turn: everything moves left
        climb = 0.001 * ud                     # metres this step
        distance = 0.45 / (self.size / 920)
        dy = 920 * climb / distance            # climbing: the hoop moves down the image
        self.cx += dx
        self.cy += dy
        session.tof += 100 * climb
        session.moved = session.moved + (dx, 0.5 * dy)
        self.size *= 1 + 0.0005 * fb
        self.angle = np.clip(self.angle + 0.002 * lr, -1.2, 1.2)

    def detect(self, frame):
        ratio = np.cos(self.angle)
        ellipse = ((self.cx, self.cy), (self.size * ratio, self.size), 0.0)
        rim, _ = labcheck.rim_fraction(ellipse, 960, 720)
        if rim < self.min_rim:
            return Detection(contract.NOT_VISIBLE.copy(), None, ellipse, None, "solid")
        obs = np.array([(self.cx - 480) / 480, (self.cy - 360) / 360, self.size / 920, ratio,
                        0, -(1 - ratio), 1, 1], np.float32)
        return Detection(obs, None, ellipse, None)


def test_hoop_survey_runs_every_stage(tmp_path):
    session, hoop = FakeSession(tmp_path), FakeHoop()
    survey = labcheck.HoopSurvey(session, behind=False)
    survey.detector = hoop.detect
    stages = set()
    for _ in range(6000):
        session.t += 0.1
        session.frame_bgr = np.zeros((720, 960, 3), np.uint8)
        stages.add(survey.stage)
        rc = survey()
        if rc is None:
            break
        session.cmd = rc
        hoop.move(rc, session)
    assert rc is None, f"stuck in {survey.stage}"
    assert {"slide left 3", "slide right 3", "slide top 3", "orbit left", "orbit right", "centre near"} <= stages
    text = "\n".join(labcheck.summarise_survey(tmp_path))
    # Every slide's prediction holds, the top one's too (tof, not image shift).
    for edge in ("left", "right", "top"):
        assert f"{edge:6s} 3/3 kept" in text, text
    # The fake detector needs 60% of the rim, and each edge should find about that.
    found = text.split("min_arc_fraction: ")[1]
    assert 0.5 <= float(found.split()[0]) <= 0.7, text
    for a in labcheck.ORBIT_ANGLES:
        line = next(x for x in text.splitlines() if f"1 m, {a} deg off" in x)
        assert "100%" in line, text
