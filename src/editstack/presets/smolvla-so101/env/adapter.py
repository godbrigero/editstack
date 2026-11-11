"""SmolVLA (LeRobot) policy on the SO-101 MuJoCo yellow-cube scene, MEASURED cameras.

Port of the original project's evalpick.py (closed loop) + bench.py (staged REACH/PICK classification).
Sim code: stack.json "sim_repo" or env SO101_REPO (the airtruck repo with the measured-camera fix of 2026-11-09). Episode spec: {"i", "x", "y", "yaw", "home": [5 floats]}; policy noise seeded by (seed, i) exactly as
evalpick.py did, so results are comparable with the old runs and paired across variants.
metrics: reach (tip within 1.5 cm horiz / 1 cm vert of cube at close, or picked), pick (cube lifted >= 2 cm).
outcome: P picked | R reached, grip failed | s short | l long | w wide | h too high | N no attempt.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = None   # the sim code: env SO101_REPO, or stack.json "sim_repo" (set per machine; never hard-coded here)
os.environ.setdefault("SO101_SIM_CAMERAS", "measured")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
SECONDS = 18.0
REACH_XY_CM, REACH_Z_CM, PICK_LIFT_M = 1.5, 1.0, 0.02


def _imports(ctx=None):
    global REPO
    if REPO is None:
        repo = os.environ.get("SO101_REPO") or (ctx.stack.cfg.get("sim_repo") if ctx is not None else None)
        if not repo:
            raise RuntimeError("set the sim code location: stack.json \"sim_repo\" or env SO101_REPO")
        REPO = Path(os.path.expanduser(repo))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mujoco  # noqa
    import torch  # noqa
    from lerobot.common.control_utils import predict_action  # noqa
    from lerobot.policies.factory import make_pre_post_processors  # noqa
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa
    import airtruck.examples.so101_pick as P  # noqa
    import airtruck.examples.so101_pick_eval as E  # noqa
    return SimpleNamespace(mujoco=mujoco, torch=torch, predict_action=predict_action,
                           make_pre_post_processors=make_pre_post_processors, SmolVLAPolicy=SmolVLAPolicy, P=P, E=E)


def load_policy(tensors, ctx):
    m = _imports(ctx)
    torch = m.torch
    if m.P.CAMERA_SET != "measured":
        raise RuntimeError(f"sim camera set is {m.P.CAMERA_SET!r}, harness expects 'measured'")
    device = torch.device(ctx.device)
    policy = m.SmolVLAPolicy.from_pretrained(str(ctx.original_dir))
    sd = policy.state_dict()
    missing = [k for k in tensors if k not in sd]
    if missing:
        raise RuntimeError(f"{len(missing)} weight names not in the policy, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in tensors.items():
            t = torch.from_numpy(np.ascontiguousarray(v)).to(device=sd[k].device, dtype=sd[k].dtype)
            if not torch.equal(sd[k], t):
                sd[k].copy_(t)
    policy.to(device).eval()
    pre, post = m.make_pre_post_processors(policy.config, pretrained_path=str(ctx.original_dir),
                                           preprocessor_overrides={"device_processor": {"device": device.type}})
    driver, arm = m.P.make_driver()
    renderers = {s.name: m.mujoco.Renderer(driver.model, s.height, s.width) for s in m.P.CAMERAS}
    cams = {s.name: m.E.camera_name(arm, s.name, s.body) for s in m.P.CAMERAS}
    return SimpleNamespace(m=m, policy=policy, pre=pre, post=post, driver=driver, arm=arm, renderers=renderers,
                           cams=cams, device=device)


class _FixedRng:
    def __init__(self, home):
        self.home = np.asarray(home)

    def uniform(self, *_):
        return self.home


def classify(actions, geo, max_lift):
    """Staged outcome (port of bench.classify). geo (T,9) = cube xyz | tip xyz | jaw xyz in arm base frame (m)."""
    g = actions[:, 5]
    picked = bool(max_lift >= PICK_LIFT_M)
    out = {"max_lift_cm": round(float(max_lift) * 100, 1)}
    opened = np.where(g > 80)[0]
    closes = np.where(g[opened[0]:] < 50)[0] if len(opened) else []
    if not len(closes):
        return picked, picked, ("P" if picked else "N"), out
    c = opened[0] + closes[0]
    w = slice(c, min(c + 16, len(geo)))
    dxy = np.linalg.norm(geo[w, 3:5] - geo[w, 0:2], axis=1)
    k = c + int(np.argmin(dxy))
    cube = geo[k, 0:2]
    ru = cube / max(np.linalg.norm(cube), 1e-9)
    tu = np.array([-ru[1], ru[0]])
    e = geo[k, 3:5] - cube
    rad, tan, dz = float(e @ ru) * 100, float(e @ tu) * 100, float(geo[k, 5] - geo[k, 2]) * 100
    reached = bool(dxy.min() * 100 <= REACH_XY_CM and abs(dz) <= REACH_Z_CM)
    oc = "P" if picked else "R" if reached else "h" if dz > REACH_Z_CM else ("s" if rad < 0 else "l") if abs(rad) >= abs(tan) else "w"
    out.update(close_t=round(c / 30, 2), tip_cube_xy_cm=round(float(dxy.min()) * 100, 2), radial_cm=round(rad, 2),
               tangential_cm=round(tan, 2), dz_cm=round(dz, 2))
    return reached or picked, picked, oc, out


def run_episode(pol, episode, seed, ctx):
    m = pol.m
    P, E, torch = m.P, m.E, m.torch
    d, arm = pol.driver, pol.arm
    xy = np.array([episode["x"], episode["y"]])
    E.reset(d, arm, xy, float(episode["yaw"]), _FixedRng(episode["home"]))
    pol.policy.reset(); pol.pre.reset(); pol.post.reset()
    torch.manual_seed(int(seed) * 100003 + int(episode["i"]))
    cid, hid, jid = d.model.body("cube").id, d.model.body("arm/hand").id, d.model.body("arm/jaw").id
    acts, states, geo = [], [], []
    max_lift, best_hold, hold, clipped = 0.0, 0, 0, 0
    # observations at every chunk boundary (every 50 steps) feed offline probes; EDITSTACK_SAVE_OBS=0 disables
    save_obs = bool(ctx.out_dir) and os.environ.get("EDITSTACK_SAVE_OBS", "1") != "0"
    if save_obs:
        Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(int(SECONDS * E.FPS)):
        state = arm.get_observation()
        images = {}
        for name, r in pol.renderers.items():
            r.update_scene(d.data, camera=pol.cams[name])
            images[name] = r.render().copy()
        obs = {"observation.state": np.array([state[n] for n in E.NAMES], dtype=np.float32),
               **{f"observation.images.{n}": im for n, im in images.items()}}
        if save_obs and step % 50 == 0:
            np.savez_compressed(Path(ctx.out_dir) / f"obs_{step:03d}.npz",
                                **{k.replace("observation.", ""): v for k, v in obs.items()})
        action = m.predict_action(obs, pol.policy, pol.device, pol.pre, pol.post, use_amp=False,
                                  task=E.TASK, robot_type="so101_follower")
        raw = action.squeeze(0).to("cpu").numpy()
        vals = np.clip(raw, E.LOW, E.HIGH)
        clipped += int(np.any(vals != raw))
        arm.send_action({n: float(v) for n, v in zip(E.NAMES, vals)})
        d.step()
        max_lift = max(max_lift, E.lift(d))
        hold = hold + 1 if E.held(d) else 0
        best_hold = max(best_hold, hold)
        geo.append(np.r_[(d.data.xpos[cid] - P.BASE) @ P.R,
                         (d.data.xpos[hid] + d.data.xmat[hid].reshape(3, 3) @ P.TIP - P.BASE) @ P.R,
                         (d.data.xpos[jid] - P.BASE) @ P.R])
        acts.append(vals)
        states.append(obs["observation.state"])
    acts, geo = np.array(acts), np.array(geo)
    reach, pick, oc, extra = classify(acts, geo, max_lift)
    extra["max_lift_m"] = float(max_lift)
    if ctx.out_dir:
        Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(ctx.out_dir) / "traj.npz", actions=acts, states=np.array(states), geo=geo)
    jit = np.abs(np.diff(acts, 2, axis=0)).mean(0)
    return {"metrics": {"reach": reach, "pick": pick}, "outcome": oc, "held_1s": best_hold >= E.FPS,
            "jitter": [round(float(x), 3) for x in jit], "clipped": clipped, "secs": round(time.time() - t0, 1), **extra}


def rescore(result, ep_dir, ctx):
    """Re-classify an episode from its stored trajectory (after the REACH/PICK definition changes: bump
    metric_version in stack.json, edit classify(), then `editstack rescore`)."""
    z = np.load(Path(ep_dir) / "traj.npz")
    max_lift = result.get("max_lift_m", result.get("max_lift_cm", 0) / 100)
    reach, pick, oc, extra = classify(z["actions"], z["geo"], max_lift)
    result.update(extra)
    result["metrics"] = {"reach": reach, "pick": pick}
    result["outcome"] = oc
    return result


def load_obs(path):
    """A saved chunk-boundary observation (runs/<v>/<suite>/s<k>/ep<i>/obs_<step>.npz) as a policy observation dict."""
    z = np.load(path)
    return {("observation." + k): z[k] for k in z.files}


def capture(pol, obs, names, task=None):
    """Run ONE action-chunk prediction on a recorded observation and return {module_name: output activation} for
    the named modules (e.g. "model.vlm_with_expert.lm_expert.layers.15.mlp"), plus "actions" (the FIRST action of
    the predicted chunk; the hooks see the whole chunk computation, stacked over the 10 denoising steps).
    Inputs to a module: prefix the name with "in:". This is the offline probe primitive (seconds, no sim)."""
    m = pol.m
    torch = m.torch
    acts, hooks = {}, []

    def hook(key, want_input):
        def f(mod, inp, out):
            t = inp[0] if want_input else (out[0] if isinstance(out, tuple) else out)
            acts.setdefault(key, []).append(t.detach().float().cpu().numpy())
        return f

    for n in names:
        want_input = n.startswith("in:")
        mod = pol.policy.get_submodule(n[3:] if want_input else n)
        hooks.append(mod.register_forward_hook(hook(n, want_input)))
    try:
        pol.policy.reset(); pol.pre.reset(); pol.post.reset()
        a = m.predict_action(dict(obs), pol.policy, pol.device, pol.pre, pol.post, use_amp=False,
                             task=task or m.E.TASK, robot_type="so101_follower")
    finally:
        for h in hooks:
            h.remove()
    out = {k: (v[0] if len(v) == 1 else np.stack(v)) for k, v in acts.items()}   # stacked over the 10 denoising steps
    out["actions"] = a.squeeze(0).cpu().numpy()
    return out
