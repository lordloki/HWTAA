# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""HWTAA paper bench: every number, table and figure of the paper comes from here.

    python3 hwtaa_bench.py quality      # all scenes x all methods        -> results/quality.json
    python3 hwtaa_bench.py ablations    # one component off at a time     -> results/ablations.json
    python3 hwtaa_bench.py window       # PSNR versus window length       -> results/window.json
    python3 hwtaa_bench.py ghost        # error by frames since uncovered -> results/ghost.json
    python3 hwtaa_bench.py frames       # frames kept for the figures     -> results/frames.npz

Methods run on the CPU reference scenes of harness/persp_scene.py (192 x 108 output, frames 30 to
59 measured, reference 6 x 6 supersampled). HWTAA runs the production GLSL (shaders/) through
moderngl; on a headless machine that is Mesa llvmpipe, so quality is exact and time is not
measured here (see gpu_timing.py and cost_model.json).
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'harness'))
import numpy as np
import moderngl
import flip_evaluator as flip
from skimage.metrics import structural_similarity
import taau_sim2 as T
import persp_scene as P
import taa_good as TG
import gpu_taau as G

RESULTS = os.path.join(HERE, 'results')
SCENES = ['pan', 'wires', 'floor', 'fastfloor', 'shadow', 'deform', 'smoke', 'noise', 'zoomin',
          'zoomout', 'panstop', 'pausestop', 'static']
MOVING = ['pan', 'wires', 'floor', 'fastfloor', 'deform', 'smoke', 'noise', 'zoomout']
STILL = ['static', 'pausestop']
# HWTAA modes: render scale per axis and window frames (eevee_game_temporal.cc window_frames_for).
MODES = [('native', 1.0, 8), ('ultra', 1 / 1.3, 14), ('quality', 1 / 1.5, 14),
         ('balanced', 1 / 1.7, 14), ('performance', 0.5, 12)]

_ctx = None


def ctx():
    global _ctx
    if _ctx is None:
        # Headless EGL on Linux; the platform default elsewhere (WGL on Windows). HWTAA_GL_BACKEND
        # overrides it (egl, x11, wgl).
        backend = os.environ.get('HWTAA_GL_BACKEND') or ('egl' if sys.platform.startswith('linux') else None)
        _ctx = moderngl.create_standalone_context(require=430, **({'backend': backend} if backend else {}))
    return _ctx


class NoAA:
    """One sample at each pixel center, no history."""

    def __init__(self, scene):
        self.scene = scene

    def frame(self, t):
        H, W = T.DISPLAY
        ry, rx = np.mgrid[0:H, 0:W].astype(float)
        return self.scene.shading_noise(self.scene.sample(rx + 0.5, ry + 0.5, t)[0], t)


class Recorder:
    """Keeps the frames a method returns, for the metrics PerspBench does not compute."""

    def __init__(self, method):
        self.method, self.outs = method, []

    def frame(self, t):
        out = self.method.frame(t)
        self.outs.append(out)
        return out


def hwtaa(bench, scale, frames, shaders=None, env=None):
    os.environ['TAAU_SHADERS'] = shaders or os.path.join(HERE, 'shaders')
    for k in ('TAAU_REACTIVE', 'TAAU_DEFINES'):
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    return G.GpuTAAU(ctx(), bench.scene, scale, frames)


def srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def image_metrics(bench, outs):
    """FLIP (LDR, 67 ppd) and SSIM of the tone mapped sRGB frames, mean over the measured ones."""
    fl, ss = [], []
    for t in range(bench.start, bench.frames):
        ref, out = srgb(T.tonemap(bench.refs[t])), srgb(T.tonemap(outs[t]))
        fl.append(flip.evaluate(ref, out, 'LDR', applyMagma=False)[1])
        ss.append(structural_similarity(ref, out, channel_axis=-1, data_range=1.0))
    return {'flip': float(np.mean(fl)), 'ssim': float(np.mean(ss))}


def run(bench, method):
    rec = Recorder(method)
    r = {k: float(v) for k, v in bench.run(rec).items()}
    r.update(image_metrics(bench, rec.outs))
    return r, rec.outs


def save(name, data):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)


def load(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def cmd_quality(scenes):
    data = load('quality.json')
    for scene in scenes:
        b = P.PerspBench(scene)
        rows = {'noaa': NoAA(b.scene), 'taa_point': TG.GoodTAA(b.scene, P.Z_WALL, k=0.0),
                'taa_gauss': TG.GoodTAA(b.scene, P.Z_WALL)}
        out = {}
        for name, m in rows.items():
            out[name] = run(b, m)[0]
            print(scene, name, out[name], flush=True)
        for name, scale, n in MODES:
            out['hwtaa_' + name] = run(b, hwtaa(b, scale, n))[0]
            print(scene, name, out['hwtaa_' + name], flush=True)
        data[scene] = out
        save('quality.json', data)


# One component off at a time: (name, shader patch {old: new}, harness env, frames override).
ABLATIONS = [
    ('full', {}, {}, None),
    ('history_only', {}, {}, 2),
    ('window_only', {"""  const bool use_history = window_mix < 1.0;""":
                     """  window_mix = (window_frames > 1) ? 1.0 : window_mix;
  const bool use_history = window_mix < 1.0;"""}, {}, None),
    ('no_missing_fill', {"""    if (missing_weight > 0.0) {""": """    if (false) {"""}, {}, None),
    ('no_locks', {"""    if (geometric_edge) {
      lock_life = 0.0;""": """    if (true) {
      lock_life = 0.0;"""}, {}, None),
    ('no_thin_front', {"""    thin_front = front_count <= THIN_FRONT_TEXELS;""":
                       """    thin_front = false;"""}, {}, None),
    ('no_deformed_window', {"""#define DEFORM_WINDOW_FRAMES 6""": """#define DEFORM_WINDOW_FRAMES 1"""}, {}, None),
    ('no_reactive', {}, {'TAAU_REACTIVE': 'none'}, None),
]
ABLATION_SCENES = ['pan', 'wires', 'floor', 'fastfloor', 'deform', 'smoke', 'shadow', 'pausestop', 'static']


def patched_shaders(name, patch):
    if not patch:
        return os.path.join(HERE, 'shaders')
    d = os.path.join(HERE, 'build', 'ablation_' + name)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(os.path.join(HERE, 'shaders')):
        s = open(os.path.join(HERE, 'shaders', f)).read()
        if f.endswith('resolve_comp.glsl'):
            for a, b in patch.items():
                assert s.count(a) == 1, (name, a)
                s = s.replace(a, b)
        open(os.path.join(d, f), 'w').write(s)
    return d


def cmd_ablations(scenes):
    data = load('ablations.json')
    for scene in scenes:
        b = P.PerspBench(scene)
        for mode, scale, n in (('native', 1.0, 8), ('quality', 1 / 1.5, 14)):
            for name, patch, env, frames in ABLATIONS:
                if name == 'no_reactive' and scene != 'smoke':
                    continue
                r = run(b, hwtaa(b, scale, frames or n, patched_shaders(name, patch), env))[0]
                data.setdefault(scene, {}).setdefault(mode, {})[name] = r
                print(scene, mode, name, r, flush=True)
                save('ablations.json', data)


def cmd_window(scenes):
    data = load('window.json')
    for scene in scenes:
        b = P.PerspBench(scene)
        for scale in (1.0, 1 / 1.5, 0.5):
            for n in (4, 6, 8, 10, 12, 14, 16):
                r = run(b, hwtaa(b, scale, n))[0]
                data.setdefault(scene, {}).setdefault(f'{scale:.4f}', {})[str(n)] = r
                print(scene, scale, n, r['psnr'], flush=True)
                save('window.json', data)


def cmd_ghost(_):
    """Error behind the moving disk by frames since it uncovered the pixel (pan scene)."""
    b = P.PerspBench('pan')
    tm = T.tonemap
    data = {}
    for mode, scale, n in (('native', 1.0, 8), ('quality', 1 / 1.5, 14), ('performance', 0.5, 12)):
        methods = {'taa': TG.GoodTAA(b.scene, P.Z_WALL, k=0.0),
                   'history_only': hwtaa(b, scale, 2),
                   'no_missing_fill': hwtaa(b, scale, n, patched_shaders('no_missing_fill', ABLATIONS[3][1])),
                   'hwtaa': hwtaa(b, scale, n)}
        for name, m in methods.items():
            if name == 'taa' and mode != 'native':
                continue
            outs = [m.frame(t) for t in range(b.frames)]
            buckets = {k: [] for k in range(1, 9)}
            for t in range(b.start, b.frames):
                e = np.abs(tm(outs[t]) - tm(b.refs[t])).mean(-1)
                now = b.keys[t] == 2
                since = np.zeros(now.shape, int)
                for k in range(8, 0, -1):
                    since[b.keys[t - k] == 2] = k
                for k in range(1, 9):
                    sel = (since == k) & ~now
                    if sel.any():
                        buckets[k].append(float(e[sel].mean()))
            data.setdefault(mode, {})[name] = {k: 1000 * float(np.mean(v)) for k, v in buckets.items()}
            print(mode, name, data[mode][name], flush=True)
    save('ghost.json', data)


FIGURE_SCENES = {'pan': 45, 'wires': 45, 'smoke': 45, 'fastfloor': 45, 'deform': 45}


def cmd_frames(_):
    arrays = {}
    for scene, t_show in FIGURE_SCENES.items():
        b = P.PerspBench(scene)
        arrays[f'{scene}/reference'] = b.refs[t_show]
        methods = [('noaa', NoAA(b.scene)), ('taa', TG.GoodTAA(b.scene, P.Z_WALL, k=0.0))]
        methods += [('hwtaa_' + name, hwtaa(b, scale, n)) for name, scale, n in MODES
                    if name in ('native', 'quality', 'performance')]
        for name, m in methods:
            outs = [m.frame(t) for t in range(t_show + 1)]
            arrays[f'{scene}/{name}'] = outs[t_show]
        print(scene, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, 'frames.npz'), **arrays)


if __name__ == '__main__':
    cmd = sys.argv[1]
    scenes = sys.argv[2:] or {'ablations': ABLATION_SCENES, 'window': ['pan', 'wires', 'floor']}.get(cmd, SCENES)
    t0 = time.time()
    globals()['cmd_' + cmd](scenes)
    print(f'{cmd}: {time.time() - t0:.0f} s')
