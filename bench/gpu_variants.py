# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""gpu_variants.py [--res 1080p]: resolve time of the performance variants of variants.py on this
GPU, and of shorter windows (option D).

Per scene ('pan': every pixel moving; 'static') and mode (native, quality), the ring, history and
locks are filled once with real frames; every variant then runs the last frame's resolve with the
same inputs (one second of back-to-back dispatches, median of 100 GL timer queries). Its output is
compared with the frame's: B must match it exactly, C (half floats in shared memory) and W
(cooperative window: other rounding of the same arithmetic) to within rounding (PSNR of the
difference reported).
"""
import argparse
import os
import re
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'harness'))
sys.path.insert(0, HERE)
import taau_sim2 as T
import variants as V

RES = {'bench': (108, 192), '1080p': (1080, 1920), '1440p': (1440, 2560), '4k': (2160, 3840)}
ap = argparse.ArgumentParser()
ap.add_argument('--res', default='1080p')
ap.add_argument('--frames', type=int, default=20)
ap.add_argument('--iters', type=int, default=100)
ap.add_argument('--variants', default='base,W')
ap.add_argument('--scenes', default='static,pan,floor')
ap.add_argument('--lengths', action='store_true', help='also time shorter windows (D)')
args = ap.parse_args()

import moderngl
backend = os.environ.get('HWTAA_GL_BACKEND')
ctx = moderngl.create_standalone_context(require=430, **({'backend': backend} if backend else {}))
if hasattr(ctx, 'gc_mode'):
    ctx.gc_mode = 'auto'
print('GPU:', ctx.info['GL_RENDERER'], '|', ctx.info['GL_VERSION'])
H, W = RES[args.res]
T.DISPLAY = (H, W)
import persp_scene as P
import gpu_taau as G

source = G.source('resolve')
UNIFORMS = re.findall(r'^uniform \w+ (\w+);$', source, re.M)
programs = {}
for v in args.variants.split(','):
    try:
        programs[v] = ctx.compute_shader(V.apply(source, v))
    except Exception as e:
        print(f'variant {v}: compile error\n{e}')


def gpu_ms(run, warmup_s=1.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        for _ in range(10):
            run()
        ctx.finish()
    queries = []
    for _ in range(args.iters):
        q = ctx.query(time=True)
        with q:
            run()
        queries.append(q)
    ctx.finish()
    return float(np.median([q.elapsed for q in queries])) / 1e6


def resolve_run(g, prog):
    r = g.current
    def run():
        g.ring.use(0)
        g.color_tx.use(7)
        g.depth_tx.use(1)
        g.id_tx.use(2)
        g.vector_tx.use(3)
        g.history[r].use(4)
        g.reactive_tx.use(5)
        g.lock[r].use(6)
        g.motion_buf.bind_to_storage_buffer(0)
        g.chain_buf.bind_to_storage_buffer(1)
        g.window_buf.bind_to_uniform_block(0)
        g.history[1 - r].bind_to_image(0, read=False, write=True)
        g.lock[1 - r].bind_to_image(1, read=False, write=True)
        prog.run((g.W + 7) // 8, (g.H + 7) // 8)
        ctx.memory_barrier()
    return run, g.history[1 - r]


def tonemap(x):
    return x / (1.0 + x)


print('variants: B prefetch, C half-float shared memory, W cooperative window (see variants.py)')
for mode, scale, n in (('native', 1.0, 8), ('quality', 1 / 1.5, 14)):
    for scene in args.scenes.split(','):
        g = G.GpuTAAU(ctx, P.PerspScene(scene), scale, n)
        for t in range(args.frames):
            out = g.frame(t)
        values = {}
        for name in UNIFORMS:
            try:
                values[name] = g.resolve[name].value
            except KeyError:
                pass  # optimized out
        times = {}
        for v, prog in programs.items():
            for name, value in values.items():
                try:
                    prog[name].value = value
                except KeyError:
                    pass
            run, target = resolve_run(g, prog)
            target.write(np.zeros((H, W, 4), np.float16).tobytes())
            times[v] = gpu_ms(run)
            again = np.frombuffer(target.read(), np.float16).reshape(H, W, 4)[..., :3].astype(np.float64)
            err = np.mean((tonemap(again) - tonemap(out)) ** 2)
            check = 'same' if err == 0 else (f'{10 * np.log10(1 / err):.0f} dB' if err > 0 else '')
            times[v] = (times[v], check)
        base = times['base'][0]
        print(f'{args.res} {mode:8s} {scene:6s} ' + '  '.join(
            f'{v}: {ms:6.3f} ({100 * (ms / base - 1):+.0f}%, {chk})' if v != 'base' else f'base: {ms:6.3f} ms ({chk})'
            for v, (ms, chk) in times.items()), flush=True)
        if scene == 'pan' and args.lengths:
            prog = programs['base']
            line = []
            for frames in range(6, n + 1, 2):
                prog['window_frames'].value = frames
                line.append(f'N={frames}: {gpu_ms(resolve_run(g, prog)[0]):6.3f}')
            prog['window_frames'].value = n
            print(f'{args.res} {mode:8s} pan    D (base shader) ' + '  '.join(line), flush=True)
        del g, out
