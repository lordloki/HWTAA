# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""gpu_profile.py [--res 1080p]: where the resolve's time goes on this GPU.

1. Compiler statistics of the store and resolve programs from the driver's program binary
   (NVIDIA: instruction and register counts of its assembly; needs PyOpenGL). The full binary is
   saved to profile_<pass>.bin for inspection.
2. Resolve time on the 'pan' (all pixels moving) and 'static' scenes with the window length
   forced to 1 (history alone), 2, 4, 8 and the mode's own: the history path, the window's fixed
   cost and its cost per frame.

Timing as in gpu_timing.py: one second of back-to-back dispatches, then the median of 100 GL timer
queries of the last frame's dispatch.
"""
import argparse
import os
import re
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'harness'))
import taau_sim2 as T

RES = {'bench': (108, 192), '1080p': (1080, 1920), '1440p': (1440, 2560), '4k': (2160, 3840)}
ap = argparse.ArgumentParser()
ap.add_argument('--res', default='1080p')
ap.add_argument('--frames', type=int, default=20)
ap.add_argument('--iters', type=int, default=100)
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


def resolve_run(g):
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
        g.resolve.run((g.W + 7) // 8, (g.H + 7) // 8)
        ctx.memory_barrier()
    return run


def binary_stats(program, name):
    """Program binary through PyOpenGL (moderngl does not expose it)."""
    try:
        from OpenGL import GL
    except ImportError:
        print(f'{name}: no compiler statistics (pip install PyOpenGL)')
        return
    try:
        length = GL.glGetProgramiv(program.glo, GL.GL_PROGRAM_BINARY_LENGTH)
        length = int(np.asarray(length).ravel()[0])
        size = GL.GLsizei(0)
        fmt = GL.GLenum(0)
        buf = (GL.GLubyte * length)()
        GL.glGetProgramBinary(program.glo, length, size, fmt, buf)
        blob = bytes(buf)[:size.value]
    except Exception as e:  # driver without program binaries, or no current context
        print(f'{name}: program binary unavailable ({e})')
        return
    open(os.path.join(HERE, f'profile_{name}.bin'), 'wb').write(blob)
    stats = sorted(set(m.decode() for m in re.findall(rb'#\s*\d+ instructions[^\n\r]*', blob)))
    lmem = len(re.findall(rb'lmem', blob))
    print(f'{name}: {len(blob)} bytes binary; ' + ('; '.join(stats) if stats else 'no statistics found') +
          (f'; {lmem} lmem references' if lmem else ''))


S_pan, S_static = P.PerspScene('pan'), P.PerspScene('static')
for mode, scale, n in (('native', 1.0, 8), ('quality', 1 / 1.5, 14)):
    for S in (S_static, S_pan):
        g = G.GpuTAAU(ctx, S, scale, n)
        for t in range(args.frames):
            g.frame(t)
        if mode == 'native' and S is S_static:
            binary_stats(g.store, 'store')
            binary_stats(g.resolve, 'resolve')
        run = resolve_run(g)
        line = []
        for frames in sorted({f for f in ((1, 2, 4, 8, n) if S is S_pan else (1, n)) if f <= n}):
            g.resolve['window_frames'].value = frames
            line.append(f'N={frames}: {gpu_ms(run):6.3f}')
        g.resolve['window_frames'].value = n
        print(f'{args.res} {mode:8s} {S.mode:6s} resolve ms  ' + '  '.join(line), flush=True)
        del g
