# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""gpu_timing.py [--res 1080p,1440p,4k] [--modes native,quality,balanced,performance]: GPU time of
the HWTAA store and resolve passes, and of a textbook TAA, on the GPU of this machine's OpenGL 4.3
driver. Run it on real hardware: llvmpipe returns 0 for timer queries.

Content: the bench scenes at the target resolution, 'pan' (the whole image in motion: the worst
case) and 'static' (nothing moves: the history path). Each mode first runs --frames frames on
the CPU scene so that the ring, the history and the locks hold real content.

Timing: the last frame's dispatches are then repeated back to back, with the same inputs, and the
median of 100 GL timer queries is reported. Timing the frames themselves is wrong twice over: the
CPU renders each scene frame for a second or more, so the GPU drops to its idle clocks between
dispatches (10 to 20 times slower memory), and the store pass would include the upload of the
CPU-rendered textures, which a game renders on the GPU.
"""
import argparse
import gc
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'harness'))
import taau_sim2 as T

RES = {'bench': (108, 192), '1080p': (1080, 1920), '1440p': (1440, 2560), '4k': (2160, 3840)}
MODES = {'native': (1.0, 8), 'ultra': (1 / 1.3, 14), 'quality': (1 / 1.5, 14),
         'balanced': (1 / 1.7, 14), 'performance': (0.5, 12)}

ap = argparse.ArgumentParser()
ap.add_argument('--res', default='1080p,1440p,4k')
ap.add_argument('--modes', default='native,quality,balanced,performance')
ap.add_argument('--frames', type=int, default=20)
ap.add_argument('--iters', type=int, default=100)
args = ap.parse_args()

import moderngl
# Platform default context (WGL on Windows, GLX on Linux with a display); HWTAA_GL_BACKEND=egl for
# a headless Linux machine.
backend = os.environ.get('HWTAA_GL_BACKEND')
ctx = moderngl.create_standalone_context(require=430, **({'backend': backend} if backend else {}))
# Release the GL objects of each run when collected: the runs of all resolutions together would
# not fit in VRAM, and paging would distort the times.
if hasattr(ctx, 'gc_mode'):
    ctx.gc_mode = 'auto'
print('GPU:', ctx.info['GL_RENDERER'], '|', ctx.info['GL_VERSION'])


def gpu_ms(run, warmup_s=1.0):
    """Median GPU time of run(), in ms, after warmup_s of back-to-back runs at full clocks."""
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


def store_run(g):
    """The last frame's store dispatch (GpuTAAU.frame() bindings; uniforms stay set)."""
    def run():
        g.color_tx.use(0)
        g.depth_tx.use(1)
        g.id_tx.use(2)
        g.reactive_tx.use(3)
        g.motion_buf.bind_to_storage_buffer(0)
        g.ring.bind_to_image(0, read=False, write=True)
        g.store.run((g.rw + 7) // 8, (g.rh + 7) // 8)
        ctx.memory_barrier()
    return run


def resolve_run(g):
    """The last frame's resolve dispatch, and the history it writes. frame() swapped g.current to
    the history it read."""
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
    return run, g.history[1 - r]


for res in args.res.split(','):
    H, W = RES[res]
    T.DISPLAY = (H, W)
    import persp_scene as P
    import gpu_taau as G
    # Textbook TAA at native resolution (harness/basic_taa.comp): the floor of any TAA.
    taa = ctx.compute_shader(open(os.path.join(HERE, 'harness', 'basic_taa.comp')).read())
    def tex(c, dt):
        t = ctx.texture((W, H), c, dtype=dt)
        t.repeat_x = t.repeat_y = False
        return t
    color, vel, hist = tex(4, 'f2'), tex(2, 'f2'), [tex(4, 'f2'), tex(4, 'f2')]
    color.write((np.random.rand(H, W, 4) * 0.5).astype(np.float16).tobytes())
    vel.write((np.random.rand(H, W, 2) * 0.002).astype(np.float16).tobytes())
    taa['size'].value = (W, H)
    def taa_run():
        color.use(0)
        vel.use(1)
        hist[0].use(2)
        hist[1].bind_to_image(0, read=False, write=True)
        taa.run((W + 7) // 8, (H + 7) // 8)
        ctx.memory_barrier()
    ms = gpu_ms(taa_run)
    print(f'{res:6s} textbook TAA            {ms:6.3f} ms')
    if ms == 0:
        print('       (this driver returns 0 for GL timer queries: run on a hardware GPU)')
    del taa, color, vel, hist
    gc.collect()
    for scene in ('static', 'pan'):
        S = P.PerspScene(scene)
        for mode in args.modes.split(','):
            scale, n = MODES[mode]
            g = G.GpuTAAU(ctx, S, scale, n)
            for t in range(args.frames):
                out = g.frame(t)
            run, target = resolve_run(g)
            # The repeated dispatches must reproduce the last frame from the same inputs: clear
            # the output first, so that a dispatch that writes elsewhere or not at all shows.
            target.write(np.zeros((H, W, 4), np.float16).tobytes())
            st = gpu_ms(store_run(g))
            rs = gpu_ms(run)
            again = np.frombuffer(target.read(), np.float16).reshape(H, W, 4)
            same = np.array_equal(again[..., :3].astype(np.float64), out)
            print(f'{res:6s} {scene:6s} {mode:11s} N{n:2d} store {st:6.3f} ms  resolve {rs:6.3f} ms  '
                  f'total {st + rs:6.3f} ms' + ('' if same else '  (output differs: timing invalid)'),
                  flush=True)
            del g, out
            gc.collect()
