# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Runs the real store + resolve shaders (GLSL 4.3 on llvmpipe) on the CPU reference scenes.

The CPU side mirrors TemporalModule (eevee_game_temporal.cc): motion table, motion chains,
frame records, window buffer, ring rotation. Jitter and idle follow the simulation so outputs
can be compared frame by frame with ideas.Combo.
"""
import os
import sys
import numpy as np
import moderngl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taau_sim2 as T
import persp_scene as P

HERE = os.path.dirname(os.path.abspath(__file__))
SHADERS = os.path.join(HERE, '..', 'shaders')
CHECK = HERE

KEY_MASK = 0xFFFF
SLOT_SHIFT = 16
MOVED_BIT = 0x20000000
VOLATILE_BIT = 0x40000000
DEFORM_BIT = 0x80000000
WINDOW_MAX = 16
AGE_STRIDE = 10
AGE_REPROJECT, AGE_VIEWPROJ, AGE_DEPTH_PARAMS, AGE_JITTER = 0, 4, 8, 9
WINDOW_MIN_FRAMES = 4
FOCAL = 100.0  # pixels
NEAR, FAR = 1.0, 1000.0


def source(name):
    shaders = os.environ.get('TAAU_SHADERS', SHADERS)
    defines = os.environ.get('TAAU_DEFINES', '').replace(';', '\n')
    def strip(path):
        return '\n'.join(l for l in open(path).read().split('\n')
                         if not l.startswith(('#include', 'COMPUTE_SHADER_CREATE_INFO', '#pragma once')))
    return ('#version 430\n' + defines + '\n' + open(f'{CHECK}/common.glsl').read() +
            open(f'{CHECK}/{name}_res.glsl').read() +
            strip(f'{shaders}/eevee_game_temporal_lib.glsl') +
            strip(f'{shaders}/eevee_game_temporal_{name}_comp.glsl'))


WINDOW_UNIT = np.sqrt(4.0 * 1.4426950408889634)
NOISE_SIGMA = float(os.environ.get('TAAU_NOISE', '0.3'))


def window_matrix(reproject, depth_params, jitter, rw, rh, W, H):
    # TemporalModule window_matrix(): texel and depth to NDC, reprojection, window units.
    texel = np.eye(4)
    texel[0, 0], texel[1, 1] = 2.0 / rw, 2.0 / rh
    texel[0, 3] = (0.5 - jitter[0]) * 2.0 / rw - 1.0
    texel[1, 3] = (0.5 - jitter[1]) * 2.0 / rh - 1.0
    a, b, persp = depth_params[0], depth_params[1], depth_params[2] != 0.0
    depth = np.eye(4)
    depth[2, 2] = b if persp else -a
    depth[2, 3] = -a if persp else b
    out = np.eye(4)
    hx, hy = 0.5 * W * WINDOW_UNIT, 0.5 * H * WINDOW_UNIT
    out[0, 0] = out[0, 3] = hx
    out[1, 1] = out[1, 3] = hy
    return out @ reproject @ depth @ texel


def depth_params_for(winmat):
    a, b, persp = winmat[2, 2], winmat[2, 3], winmat[3, 2] != 0.0
    return (a, b, 1.0 if persp else 0.0, 0.0 if persp else (b - 1.0) / a)


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def colmajor(m):
    return tuple(float(v) for v in np.asarray(m, dtype=np.float64).T.flatten())


class GpuTAAU:
    def __init__(self, ctx, scene, scale, ring_layers):
        self.ctx = ctx
        self.scene = scene
        H, W = T.DISPLAY
        self.W, self.H = W, H
        self.rh, self.rw = max(1, round(H * scale)), max(1, round(W * scale))
        ratio = 1.0 / scale
        self.phases = int(np.clip(np.ceil(8 * ratio * ratio), 8, 64))
        self.max_conf = float(np.clip(8 * ratio * ratio, 8, 32))
        self.L = ring_layers if ring_layers >= WINDOW_MIN_FRAMES else 2
        self.window_volatile = scale >= 0.6 and self.L >= 8
        # TemporalModule moving_phase_count_ (MOVING_PHASES_MIN_SCALE).
        self.move_phases = min(self.L, self.phases) if scale >= 0.6 and self.L >= WINDOW_MIN_FRAMES else self.phases
        if os.environ.get('TAAU_MOVE_PHASES'):  # experiments only
            self.move_phases = int(os.environ['TAAU_MOVE_PHASES'])
        self.store = ctx.compute_shader(source('store'))
        self.resolve = ctx.compute_shader(source('resolve'))

        def tex(size, comps, dtype, filt=moderngl.NEAREST):
            t = ctx.texture(size, comps, dtype=dtype)
            t.filter = (filt, filt)
            t.repeat_x = t.repeat_y = False
            return t

        rw, rh = self.rw, self.rh
        self.color_tx = tex((rw, rh), 4, 'f2')
        self.depth_tx = tex((rw, rh), 1, 'f4')
        self.id_tx = tex((rw, rh), 1, 'u2')
        self.vector_tx = tex((rw, rh), 4, 'f2')
        self.vector_tx.write(np.zeros((rh, rw, 4), np.float16).tobytes())
        # TAAU_REACTIVE: none (1x1 zero, no transparency_end()), alpha (exact coverage of the
        # transparent surfaces) or shader (the reactive pass on the opaque and final colors).
        self.reactive = os.environ.get('TAAU_REACTIVE', 'shader') if scene.mode == 'smoke' else 'none'
        if self.reactive == 'none':
            self.reactive_tx = tex((1, 1), 1, 'f1')
            self.reactive_tx.write(np.zeros(1, np.uint8).tobytes())
        else:
            self.reactive_tx = tex((rw, rh), 1, 'f1', moderngl.LINEAR)
        if self.reactive == 'shader':
            self.reactive_cs = ctx.compute_shader(source('reactive'))
            self.opaque_tx = tex((rw, rh), 4, 'f1')
        self.history = [tex((W, H), 4, 'f2', moderngl.LINEAR) for _ in range(2)]
        self.lock = [tex((W, H), 2, 'f1') for _ in range(2)]
        for t in self.history + self.lock:
            t.write(np.zeros((H, W, t.components), np.float16 if t.dtype == 'f2' else np.uint8).tobytes())
        self.ring = ctx.texture_array((rw, rh, self.L), 2, dtype='u4')
        self.ring.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ring.repeat_x = self.ring.repeat_y = False
        # TAAU_TIMING=1: GPU time of the store and resolve dispatches (ns), per frame.
        self.timing = os.environ.get('TAAU_TIMING') == '1'
        self.times = []
        self.motion_buf = ctx.buffer(reserve=16 * 4 * 2)
        self.chain_buf = ctx.buffer(reserve=16 * 3 * WINDOW_MAX * 4)
        self.window_buf = ctx.buffer(reserve=16 * AGE_STRIDE * WINDOW_MAX)

        self.current = 0
        self.ring_head = 0
        self.window_frames = 1
        self.frames = {}
        self.mover = None  # disk: {'mats': {frame: 4x4}, 'moved': frame}
        self.keys16 = [0, 0x1001, 0x2002, 0x3003, 0x4004]

    # Camera ------------------------------------------------------------------------------
    def winmat(self, t=0):
        W, H = self.W, self.H
        focal = FOCAL * self.scene.zoom(t)
        m = np.zeros((4, 4))
        m[0, 0] = 2 * focal / W
        m[1, 1] = 2 * focal / H
        m[2, 2] = -(FAR + NEAR) / (FAR - NEAR)
        m[2, 3] = -2 * FAR * NEAR / (FAR - NEAR)
        m[3, 2] = -1.0
        if os.environ.get('TAAU_ORTHO'):
            # Orthographic, same scale as the perspective camera on the wall.
            m = np.zeros((4, 4))
            m[0, 0] = 2 * FOCAL / (W * P.Z_WALL)
            m[1, 1] = 2 * FOCAL / (H * P.Z_WALL)
            m[2, 2] = -2.0 / (FAR - NEAR)
            m[2, 3] = -(FAR + NEAR) / (FAR - NEAR)
            m[3, 3] = 1.0
        return m

    def viewmat(self, t):
        # Screen shift of the wall (depth Z_WALL) is the sim's camera offset.
        c = self.scene.camera(t) * P.Z_WALL / FOCAL
        return translate(-c[0], -c[1], 0.0)

    def disk_matrix(self, t):
        # World translation that puts the disk where the sim draws it (its own motion only).
        c = self.scene.disk_center(t)
        return translate(c[0] * P.Z_DISK / FOCAL, c[1] * P.Z_DISK / FOCAL, 0.0)

    # Frame -------------------------------------------------------------------------------
    def frame(self, t):
        S, ctx = self.scene, self.ctx
        W, H, rw, rh = self.W, self.H, self.rw, self.rh
        sx, sy = rw / W, rh / H
        idle = S.is_idle(t)
        phases = 64 if idle else self.phases
        # The CPU knows the camera moved one frame late (camera_moved_ of the last resolve).
        if not idle and t > 1 and np.any(S.camera(t - 1) != S.camera(t - 2)):
            phases = self.move_phases
        max_conf = 256.0 if idle else self.max_conf
        idx = (t % phases) + 1
        jit = np.array([T.halton(idx, 2), T.halton(idx, 3)]) - 0.5

        # Prepass + shading at render resolution.
        ry, rx = np.mgrid[0:rh, 0:rw].astype(float)
        color, z, key = S.sample((rx + 0.5 - jit[0]) / sx, (ry + 0.5 - jit[1]) / sy, t)
        if S.mode == 'noise':
            # Stochastic shading (one sample per pixel GI or AO): every render sample gets its
            # own noise each frame, unit variance uniform times NOISE_SIGMA. The reference is
            # the noise free mean.
            rng = np.random.default_rng(1000 + t)
            n = (rng.random(key.shape) * 2.0 - 1.0) * np.sqrt(3.0)
            color = color * np.maximum(1.0 + NOISE_SIGMA * n, 0.0)[..., None]
        winmat = self.winmat(t)
        with np.errstate(invalid='ignore', divide='ignore'):
            ndc = (winmat[2, 2] * -z + winmat[2, 3]) / (z if winmat[3, 2] != 0 else 1.0)
            depth = np.where(np.isinf(z), 0.0, 1.0 - (ndc + 1.0) * 0.5)
        rgba = np.concatenate([color, np.ones(color.shape[:2] + (1,))], -1)
        self.color_tx.write(rgba.astype(np.float16).tobytes())
        self.depth_tx.write(depth.astype(np.float32).tobytes())
        self.id_tx.write(key.astype(np.uint16).tobytes())
        if self.reactive == 'alpha':
            self.reactive_tx.write(np.round(np.clip(S.last_alpha, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes())
        elif self.reactive == 'shader':
            # transparency_begin() on the opaque color, transparency_end() on the final one.
            cs = self.reactive_cs
            cs['render_size'].value = (rw, rh)
            self.opaque_tx.bind_to_image(0, read=True, write=True)
            self.reactive_tx.bind_to_image(1, read=False, write=True)
            for capture, c in ((True, S.last_opaque), (False, color)):
                self.color_tx.write(np.concatenate([c, np.ones(c.shape[:2] + (1,))], -1).astype(np.float16).tobytes())
                self.color_tx.use(0)
                cs['capture'].value = capture
                cs.run((rw + 7) // 8, (rh + 7) // 8)
                ctx.memory_barrier()

        # Sync: motion table and chains (TemporalModule::end_sync).
        history_valid = t > 0
        ages = self.L - 1
        entries = [0] + [self.keys16[i] for i in (1, 2, 3, 4)]
        if S.volatile_keys:
            for k in S.volatile_keys:
                entries[k] |= VOLATILE_BIT
        chains = np.zeros((max(ages, 1) * 3, 4), np.float32)
        if S.mode in ('pan', 'wires', 'floor', 'panstop', 'fastfloor', 'noise'):
            prev, curr = self.disk_matrix(t - 1), self.disk_matrix(t)
            if self.mover is None:
                self.mover = {'mats': {f: prev for f in range(t - WINDOW_MAX, t)}, 'moved': t}
            m = self.mover
            m['mats'][t - 1] = prev
            m['mats'][t] = curr
            m['moved'] = t
            inv = np.linalg.inv(m['mats'][t])
            for age in range(1, ages + 1):
                c = m['mats'][t - age] @ inv
                chains[(age - 1) * 3:(age - 1) * 3 + 3] = c[:3]
            entries[2] |= (1 << SLOT_SHIFT) | MOVED_BIT
        if S.mode == 'deform':
            # Deformed: no chain, the velocity prepass gives the motion (previous minus current
            # UV, packed * 0.01).
            entries[2] |= DEFORM_BIT
            mx, my = S.deform_motion((rx + 0.5 - jit[0]) / sx, (ry + 0.5 - jit[1]) / sy, t)
            on = key == 2
            vec = np.zeros((rh, rw, 4))
            vec[..., 0] = np.where(on, mx / W, 0.0) * 0.01
            vec[..., 1] = np.where(on, my / H, 0.0) * 0.01
            self.vector_tx.write(vec.astype(np.float16).tobytes())
        self.motion_buf.write(np.array(entries, np.uint32).tobytes())
        self.chain_buf.write(chains.tobytes())

        # Resolve (TemporalModule::resolve).
        self.ring_head = (self.ring_head + 1) % self.L
        self.window_frames = min(self.window_frames + 1, self.L) if history_valid else 1
        viewmat = self.viewmat(t)
        vp = winmat @ viewmat
        rec = dict(viewproj=vp, viewprojinv=np.linalg.inv(vp),
                   depth_params=depth_params_for(winmat), jitter=jit,
                   layer=self.ring_head)
        self.frames[t] = rec
        window = np.zeros((WINDOW_MAX * AGE_STRIDE, 4), np.float32)
        for age in range(max(self.window_frames, 2)):
            r = self.frames[t - age] if age < self.window_frames else rec
            reproject = window_matrix(rec['viewproj'] @ r['viewprojinv'], r['depth_params'], r['jitter'], rw, rh, W, H)
            base = age * AGE_STRIDE
            window[base + AGE_REPROJECT:base + AGE_REPROJECT + 4] = reproject.T
            window[base + AGE_VIEWPROJ:base + AGE_VIEWPROJ + 4] = r['viewproj'].T
            window[base + AGE_DEPTH_PARAMS] = r['depth_params']
            window[base + AGE_JITTER] = (r['jitter'][0], r['jitter'][1], r['layer'], 0.0)
        self.window_buf.write(window.tobytes())

        # Store pass.
        st = self.store
        self.color_tx.use(0)
        self.depth_tx.use(1)
        self.id_tx.use(2)
        self.reactive_tx.use(3)
        self.motion_buf.bind_to_storage_buffer(0)
        self.ring.bind_to_image(0, read=False, write=True)
        st['depth_params'].value = rec['depth_params']
        st['render_size'].value = (rw, rh)
        st['motion_len'].value = 5
        st['ring_layer'].value = self.ring_head
        if self.timing:
            q_store = ctx.query(time=True)
            with q_store:
                st.run((rw + 7) // 8, (rh + 7) // 8)
        else:
            st.run((rw + 7) // 8, (rh + 7) // 8)
        ctx.memory_barrier()

        # Resolve pass.
        rs = self.resolve
        prev = 1 - self.current
        self.ring.use(0)
        self.color_tx.use(7)
        self.depth_tx.use(1)
        self.id_tx.use(2)
        self.vector_tx.use(3)
        self.history[prev].use(4)
        self.reactive_tx.use(5)
        self.lock[prev].use(6)
        self.motion_buf.bind_to_storage_buffer(0)
        self.chain_buf.bind_to_storage_buffer(1)
        self.window_buf.bind_to_uniform_block(0)
        self.history[self.current].bind_to_image(0, read=False, write=True)
        self.lock[self.current].bind_to_image(1, read=False, write=True)
        scale_max = max(rw / W, rh / H)
        uniforms = {
            'curr_viewprojinv': colmajor(rec['viewprojinv']),
            'prev_reproject': colmajor((self.frames[t - 1] if self.window_frames > 1 else rec)['viewproj'] @ rec['viewprojinv']),
            'curr_depth_params': rec['depth_params'],
            'render_scale': (rw / W, rh / H),
            'jitter_px': tuple(jit),
            'render_size': (rw, rh),
            'display_size': (W, H),
            'tile_dim': int(np.ceil(7.0 * scale_max)) + 3,
            'motion_len': 5,
            'ring_layer': self.ring_head,
            'window_frames': self.window_frames if self.L >= WINDOW_MIN_FRAMES else 1,
            'chain_stride': (self.L - 1) * 3,
            'max_confidence': max_conf,
            'frequency_split': bool(idle and max(rw / W, rh / H) <= 0.6),
            'lock_decay': 1.0 / phases,
            'history_valid': history_valid,
            'window_volatile': self.window_volatile,
        }
        for name, value in uniforms.items():
            try:
                rs[name].value = value
            except KeyError:
                pass  # optimized out
        if self.timing:
            q_resolve = ctx.query(time=True)
            with q_resolve:
                rs.run((W + 7) // 8, (H + 7) // 8)
            ctx.finish()
            self.times.append((q_store.elapsed, q_resolve.elapsed))
        else:
            rs.run((W + 7) // 8, (H + 7) // 8)
        ctx.memory_barrier()
        out = np.frombuffer(self.history[self.current].read(), np.float16).reshape(H, W, 4)
        self.current = prev
        return out[..., :3].astype(np.float64)
