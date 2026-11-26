# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
CPU reference of eevee_game_temporal_resolve_comp.glsl, version 2 (feature flags).

Scenes (Scene.mode):
  pan     camera pans at sub-pixel speed, a striped disk (moving object) crosses the wall.
  static  nothing moves.
  shadow  static camera and geometry, a shadow band slides over the wall (shading change).
  wires   pan + two 0.4 px dark cables in front, with parallax (they move 1.6x the wall).

Keys: 1 wall, 2 disk (moving object), 3 cables (static in world, closer: parallax).
Reprojection is exact per key, as the delta matrix + depth unprojection would give.
Reference: 6x6 box supersampling at display resolution.
"""
import numpy as np

DISPLAY = (108, 192)  # H, W


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, mode):
        self.mode = mode
        self.parallax = {1: 1.0, 2: 1.0, 3: 1.6}
        # Engine knowledge for idea 4: objects receiving dynamic light or shadow.
        self.volatile_keys = {1} if mode == 'shadow' else set()

    def camera(self, t):
        if self.mode in ('static', 'shadow'):
            return np.array([0.0, 0.0])
        return np.array([0.37 * t, 0.13 * t])

    def disk_center(self, t):
        if self.mode == 'static':
            return np.array([90.0, 55.0])
        if self.mode == 'shadow':
            return np.array([-100.0, 55.0])
        return np.array([-30.0 + 1.7 * t, 55.0])

    def object_motion(self, key):
        """World motion per frame of moving objects (the delta matrix)."""
        if self.mode in ('pan', 'wires'):
            return np.where(key[..., None] == 2, np.array([1.7, 0.0]), 0.0)
        return np.zeros(key.shape + (2,))

    def screen_delta(self, key, t):
        """Screen position now minus screen position in the previous frame, per key."""
        cam_delta = self.camera(t) - self.camera(t - 1)
        par = np.vectorize(lambda k: self.parallax[int(k)])(key) if key.size else key
        return self.object_motion(key) - cam_delta * par[..., None]

    def is_idle(self, t):
        return self.mode == 'static'

    def sample(self, x, y, t):
        cam = self.camera(t)
        wx, wy = x + cam[0], y + cam[1]
        rgb = np.empty(x.shape + (3,))
        rgb[...] = (0.25 + 0.05 * np.sin(wx * 0.05))[..., None]
        u = wx * np.cos(0.35) + wy * np.sin(0.35)
        rgb[np.mod(u, 9.0) < 0.35] = 0.9
        checker = (wy > 70) & (wy < 100) & ((np.floor(wx / 2) + np.floor(wy / 2)) % 2 == 0)
        rgb[checker] = 0.6
        circle = (wx - 60) ** 2 + (wy - 35) ** 2 < 30 ** 2
        rgb[circle] = rgb[circle] * 0.4 + np.array([0.8, 0.5, 0.2]) * 0.6
        dots = ((np.mod(wx, 23.0) - 11.5) ** 2 + (np.mod(wy, 23.0) - 11.5) ** 2) < 0.3 ** 2
        rgb[dots] = 12.0
        if self.mode == 'shadow':
            sx = 10.0 + 1.5 * t
            rgb[(x > sx) & (x < sx + 25)] *= 0.25
        depth = np.full(x.shape, 10.0)
        key = np.ones(x.shape, dtype=np.int64)
        c = self.disk_center(t)
        dx, dy = wx - c[0], wy - c[1]
        disk = dx * dx + dy * dy < 16.0 ** 2
        stripes = np.mod(dx + 0.3 * dy, 5.0) < 2.5
        rgb[disk & stripes] = np.array([0.9, 0.3, 0.1])
        rgb[disk & ~stripes] = np.array([0.2, 0.1, 0.05])
        depth[disk] = 5.0
        key[disk] = 2
        if self.mode == 'wires':
            p = self.parallax[3]
            cx, cy = x + cam[0] * p, y + cam[1] * p
            w1 = np.abs((cx - 20.0) * np.sin(0.3) - (cy - 30.0) * np.cos(0.3)) < 0.2
            w2 = np.abs((cx - 0.0) * np.sin(0.04) - (cy - 80.0) * np.cos(0.04)) < 0.2
            wire = w1 | w2
            rgb[wire] = 0.02
            depth[wire] = 3.0
            key[wire] = 3
        return rgb, depth, key

    def reference(self, t, ss=6):
        H, W = DISPLAY
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        acc = np.zeros((H, W, 3))
        for j in range(ss):
            for i in range(ss):
                acc += self.sample(xs + (i + 0.5) / ss, ys + (j + 0.5) / ss, t)[0]
        return acc / (ss * ss)

    def wire_mask(self, t):
        """Display pixels within ~1 px of a cable."""
        H, W = DISPLAY
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        hit = np.zeros((H, W), bool)
        for j in range(4):
            for i in range(4):
                hit |= self.sample(xs + (i + 0.5) / 4, ys + (j + 0.5) / 4, t)[2] == 3
        out = hit.copy()
        out[1:] |= hit[:-1]
        out[:-1] |= hit[1:]
        out[:, 1:] |= hit[:, :-1]
        out[:, :-1] |= hit[:, 1:]
        return out


def halton(index, base):
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


# ---------------------------------------------------------------- shared math
def max3(v):
    return v.max(axis=-1)


TONEMAP = 1.0


def to_work(rgb):
    rgb = np.maximum(rgb, 0.0)
    rgb = rgb / (1.0 + TONEMAP * max3(rgb))[..., None]
    return np.stack([rgb @ np.array([0.25, 0.5, 0.25]),
                     rgb @ np.array([0.5, 0.0, -0.5]),
                     rgb @ np.array([-0.25, 0.5, -0.25])], axis=-1)


def from_work(w):
    y, co, cg = w[..., 0], w[..., 1], w[..., 2]
    rgb = np.maximum(np.stack([y + co - cg, y + cg, y - co - cg], axis=-1), 0.0)
    return rgb / np.maximum(1.0 - TONEMAP * max3(rgb), 1.0 / 65504.0)[..., None]


def ycocg(rgb):
    return np.stack([rgb @ np.array([0.25, 0.5, 0.25]),
                     rgb @ np.array([0.5, 0.0, -0.5]),
                     rgb @ np.array([-0.25, 0.5, -0.25])], axis=-1)


def rgb_of(w):
    y, co, cg = w[..., 0], w[..., 1], w[..., 2]
    return np.stack([y + co - cg, y + cg, y - co - cg], axis=-1)


def texel_clamped(tex, x, y):
    H, W = tex.shape[:2]
    return tex[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)]


def bilinear(tex, x, y):
    x = x - 0.5
    y = y - 0.5
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = texel_clamped(tex, x0, y0) * (1 - fx) + texel_clamped(tex, x0 + 1, y0) * fx
    b = texel_clamped(tex, x0, y0 + 1) * (1 - fx) + texel_clamped(tex, x0 + 1, y0 + 1) * fx
    return a * (1 - fy) + b * fy


def gather(tex, x, y):
    x0 = np.floor(x - 0.5).astype(int)
    y0 = np.floor(y - 0.5).astype(int)
    return np.stack([texel_clamped(tex, x0 + ox, y0 + oy) for oy in (0, 1) for ox in (0, 1)], -1)


def catmull_rom(tex, uv, size):
    pos = np.stack([uv[..., 0] * size[1], uv[..., 1] * size[0]], axis=-1)
    t1 = np.floor(pos - 0.5) + 0.5
    f = pos - t1
    w0 = f * (-0.5 + f * (1.0 - 0.5 * f))
    w1 = 1.0 + f * f * (-2.5 + 1.5 * f)
    w2 = f * (0.5 + f * (2.0 - 1.5 * f))
    w3 = f * f * (-0.5 + 0.5 * f)
    w12 = w1 + w2
    p0, p3, p12 = t1 - 1.0, t1 + 2.0, t1 + w2 / w12
    taps = [((p12[..., 0], p0[..., 1]), w12[..., 0] * w0[..., 1]),
            ((p0[..., 0], p12[..., 1]), w0[..., 0] * w12[..., 1]),
            ((p12[..., 0], p12[..., 1]), w12[..., 0] * w12[..., 1]),
            ((p3[..., 0], p12[..., 1]), w3[..., 0] * w12[..., 1]),
            ((p12[..., 0], p3[..., 1]), w12[..., 0] * w3[..., 1])]
    acc, wsum = 0.0, 0.0
    for (x, y), w in taps:
        acc = acc + bilinear(tex, x, y) * w[..., None]
        wsum = wsum + w
    return acc / wsum[..., None]


def lanczos2(tex, uv, size, drop_corners=False):
    """Separable Lanczos-2 over the 4x4 texels around the sample point."""
    px = uv[..., 0] * size[1] - 0.5
    py = uv[..., 1] * size[0] - 0.5
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    fx, fy = px - x0, py - y0

    def lw(d):
        d = np.abs(d)
        out = np.where(d < 1e-5, 1.0, 2.0 * np.sin(np.pi * d) * np.sin(np.pi * d / 2.0) /
                       np.maximum(np.pi * np.pi * d * d, 1e-12))
        return np.where(d < 2.0, out, 0.0)

    acc, wsum = 0.0, 0.0
    for j in range(-1, 3):
        wy = lw(fy - j)
        for i in range(-1, 3):
            if drop_corners and i in (-1, 2) and j in (-1, 2):
                continue
            w = lw(fx - i) * wy
            acc = acc + texel_clamped(tex, x0 + i, y0 + j) * w[..., None]
            wsum = wsum + w
    return acc / wsum[..., None]


# ---------------------------------------------------------------- resolve
QUADS = (0b000011011, 0b000110110, 0b011011000, 0b110110000)


class Ours:
    RECON = 4.0
    GAMMA = 1.0
    DEPTH_TOL = 0.02
    SPATIAL_CONF = 1.0
    MOTION_FALLOFF = 0.5
    CLIP_CONF_POWER = 2.0
    LOCK_SIMILAR = 1.05
    LOCK_KILL = 1.25
    LOCK_EDGE_DEPTH = 1.02
    # Ideas.
    GEO_LOCKS = False         # 1
    GEO_ABSENT_FRAMES = None  # 1: frames a geometric lock survives without its object in the 3x3
    VOLATILE = False          # 4
    VOLATILE_GAMMA = 1.0      # 4
    VOLATILE_KILLS_LOCKS = True
    HISTORY = 'cr'            # 5: 'cr', 'lanczos', 'lanczos12'
    ANISO = 0.0               # 6: 0 = isotropic reconstruction, 1 = fully edge aligned
    IDLE = False              # 7
    FRONT_VALID = False       # 1b: accept history for static surfaces in front of the old ones
    HYBRID = False            # clip in tone mapped space, accumulate in linear space
    HYBRID_MOTION = 0.0       # > 0: blend toward tone mapped accumulation with motion (px)
    SRAA = False              # 8: 2x2 geometry prepass (key + depth), edge-only reconstruction
    IDLE_PHASES = 64
    IDLE_CONF = 256.0

    SHARPEN = 0.0             # output-only sharpening, scaled by motion (not fed back)
    GATHER_JITTER = False     # validation footprint centered with the previous jitter

    def __init__(self, scene, scale, display=None, **flags):
        for k, v in flags.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)
        self.display = display or DISPLAY
        H, W = self.display
        self.scene = scene
        self.render = (max(1, round(H * scale)), max(1, round(W * scale)))
        ratio = 1.0 / scale
        self.phases = int(np.clip(np.ceil(8 * ratio * ratio), 8, 64))
        self.max_conf = float(np.clip(8 * ratio * ratio, 8, 32))
        if self.GEO_ABSENT_FRAMES is None:
            self.GEO_ABSENT_FRAMES = int(np.ceil(2 * ratio * ratio)) + 1
        self.history = None
        self.lock = None  # life, luma, key (0 = luma lock)
        self.prev_depth = None
        self.prev_key = None

    def frame(self, t):
        S = self.scene
        H, W = self.display
        rh, rw = self.render
        sx, sy = rw / W, rh / H
        idle = self.IDLE and S.is_idle(t)
        phases = self.IDLE_PHASES if idle else self.phases
        max_conf = self.IDLE_CONF if idle else self.max_conf
        idx = (t % phases) + 1
        jit = np.array([halton(idx, 2), halton(idx, 3)]) - 0.5

        ry, rx = np.mgrid[0:rh, 0:rw].astype(float)
        color, zworld, key = S.sample((rx + 0.5 - jit[0]) / sx, (ry + 0.5 - jit[1]) / sy, t)
        depth = 1.0 - zworld / 100.0
        work = to_work(color)
        work_lin = ycocg(np.maximum(color, 0.0))
        moving = key == 2
        volatile = np.isin(key, list(S.volatile_keys)) if S.volatile_keys else np.zeros_like(moving)

        oy, ox = np.mgrid[0:H, 0:W].astype(float)
        cx, cy = (ox + 0.5) * sx, (oy + 0.5) * sy
        bx = np.floor(cx + jit[0]).astype(int)
        by = np.floor(cy + jit[1]).astype(int)
        if self.SRAA:
            # Geometry-only prepass at 2x2 the render resolution: subsample (i, j) of texel t sits
            # at t + 0.25 + 0.5 * (i, j) - jitter (render pixels). Only key and depth.
            gy, gx = np.mgrid[0:2 * rh, 0:2 * rw].astype(float)
            gpx = ((gx + 0.5) * 0.5 - jit[0]) / sx
            gpy = ((gy + 0.5) * 0.5 - jit[1]) / sy
            _, gz, gkey = S.sample(gpx, gpy, t)

        cls = []
        cs, ds, ks, ms, vs, dxs, dys = [], [], [], [], [], [], []
        for i in range(9):
            ox_, oy_ = i % 3 - 1, i // 3 - 1
            tx, ty = np.clip(bx + ox_, 0, rw - 1), np.clip(by + oy_, 0, rh - 1)
            cs.append(work[ty, tx])
            cls.append(work_lin[ty, tx])
            ds.append(depth[ty, tx])
            ks.append(key[ty, tx])
            ms.append(moving[ty, tx])
            vs.append(volatile[ty, tx])
            dxs.append((bx + ox_ + 0.5 - jit[0] - cx) / sx)
            dys.append((by + oy_ + 0.5 - jit[1] - cy) / sy)
        C = np.stack(cs, -2)            # H W 9 3
        CL = np.stack(cls, -2) if self.HYBRID else C   # values that get averaged
        D = np.stack(ds, -1)            # H W 9
        K = np.stack(ks, -1)
        M = np.stack(ms, -1)
        Y = C[..., 0]
        DX = np.stack(dxs, -1)
        DY = np.stack(dys, -1)
        mean = C.mean(-2)
        sigma = np.sqrt(np.maximum((C * C).mean(-2) - mean * mean, 0.0))

        # Reconstruction weights: isotropic, or stretched along the local edge (idea 6).
        gx = (Y[..., 2] + 2 * Y[..., 5] + Y[..., 8]) - (Y[..., 0] + 2 * Y[..., 3] + Y[..., 6])
        gy = (Y[..., 6] + 2 * Y[..., 7] + Y[..., 8]) - (Y[..., 0] + 2 * Y[..., 1] + Y[..., 2])
        gl = np.sqrt(gx * gx + gy * gy)
        nx = np.where(gl > 1e-3, gx / np.maximum(gl, 1e-9), 0.0)[..., None]
        ny = np.where(gl > 1e-3, gy / np.maximum(gl, 1e-9), 0.0)[..., None]
        iso = np.exp(-self.RECON * (DX * DX + DY * DY))
        if self.ANISO > 0.0:
            along = DX * -ny + DY * nx
            across = DX * nx + DY * ny
            aniso_w = np.exp(-self.RECON * (0.5 * along * along + 2.0 * across * across))
            strength = self.ANISO * np.clip(gl / 0.05, 0.0, 1.0)[..., None]
            w = iso + (aniso_w - iso) * strength
        else:
            w = iso
        wsum = w.sum(-1)
        wmax = iso.max(-1)  # coverage: how close the nearest sample is, independent of shape
        current = (CL * w[..., None]).sum(-2) / wsum[..., None]
        sraa_edge = np.zeros((H, W), bool)
        if self.SRAA:
            # Subsamples of the 3x3 texels inside the output pixel (box, the ground truth filter).
            # Each takes the color of the nearest shaded samples of its own object.
            acc = np.zeros((H, W, 3))
            cnt = np.zeros((H, W))
            keys_seen_min = np.full((H, W), 1 << 30)
            keys_seen_max = np.full((H, W), -1)
            near_c = np.zeros((H, W, 3))
            near_d = np.full((H, W), np.inf)
            for i in range(9):
                ox_, oy_ = i % 3 - 1, i // 3 - 1
                for sj in range(2):
                    for si in range(2):
                        gxi = np.clip(2 * (bx + ox_) + si, 0, 2 * rw - 1)
                        gyi = np.clip(2 * (by + oy_) + sj, 0, 2 * rh - 1)
                        skey = gkey[gyi, gxi]
                        # Subsample position in output pixels relative to the pixel center.
                        spx = ((2 * (bx + ox_) + si + 0.5) * 0.5 - jit[0]) / sx - (ox + 0.5)
                        spy = ((2 * (by + oy_) + sj + 0.5) * 0.5 - jit[1]) / sy - (oy + 0.5)
                        inside = (np.abs(spx) < 0.5) & (np.abs(spy) < 0.5)
                        # Color: shaded samples of the same object, Gaussian in render pixels.
                        cw = np.zeros((H, W))
                        cc = np.zeros((H, W, 3))
                        for k in range(9):
                            same = K[..., k] == skey
                            ddx = (DX[..., k] - spx) * sx
                            ddy = (DY[..., k] - spy) * sy
                            wk = np.where(same, np.exp(-2.0 * (ddx * ddx + ddy * ddy)), 0.0)
                            cc += CL[..., k, :] * wk[..., None]
                            cw += wk
                        col = np.where((cw > 1e-6)[..., None], cc / np.maximum(cw, 1e-12)[..., None], current)
                        acc += np.where(inside[..., None], col, 0.0)
                        cnt += inside
                        keys_seen_min = np.where(inside, np.minimum(keys_seen_min, skey), keys_seen_min)
                        keys_seen_max = np.where(inside, np.maximum(keys_seen_max, skey), keys_seen_max)
            sraa_edge = (cnt > 0) & (keys_seen_min != keys_seen_max)
            current = np.where(sraa_edge[..., None], acc / np.maximum(cnt, 1)[..., None], current)

        closest_i = np.argmax(D, axis=-1)
        # Ties: keep the center (shader starts from the center and needs strictly greater).
        center_best = D[..., 4] >= D.max(-1)
        closest_i = np.where(center_best, 4, closest_i)
        take = lambda A: np.take_along_axis(A, closest_i[..., None], -1)[..., 0]
        kc = take(K)
        zc = 100.0 * (1.0 - take(D))
        zfar = 100.0 * (1.0 - D.min(-1))

        delta = S.screen_delta(kc, t)
        uvp = np.stack([(ox + 0.5 - delta[..., 0]) / W, (oy + 0.5 - delta[..., 1]) / H], -1)

        valid = np.full((H, W), self.history is not None)
        valid &= (uvp[..., 0] >= 0) & (uvp[..., 0] <= 1) & (uvp[..., 1] >= 0) & (uvp[..., 1] <= 1)
        if self.history is not None:
            gxp, gyp = uvp[..., 0] * rw, uvp[..., 1] * rh
            if getattr(self, 'GATHER_JITTER', False):
                gxp, gyp = gxp + self.prev_jit[0], gyp + self.prev_jit[1]
            pk = gather(self.prev_key, gxp, gyp)
            pz = gather(self.prev_depth, gxp, gyp)
            same = pk == kc[..., None]
            zmin = np.where(same, pz, 1e30).min(-1)
            zmax = np.where(same, pz, 0.0).max(-1)
            id_ok = same.any(-1) & (zc >= zmin * (1 - self.DEPTH_TOL)) & (zc <= zmax * (1 + self.DEPTH_TOL))
            if self.FRONT_VALID:
                # A static surface in front of everything that was there could not have been
                # hidden: it was just not sampled (thin geometry, stochastic alpha).
                front = (zc < pz.min(-1) * (1 - self.DEPTH_TOL)) & ~take(M)
                id_ok |= front
            valid &= id_ok

        conf = np.zeros((H, W))
        hist = np.zeros((H, W, 3))
        hist_rgb = np.zeros((H, W, 3))
        if self.history is not None:
            if self.HISTORY == 'cr':
                h = catmull_rom(self.history, uvp, (H, W))
            else:
                h = lanczos2(self.history, uvp, (H, W), drop_corners=self.HISTORY == 'lanczos12')
            hist = np.where(valid[..., None], to_work(h[..., :3]), 0.0)
            hist_rgb = np.where(valid[..., None], np.maximum(h[..., :3], 0.0), 0.0)
            conf = np.where(valid, np.maximum(h[..., 3], 0.0), 0.0)

        # ---- Locks.
        life = np.zeros((H, W))
        lluma = np.zeros((H, W))
        lkey = np.zeros((H, W), dtype=np.int64)
        median = np.median(Y, -1)
        if self.lock is not None:
            lx = np.clip(np.floor(uvp[..., 0] * W).astype(int), 0, W - 1)
            ly = np.clip(np.floor(uvp[..., 1] * H).astype(int), 0, H - 1)
            prev = self.lock[ly, lx]
            life = np.where(valid, prev[..., 0], 0.0)
            lluma = prev[..., 1]
            lkey = prev[..., 2].astype(np.int64)
            ratio = (np.maximum(median, lluma) + 0.01) / (np.minimum(median, lluma) + 0.01)
            life = np.where(ratio > self.LOCK_KILL, 0.0, life)
            geo = lkey != 0
            present = (K == lkey[..., None]).any(-1)
            life = np.where(geo & present, 1.0, life)  # refreshed while its object is around
            decay = np.where(geo, 1.0 / self.GEO_ABSENT_FRAMES, 1.0 / phases)
            life = np.maximum(life - decay, 0.0)
        depth_edge = zfar > zc * self.LOCK_EDGE_DEPTH
        any_moving = M.any(-1)
        geo_now = lkey != 0
        kill = np.where(geo_now, any_moving, depth_edge)
        if self.VOLATILE and self.VOLATILE_KILLS_LOCKS:
            kill |= volatile_center(vs)
        life = np.where(kill, 0.0, life)
        locked = life > 0.0

        # Luma locks (thin shading features).
        yc = Y[..., 4]
        r = np.maximum(Y, yc[..., None]) / np.maximum(np.minimum(Y, yc[..., None]), 1e-4)
        sim = r < self.LOCK_SIMILAR
        sim[..., 4] = True
        bits = (sim * (1 << np.arange(9))).sum(-1)
        dis_min = np.where(sim, np.inf, Y).min(-1)
        dis_max = np.where(sim, 0.0, Y).max(-1)
        ridge = (yc > dis_max) | (yc < dis_min)
        thin = ridge.copy()
        for q in QUADS:
            thin &= (bits & q) != q
        new_luma = thin & ~depth_edge
        if self.VOLATILE and self.VOLATILE_KILLS_LOCKS:
            new_luma &= ~volatile_center(vs)
        # Geometric locks (idea 1): center object differs from its neighbors and is in front.
        new_geo = np.zeros((H, W), bool)
        if self.GEO_LOCKS:
            kcen = K[..., 4]
            same_obj = K == kcen[..., None]
            gbits = (same_obj * (1 << np.arange(9))).sum(-1)
            others = ~same_obj
            zcen = 100.0 * (1.0 - D[..., 4])
            zall = 100.0 * (1.0 - D)
            in_front = np.where(others, zcen[..., None] < zall * (1.0 - self.DEPTH_TOL), True).all(-1)
            gthin = others.any(-1) & in_front
            for q in QUADS:
                gthin &= (gbits & q) != q
            new_geo = gthin & ~M[..., 4] & ~any_moving
            if self.VOLATILE:
                new_geo &= ~volatile_center(vs)
        fresh = (new_luma | new_geo) & (life <= 0.0)
        lluma = np.where(fresh, median, lluma)
        lkey = np.where(fresh, np.where(new_geo, K[..., 4], 0), lkey)
        life = np.where(new_luma | new_geo, 1.0, life)

        # ---- Spatial fallback for fresh pixels.
        spatial = np.clip(1.0 - conf / self.SPATIAL_CONF, 0.0, 1.0)
        # Edge weights use render pixel distances, as in the shader.
        along_r = (DX * -ny + DY * nx) * sx
        across_r = (DX * nx + DY * ny) * sx
        ew = np.exp(-(0.5 * along_r * along_r + 4.0 * across_r * across_r))
        edge = (CL * ew[..., None]).sum(-2) / ew.sum(-1)[..., None]
        use = (spatial > 0) & (gl > 1e-3) & ~sraa_edge
        current = np.where(use[..., None], current + (edge - current) * spatial[..., None], current)

        # ---- Clip, confidence, blend.
        motion_px = np.sqrt((uvp[..., 0] * W - (ox + 0.5)) ** 2 + (uvp[..., 1] * H - (oy + 0.5)) ** 2)
        gamma = np.full((H, W), self.GAMMA)
        if self.VOLATILE:
            gamma = np.where(volatile_center(vs), self.VOLATILE_GAMMA, gamma)
        d = hist - mean
        units = np.abs(d / np.maximum(sigma * gamma[..., None], 1e-5)).max(-1)
        clip = (units > 1.0) & ~locked
        hist = np.where(clip[..., None], mean + d / np.maximum(units, 1e-9)[..., None], hist)
        conf = np.where(clip, conf / np.maximum(units, 1.0) ** self.CLIP_CONF_POWER, conf)
        cap = np.maximum(max_conf / (1.0 + motion_px * self.MOTION_FALLOFF), 1.0)
        conf_new = np.minimum(conf + wmax, cap)
        alpha = np.clip(wmax / conf_new, 0.0, 1.0)
        if self.HYBRID and self.HYBRID_MOTION > 0.0:
            # Average in a space between linear (still) and tone mapped (moving): bias-free where
            # the history converges, firefly-robust where it smears.
            sm = np.clip(motion_px / self.HYBRID_MOTION, 0.0, 1.0)[..., None]
            def to_s(rgb):
                rgb = np.maximum(rgb, 0.0)
                return ycocg(rgb / (1.0 + sm * max3(rgb)[..., None]))
            def from_s(w):
                rgb = np.maximum(rgb_of(w), 0.0)
                return rgb / np.maximum(1.0 - sm * max3(rgb)[..., None], 1.0 / 65504.0)
            cur_s = to_s(rgb_of(current))
            hist_s = np.where(clip[..., None], to_s(from_work(hist)), to_s(hist_rgb))
            out = from_s(hist_s + (cur_s - hist_s) * alpha[..., None])
        elif self.HYBRID:
            # Statistics and clip in tone mapped space, averages in linear space.
            hist_lin = np.where(clip[..., None], ycocg(from_work(hist)), ycocg(hist_rgb))
            out = np.maximum(rgb_of(hist_lin + (current - hist_lin) * alpha[..., None]), 0.0)
        else:
            out = from_work(hist + (current - hist) * alpha[..., None])

        f16 = lambda a: a.astype(np.float16).astype(float)
        self.history = f16(np.concatenate([out, conf_new[..., None]], -1))
        if self.SHARPEN > 0.0:
            # Compensate the resampling blur on the output only: the history never sees it.
            blur = np.zeros_like(out)
            for j in (-1, 0, 1):
                for i in (-1, 0, 1):
                    wgt = (2 - abs(i)) * (2 - abs(j)) / 16.0
                    blur += texel_clamped(out, ox.astype(int) + i, oy.astype(int) + j) * wgt
            amount = self.SHARPEN * np.clip(motion_px, 0.0, 1.0)[..., None]
            out = np.maximum(out + (out - blur) * amount, 0.0)
        self.lock = np.stack([f16(life), f16(lluma), lkey.astype(float)], -1)
        self.prev_depth = zworld
        self.prev_key = key
        self.prev_jit = jit
        return out


def volatile_center(vs):
    return vs[4]


# ---------------------------------------------------------------- baselines
class NoAA:
    def __init__(self, scene, scale):
        H, W = DISPLAY
        self.scene = scene
        self.render = (max(1, round(H * scale)), max(1, round(W * scale)))

    def frame(self, t):
        H, W = DISPLAY
        rh, rw = self.render
        ry, rx = np.mgrid[0:rh, 0:rw].astype(float)
        color = self.scene.sample((rx + 0.5) * W / rw, (ry + 0.5) * H / rh, t)[0]
        if (rh, rw) == (H, W):
            return color
        oy, ox = np.mgrid[0:H, 0:W].astype(float)
        return bilinear(color, (ox + 0.5) * rw / W, (oy + 0.5) * rh / H)


class BasicTAA:
    def __init__(self, scene):
        self.scene = scene
        self.history = None

    def frame(self, t):
        H, W = DISPLAY
        jit = np.array([halton((t % 8) + 1, 2), halton((t % 8) + 1, 3)]) - 0.5
        ry, rx = np.mgrid[0:H, 0:W].astype(float)
        color, _, key = self.scene.sample(rx + 0.5 - jit[0], ry + 0.5 - jit[1], t)
        lo = np.full((H, W, 3), np.inf)
        hi = np.full((H, W, 3), -np.inf)
        iy, ix = ry.astype(int), rx.astype(int)
        for i in range(9):
            c = texel_clamped(color, ix + i % 3 - 1, iy + i // 3 - 1)
            lo, hi = np.minimum(lo, c), np.maximum(hi, c)
        if self.history is None:
            out = color
        else:
            d = self.scene.screen_delta(key, t)
            h = np.clip(bilinear(self.history, rx + 0.5 - d[..., 0], ry + 0.5 - d[..., 1]), lo, hi)
            out = h + (color - h) * 0.1
        self.history = out
        return out


# ---------------------------------------------------------------- metrics
def tonemap(c):
    return c / (1.0 + c)


class Bench:
    """Caches references; measures frames [start, frames)."""

    def __init__(self, mode, frames=60, start=30):
        self.scene = Scene(mode)
        self.frames, self.start = frames, start
        self.refs = [self.scene.reference(t) for t in range(frames)]
        H, W = DISPLAY
        oy, ox = np.mgrid[0:H, 0:W].astype(float)
        self.keys = [self.scene.sample(ox + 0.5, oy + 0.5, t)[2] for t in range(frames)]
        self.ox = ox
        self.wire = [self.scene.wire_mask(t) for t in range(frames)] if mode == 'wires' else None

    def run(self, method):
        outs = [method.frame(t) for t in range(self.frames)]
        r = {}
        rng = range(self.start, self.frames)
        err = [np.abs(tonemap(outs[t]) - tonemap(self.refs[t])) for t in rng]
        r['psnr'] = np.mean([10 * np.log10(1.0 / np.mean(e ** 2)) for e in err])
        r['flicker'] = 1000 * np.mean([np.mean(np.abs((tonemap(outs[t]) - tonemap(outs[t - 1]))
                                                      - (tonemap(self.refs[t]) - tonemap(self.refs[t - 1]))))
                                       for t in rng])
        mode = self.scene.mode
        if mode in ('pan', 'wires'):
            g = []
            for t, e in zip(rng, err):
                recent = np.zeros(self.keys[t].shape, bool)
                for k in range(1, 9):
                    recent |= self.keys[t - k] == 2
                reg = recent & (self.keys[t] == 1)
                g.append(np.mean(e[reg]))
            r['ghost'] = 1000 * np.mean(g)
        if mode == 'shadow':
            tr = []
            for t, e in zip(rng, err):
                sx = 10.0 + 1.5 * t
                tr.append(np.mean(e[(self.ox + 0.5 > sx - 9) & (self.ox + 0.5 < sx)]))
            r['trail'] = 1000 * np.mean(tr)
        if mode == 'wires':
            r['wire'] = 1000 * np.mean([np.mean(e[self.wire[t]]) for t, e in zip(rng, err)])
        return r
