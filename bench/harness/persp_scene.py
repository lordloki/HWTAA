# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Scene of taau_sim2 made consistent with a real perspective camera, for the GPU harness.

The sim moves each key by a parallax factor; a pinhole camera translating parallel to the wall
moves a point at depth z by f * dT / z. Depths are chosen so both agree: wall 10 (parallax 1),
disk 9.5 (1.0526), cables 6.25 (1.6). The disk keeps its own motion of 1.7 px per frame.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taau_sim2 as T

Z_WALL, Z_DISK, Z_WIRE = 10.0, 9.5, 6.25


class PerspScene(T.Scene):
    def __init__(self, mode):
        super().__init__(mode)
        self.parallax = {1: 1.0, 2: Z_WALL / Z_DISK, 3: Z_WALL / Z_WIRE}

    def camera(self, t):
        if self.mode in ZOOM_MODES:
            return np.array([0.0, 0.0])
        if self.mode == 'fastfloor':
            # Fast pan (about 4 px per frame on the wall, 6 on the near floor).
            return np.array([4.0, 0.0]) * t
        if self.mode in ('panstop', 'pausestop'):
            # Pans, then stops at frame STOP_FRAME: the history takes over from the window.
            return np.array([0.37, 0.13]) * min(t, STOP_FRAME)
        return super().camera(t)

    def disk_center(self, t):
        if self.mode in ('pausestop', 'smoke') + ZOOM_MODES:
            return np.array([90.0, 55.0])
        return super().disk_center(t)

    def is_idle(self, t):
        # pausestop: the disk does not move, so the engine goes idle once the camera stops.
        return self.mode == 'static' or (self.mode == 'pausestop' and t > STOP_FRAME)

    def zoom(self, t):
        """Focal length over FOCAL: zoomin magnifies by ZOOM_RATE per frame, zoomout the reverse
        (it ends at 1)."""
        if self.mode == 'zoomin':
            return np.exp(ZOOM_RATE * t)
        if self.mode == 'zoomout':
            return np.exp(ZOOM_RATE * (ZOOM_END - t))
        return 1.0

    def sample(self, x, y, t):
        if self.mode in ZOOM_MODES:
            # Field of view zoom around the image center: no parallax, depths unchanged.
            H, W = T.DISPLAY
            z = self.zoom(t)
            x, y = W / 2 + (x - W / 2) / z, H / 2 + (y - H / 2) / z
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
        depth = np.full(x.shape, Z_WALL)
        key = np.ones(x.shape, dtype=np.int64)
        c = self.disk_center(t)
        p2 = self.parallax[2]
        dx, dy = x + cam[0] * p2 - c[0], y + cam[1] * p2 - c[1]
        disk = dx * dx + dy * dy < 16.0 ** 2
        if self.mode == 'deform':
            # Wobble: the disk's pattern rotates by an angle that falls with the radius
            # (non-rigid), back and forth.
            th = self.swirl(np.sqrt(dx * dx + dy * dy), t)
            ct, st = np.cos(th), np.sin(th)
            dx, dy = ct * dx + st * dy, -st * dx + ct * dy
        stripes = np.mod(dx + 0.3 * dy, 5.0) < 2.5
        rgb[disk & stripes] = np.array([0.9, 0.3, 0.1])
        rgb[disk & ~stripes] = np.array([0.2, 0.1, 0.05])
        depth[disk] = Z_DISK
        key[disk] = 2
        if self.mode == 'wires':
            p = self.parallax[3]
            cx, cy = x + cam[0] * p, y + cam[1] * p
            w1 = np.abs((cx - 20.0) * np.sin(0.3) - (cy - 30.0) * np.cos(0.3)) < 0.2
            w2 = np.abs((cx - 0.0) * np.sin(0.04) - (cy - 80.0) * np.cos(0.04)) < 0.2
            wire = w1 | w2
            rgb[wire] = 0.02
            depth[wire] = Z_WIRE
            key[wire] = 3
        if self.mode in ('floor', 'fastfloor'):
            # Ground plane Y = FLOOR_Y (world units, camera at height cam_y): row y sees it at
            # depth f * (FLOOR_Y - cam_y) / (y - H / 2) when that is positive and closer.
            H, W = T.DISPLAY
            cam_w = cam * Z_WALL / FOCAL
            dy = y - H / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                zf = np.where(dy < -1e-3, FOCAL * (FLOOR_Y - cam_w[1]) / dy, np.inf)
            on_floor = (zf > 0) & (zf < depth)
            X = (x - W / 2) * zf / FOCAL + cam_w[0]
            Z = zf
            fx, fz = np.where(on_floor, X, 0.0), np.where(on_floor, Z, 0.0)
            check = (np.floor(fx * 2.0) + np.floor(fz * 2.0)) % 2 == 0
            col = np.where(check[..., None], np.array([0.35, 0.38, 0.45]), np.array([0.12, 0.13, 0.16]))
            lines = (np.abs(np.mod(fx, 2.0) - 1.0) < 0.02) | (np.abs(np.mod(fz, 2.0) - 1.0) < 0.02)
            col = np.where(lines[..., None], np.array([0.9, 0.85, 0.3]), col)
            rgb = np.where(on_floor[..., None], col, rgb)
            depth = np.where(on_floor, zf, depth)
            key = np.where(on_floor, 4, key)
            # Sky below row SKY_ROW (wall and disk cut there): a function of the view direction
            # only, and the camera does not rotate, so it is fixed on screen. Stars test thin
            # features at infinity.
            # The wall's bottom edge moves with the wall. TAAU_SKY_SCREEN=1: fixed on screen
            # instead, as before 15 Nov (no real edge does that while the wall scrolls; it
            # penalized filling the uncovered sky, see the resolve's missing_weight).
            edge_y = y + (0.0 if os.environ.get('TAAU_SKY_SCREEN') else cam[1])
            sky = (edge_y > SKY_ROW) & ~on_floor
            g = 0.3 + 0.4 * (y - SKY_ROW) / (H - SKY_ROW)
            sky_col = np.stack([0.2 * g, 0.35 * g, 0.8 * g], -1)
            stars = ((np.mod(x, 13.0) - 6.5) ** 2 + (np.mod(y * 1.7 + np.floor(x / 13.0) * 5.0, 11.0) - 5.5) ** 2) < 0.25 ** 2
            sky_col = np.where(stars[..., None], 6.0, sky_col)
            rgb = np.where(sky[..., None], sky_col, rgb)
            depth = np.where(sky, np.inf, depth)
            key = np.where(sky, 0, key)
        if self.mode == 'smoke':
            rgb, depth, key = self.transparents(x, y, t, cam, rgb, depth, key)
        return rgb, depth, key

    @staticmethod
    def swirl(r, t):
        return SWIRL_ANGLE * np.clip(1.0 - r / 16.0, 0.0, 1.0) * np.sin(SWIRL_RATE * t)

    def deform_motion(self, x, y, t):
        """Screen motion (previous minus current position, pixels) of the disk surface at x, y:
        its translation and the change of its wobble angle."""
        p2 = self.parallax[2]
        center = self.disk_center(t) - self.camera(t) * p2
        center_prev = self.disk_center(t - 1) - self.camera(t - 1) * p2
        qx, qy = x - center[0], y - center[1]
        r = np.sqrt(qx * qx + qy * qy)
        d = self.swirl(r, t - 1) - self.swirl(r, t)
        cd, sd = np.cos(d), np.sin(d)
        px, py = cd * qx - sd * qy, sd * qx + cd * qy
        return px + center_prev[0] - x, py + center_prev[1] - y

    def screen_motion(self, x, y, key, t):
        """Exact screen motion (now minus previous position, pixels) where the CPU TAA baseline's
        camera and disk model does not hold: zoom, the deformed disk. None elsewhere."""
        H, W = T.DISPLAY
        if self.mode in ZOOM_MODES:
            r = self.zoom(t - 1) / self.zoom(t)
            return np.stack([(x - W / 2) * (1.0 - r), (y - H / 2) * (1.0 - r)], -1)
        if self.mode == 'deform':
            dcam = self.camera(t) - self.camera(t - 1)
            par = np.vectorize(lambda k: self.parallax.get(int(k), 1.0))(key)
            m = -dcam[None, None, :] * par[..., None]
            mx, my = self.deform_motion(x, y, t)
            return np.where((key == 2)[..., None], -np.stack([mx, my], -1), m)
        return None

    def shading_noise(self, color, t):
        """The noise scene's stochastic shading, for CPU methods (gpu_taau applies the same)."""
        if self.mode != 'noise':
            return color
        rng = np.random.default_rng(1000 + t)
        n = (rng.random(color.shape[:2]) * 2.0 - 1.0) * np.sqrt(3.0)
        return color * np.maximum(1.0 + NOISE_SIGMA * n, 0.0)[..., None]

    def transparents(self, x, y, t, cam, rgb, depth, key):
        """Forward transparent surfaces composited over the opaque scene, depth not written.

        Smoke: a particle drifting across the screen, its density animated. Glass: a static pane
        at depth Z_GLASS in front of the wall (its own parallax), with streaks. Spark: additive,
        fast, no coverage (alpha 0). Stashes the
        opaque color and both alphas of the last call (reactive mask, metrics)."""
        self.last_opaque = rgb.copy()
        p = Z_WALL / Z_GLASS
        gx, gy = x + cam[0] * p, y + cam[1] * p
        glass = (gx > 105.0) & (gx < 150.0) & (gy > 8.0) & (gy < 62.0)
        streak = np.mod(gx * 0.8 + gy * 0.3, 6.0) < 1.2
        glass_a = np.where(glass, np.where(streak, 0.55, 0.3), 0.0)
        glass_c = np.array([0.3, 0.5, 0.9])
        sx, sy = 30.0 + 1.6 * t, 40.0 + 0.5 * t
        r2 = ((x - sx) ** 2 + (y - sy) ** 2) / SMOKE_RADIUS ** 2
        density = 0.65 + 0.35 * np.sin(0.45 * (x + 0.7 * y) - 0.6 * t)
        smoke_a = np.where(r2 < 1.0, 0.75 * (1.0 - r2) ** 2 * density, 0.0)
        smoke_c = np.array([0.95, 0.9, 0.8])
        rgb = rgb * (1.0 - glass_a[..., None]) + glass_c * glass_a[..., None]
        rgb = rgb * (1.0 - smoke_a[..., None]) + smoke_c * smoke_a[..., None]
        # Spark: additive (emission over a fully transparent surface), no coverage of its own.
        kx, ky = 150.0 - 2.2 * t, 20.0 + 0.9 * t
        k2 = ((x - kx) ** 2 + (y - ky) ** 2) / SPARK_RADIUS ** 2
        spark = np.where(k2 < 1.0, (1.0 - k2) ** 2, 0.0)
        rgb = rgb + spark[..., None] * np.array([2.0, 1.2, 0.4])
        self.last_spark = spark
        self.last_glass_a, self.last_smoke_a = glass_a, smoke_a
        self.last_alpha = 1.0 - (1.0 - glass_a) * (1.0 - smoke_a)
        return rgb, depth, key


NOISE_SIGMA = float(os.environ.get('TAAU_NOISE', '0.3'))
ZOOM_MODES = ('zoomin', 'zoomout')
ZOOM_RATE = float(os.environ.get('TAAU_ZOOM_RATE', '0.03'))
ZOOM_END = 59
SWIRL_ANGLE = 0.6
SWIRL_RATE = 0.15
Z_GLASS = 8.0
SMOKE_RADIUS = 14.0
SPARK_RADIUS = 5.0
FOCAL = 100.0
FLOOR_Y = -1.5
SKY_ROW = 92
STOP_FRAME = 30


class PerspBench(T.Bench):
    def __init__(self, mode, frames=60, start=30):
        self.scene = PerspScene(mode)
        self.frames, self.start = frames, start
        self.refs = [self.scene.reference(t) for t in range(frames)]
        H, W = T.DISPLAY
        oy, ox = np.mgrid[0:H, 0:W].astype(float)
        self.keys, self.smoke_a, self.glass_a, self.spark = [], [], [], []
        for t in range(frames):
            self.keys.append(self.scene.sample(ox + 0.5, oy + 0.5, t)[2])
            if mode == 'smoke':
                self.smoke_a.append(self.scene.last_smoke_a)
                self.glass_a.append(self.scene.last_glass_a)
                self.spark.append(self.scene.last_spark)
        self.ox = ox
        self.wire = [self.scene.wire_mask(t) for t in range(frames)] if mode == 'wires' else None

    def run(self, method):
        if self.scene.mode == 'smoke':
            return self.run_smoke(method)
        if self.scene.mode == 'deform':
            return self.run_deform(method)
        if self.scene.mode not in ('floor', 'fastfloor'):
            return super().run(method)
        outs = [method.frame(t) for t in range(self.frames)]
        tm = T.tonemap
        rng = range(self.start, self.frames)
        err = [np.abs(tm(outs[t]) - tm(self.refs[t])) for t in rng]
        r = {'psnr': np.mean([10 * np.log10(1.0 / np.mean(e ** 2)) for e in err])}
        r['flicker'] = 1000 * np.mean([np.mean(np.abs((tm(outs[t]) - tm(outs[t - 1])) - (tm(self.refs[t]) - tm(self.refs[t - 1])))) for t in rng])
        g, fl = [], []
        for t, e in zip(rng, err):
            recent = np.zeros(self.keys[t].shape, bool)
            for k in range(1, 9):
                recent |= self.keys[t - k] == 2
            g.append(np.mean(e[recent & (self.keys[t] != 2)]))
            fl.append(np.mean(e[self.keys[t] == 4]))
        r['ghost'] = 1000 * np.mean(g)
        r['floor'] = 1000 * np.mean(fl)
        r['sky'] = 1000 * np.mean([np.mean(e[self.keys[t] == 0]) for t, e in zip(rng, err)])
        return r

    def run_smoke(self, method):
        """Errors (x1000) on the smoke now, on where it was in the last 8 frames only (trail),
        on the glass, and on where the additive spark was in the last 8 frames (its trail)."""
        outs = [method.frame(t) for t in range(self.frames)]
        tm = T.tonemap
        rng = range(self.start, self.frames)
        err = [np.abs(tm(outs[t]) - tm(self.refs[t])) for t in rng]
        r = {'psnr': np.mean([10 * np.log10(1.0 / np.mean(e ** 2)) for e in err])}
        r['flicker'] = 1000 * np.mean([np.mean(np.abs((tm(outs[t]) - tm(outs[t - 1])) - (tm(self.refs[t]) - tm(self.refs[t - 1])))) for t in rng])
        sm, tr, gl, sk = [], [], [], []
        for t, e in zip(rng, err):
            now = self.smoke_a[t] > 0.02
            spark = self.spark[t] > 0.02
            recent = np.zeros(now.shape, bool)
            spark_recent = np.zeros(now.shape, bool)
            for k in range(1, 9):
                recent |= self.smoke_a[t - k] > 0.02
                spark_recent |= self.spark[t - k] > 0.02
            sm.append(np.mean(e[now]))
            tr.append(np.mean(e[recent & ~now & ~spark & (self.glass_a[t] == 0.0)]))
            gl.append(np.mean(e[(self.glass_a[t] > 0.0) & ~now & ~spark]))
            sk.append(np.mean(e[spark_recent & ~spark & ~now]))
        r['smoke'] = 1000 * np.mean(sm)
        r['trail'] = 1000 * np.mean(tr)
        r['glass'] = 1000 * np.mean(gl)
        r['spark'] = 1000 * np.mean(sk)
        return r

    def run_deform(self, method):
        """Errors (x1000) on the deformed disk and behind it (ghost)."""
        outs = [method.frame(t) for t in range(self.frames)]
        tm = T.tonemap
        rng = range(self.start, self.frames)
        err = [np.abs(tm(outs[t]) - tm(self.refs[t])) for t in rng]
        r = {'psnr': np.mean([10 * np.log10(1.0 / np.mean(e ** 2)) for e in err])}
        r['flicker'] = 1000 * np.mean([np.mean(np.abs((tm(outs[t]) - tm(outs[t - 1])) - (tm(self.refs[t]) - tm(self.refs[t - 1])))) for t in rng])
        dk, g = [], []
        for t, e in zip(rng, err):
            now = self.keys[t] == 2
            recent = np.zeros(now.shape, bool)
            for k in range(1, 9):
                recent |= self.keys[t - k] == 2
            dk.append(np.mean(e[now]))
            g.append(np.mean(e[recent & ~now]))
        r['disk'] = 1000 * np.mean(dk)
        r['ghost'] = 1000 * np.mean(g)
        return r
