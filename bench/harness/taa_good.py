# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Production style TAA baseline (Karis 2014 / UE4 class), native resolution only.

Halton(2, 3) jitter over 8 frames; current frame: the pixel's own jittered sample (k = 0), or a
Gaussian of the 3x3 jittered samples (k > 0: Karis' Blackman-Harris fit is 2.29); motion of the closest sample of the 3x3 (dilation), exact per
pixel from depth and the camera; Catmull-Rom history; variance clip (gamma 1) of the tone mapped
YCoCg history toward the 3x3 mean; blend alpha 0.1 in that space."""
import numpy as np
import taau_sim2 as T


class GoodTAA:
    def __init__(self, scene, z_wall, alpha=0.1, k=2.29, gamma=1.0):
        self.scene, self.z_wall = scene, z_wall
        self.alpha, self.k, self.gamma = alpha, k, gamma
        self.history = None

    def motion(self, depth, key, t):
        S = self.scene
        dcam = S.camera(t) - S.camera(t - 1)
        par = np.where(np.isfinite(depth), self.z_wall / np.where(np.isfinite(depth), depth, 1.0), 0.0)
        d = -dcam[None, None, :] * par[..., None]
        disk = S.disk_center(t) - S.disk_center(t - 1)
        return np.where((key == 2)[..., None], disk[None, None, :] + d, d)

    def frame(self, t):
        S = self.scene
        H, W = T.DISPLAY
        jit = np.array([T.halton((t % 8) + 1, 2), T.halton((t % 8) + 1, 3)]) - 0.5
        ry, rx = np.mgrid[0:H, 0:W].astype(float)
        color, depth, key = S.sample(rx + 0.5 - jit[0], ry + 0.5 - jit[1], t)
        noise = getattr(S, 'shading_noise', None)
        if noise:
            color = noise(color, t)
        iy, ix = ry.astype(int), rx.astype(int)
        acc, wsum = 0.0, 0.0
        m1, m2 = 0.0, 0.0
        best = np.full((H, W), np.inf)
        bx, by = ix.copy(), iy.copy()
        for i in range(9):
            ox, oy = i % 3 - 1, i // 3 - 1
            c = T.texel_clamped(color, ix + ox, iy + oy)
            z = T.texel_clamped(depth, ix + ox, iy + oy)
            d2 = (ox - jit[0]) ** 2 + (oy - jit[1]) ** 2
            w = np.exp(-self.k * d2)
            acc, wsum = acc + c * w, wsum + w
            wk = T.to_work(c)
            m1, m2 = m1 + wk, m2 + wk * wk
            closer = z < best
            best = np.where(closer, z, best)
            bx, by = np.where(closer, ix + ox, bx), np.where(closer, iy + oy, by)
        cur = acc / wsum if self.k > 0 else color
        if self.history is None or t == 0:
            self.history = cur
            return cur
        exact = getattr(S, 'screen_motion', None)
        mot = exact(rx + 0.5, ry + 0.5, key, t) if exact else None
        if mot is None:
            mot = self.motion(depth, key, t)
        mv = mot[np.clip(by, 0, H - 1), np.clip(bx, 0, W - 1)]
        uv = np.stack([(rx + 0.5 - mv[..., 0]) / W, (ry + 0.5 - mv[..., 1]) / H], axis=-1)
        hist = T.catmull_rom(self.history, uv, (H, W))
        mean = m1 / 9.0
        sigma = np.sqrt(np.maximum(m2 / 9.0 - mean * mean, 0.0)) * self.gamma
        hw = T.to_work(hist)
        delta = hw - mean
        ratio = np.max(np.abs(delta) / np.maximum(sigma, 1e-5), axis=-1, keepdims=True)
        hw = np.where(ratio > 1.0, mean + delta / np.maximum(ratio, 1.0), hw)
        valid = np.all((uv >= 0.0) & (uv <= 1.0), axis=-1, keepdims=True)
        cw = T.to_work(cur)
        out = np.where(valid, hw + (cw - hw) * self.alpha, cw)
        self.history = T.from_work(out)
        return self.history
