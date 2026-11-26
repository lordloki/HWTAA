# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""make_figures.py: figures of the paper from results/ (PDF for the plots, PNG for the crops).

    crops.png   frames of five scenes at t = 45, every method (frames.npz)
    window.pdf  PSNR versus window length per render scale (window.json)
    ghost.pdf   error behind the moving disk by frames since uncovered (ghost.json)

Labels in English or Spanish: python3 make_figures.py [en|es].
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'
OUT = os.path.join(HERE, '..', 'figures', LANG)
os.makedirs(OUT, exist_ok=True)

# Categorical slots 1-3 of the validated reference palette (white surface: all pairs pass; the
# aqua is below 3:1 contrast, so every series is also labeled directly). Reference baselines are
# a neutral gray, dashed.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY = '#8a8985'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

T = {
    'en': {'reference': 'Reference (36 spp)', 'noaa': 'No AA', 'taa': 'TAA',
           'hwtaa_native': 'HWTAA Native', 'hwtaa_quality': 'HWTAA Quality',
           'hwtaa_performance': 'HWTAA Perf.',
           'scenes': {'pan': 'pan', 'wires': 'cables', 'fastfloor': 'fast floor',
                      'deform': 'deform', 'smoke': 'smoke'},
           'n': 'Window length $N$ (frames)', 'psnr': 'PSNR (dB)',
           'scale': {'1.0000': 'Native (1)', '0.6667': 'Quality (1/1.5)', '0.5000': 'Performance (1/2)'},
           'chosen': 'shipped $N$',
           'since': 'Frames since the disk uncovered the pixel', 'err': 'Mean error ($\\times 1000$)',
           'modes': {'native': 'Native', 'quality': 'Quality', 'performance': 'Performance'},
           'ghost': {'hwtaa': 'HWTAA', 'no_missing_fill': 'no missing-weight fill',
                     'history_only': 'history only', 'taa': 'TAA (native)'}},
    'es': {'reference': 'Referencia (36 mpp)', 'noaa': 'Sin AA', 'taa': 'TAA',
           'hwtaa_native': 'HWTAA Nativo', 'hwtaa_quality': 'HWTAA Calidad',
           'hwtaa_performance': 'HWTAA Rend.',
           'scenes': {'pan': 'paneo', 'wires': 'cables', 'fastfloor': 'suelo rápido',
                      'deform': 'deformación', 'smoke': 'humo'},
           'n': 'Longitud de la ventana $N$ (fotogramas)', 'psnr': 'PSNR (dB)',
           'scale': {'1.0000': 'Nativo (1)', '0.6667': 'Calidad (1/1.5)', '0.5000': 'Rendimiento (1/2)'},
           'chosen': '$N$ elegido',
           'since': 'Fotogramas desde que el disco descubrió el píxel', 'err': 'Error medio ($\\times 1000$)',
           'modes': {'native': 'Nativo', 'quality': 'Calidad', 'performance': 'Rendimiento'},
           'ghost': {'hwtaa': 'HWTAA', 'no_missing_fill': 'sin relleno de peso ausente',
                     'history_only': 'solo historia', 'taa': 'TAA (nativo)'}},
}[LANG]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42,
    'axes.formatter.use_locale': False})
if LANG == 'es':
    # Decimal comma on the axes.
    plt.rcParams['axes.formatter.use_locale'] = False


def comma(ax):
    # Decimal point in both languages (accepted in Spanish by the RAE since 2010).
    return
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('.', ',')))


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def to_srgb8(x):
    """Display transform of the metrics: c / (1 + c), then sRGB. The harness renders with row 0
    at the bottom (GL): flipped for display."""
    x = np.clip(x / (1.0 + x), 0.0, 1.0)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (np.flipud(x) * 255 + 0.5).astype(np.uint8)


# Crops (x0, y0 after the flip, 96 x 54 output pixels).
CROPS = {'pan': (0, 20), 'wires': (0, 20), 'fastfloor': (48, 54), 'deform': (0, 20), 'smoke': (24, 20)}
CW, CH = 96, 54


def fig_crops():
    f = np.load(os.path.join(RES, 'frames.npz'))
    methods = ['reference', 'noaa', 'taa', 'hwtaa_native', 'hwtaa_quality', 'hwtaa_performance']
    scenes = list(CROPS)
    fig, axes = plt.subplots(len(scenes), len(methods), figsize=(7.0, 7.0 * len(scenes) * CH / (len(methods) * CW) + 0.35),
                             gridspec_kw={'wspace': 0.03, 'hspace': 0.03})
    for i, s in enumerate(scenes):
        x0, y0 = CROPS[s]
        for j, m in enumerate(methods):
            ax = axes[i, j]
            img = to_srgb8(f[f'{s}/{m}'])[y0:y0 + CH, x0:x0 + CW]
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(T[m], fontsize=7, color=INK, pad=3)
            if j == 0:
                ax.set_ylabel(T['scenes'][s], fontsize=7, color=INK)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.955, bottom=0.005)
    fig.savefig(os.path.join(OUT, 'crops.png'), dpi=400)
    plt.close(fig)


def fig_window():
    w = json.load(open(os.path.join(RES, 'window.json')))
    scenes = ['pan', 'wires', 'floor']
    chosen = {'1.0000': 8, '0.6667': 14, '0.5000': 12}
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    style(ax)
    for scale, color in (('1.0000', BLUE), ('0.6667', ORANGE), ('0.5000', AQUA)):
        ns = sorted({int(n) for s in scenes for n in w[s][scale]})
        ys = [np.mean([w[s][scale][str(n)]['psnr'] for s in scenes]) for n in ns]
        ax.plot(ns, ys, color=color, linewidth=1.5, marker='o', markersize=3.5,
                markeredgecolor='white', markeredgewidth=0.6, label=T['scale'][scale], zorder=3)
        k = ns.index(chosen[scale])
        ax.plot([ns[k]], [ys[k]], marker='o', markersize=7, markerfacecolor='none',
                markeredgecolor=INK, markeredgewidth=0.9, zorder=4)
        # Direct label at the right end; the two upscaled curves end close, so they are spread.
        dy = {'1.0000': 0, '0.6667': 5, '0.5000': -5}[scale]
        ax.annotate(T['scale'][scale].split(' (')[0], (ns[-1], ys[-1]), xytext=(5, dy),
                    textcoords='offset points', va='center', ha='left', fontsize=7, color=INK)
    ax.plot([], [], marker='o', markersize=7, markerfacecolor='none', markeredgecolor=INK,
            markeredgewidth=0.9, linestyle='none', label=T['chosen'])
    ax.set_xlabel(T['n'])
    ax.set_ylabel(T['psnr'])
    ax.set_xticks(range(4, 17, 2))
    ax.set_xlim(3.5, 16.5)
    comma(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5, handlelength=1.6)
    fig.subplots_adjust(left=0.14, right=0.74, bottom=0.36, top=0.97)
    fig.savefig(os.path.join(OUT, 'window.pdf'))
    plt.close(fig)


def fig_ghost():
    g = json.load(open(os.path.join(RES, 'ghost.json')))
    modes = ['native', 'quality', 'performance']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    taa = g['native']['taa']
    xs = list(range(1, 9))
    for ax, mode in zip(axes, modes):
        style(ax)
        ax.plot(xs, [taa[str(x)] for x in xs], color=GRAY, linewidth=1.2, linestyle=(0, (4, 2)),
                label=T['ghost']['taa'], zorder=2)
        for key, color in (('history_only', AQUA), ('no_missing_fill', ORANGE), ('hwtaa', BLUE)):
            v = g[mode][key]
            ax.plot(xs, [v[str(x)] for x in xs], color=color, linewidth=1.5, marker='o', markersize=3,
                    markeredgecolor='white', markeredgewidth=0.5, label=T['ghost'][key], zorder=3)
        ax.set_title(T['modes'][mode], fontsize=8, color=INK, pad=3)
        ax.set_xticks(xs)
        ax.set_xlabel(T['since'] if mode == 'quality' else '')
        comma(ax)
    axes[0].set_ylabel(T['err'])
    axes[0].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.80, wspace=0.08)
    fig.savefig(os.path.join(OUT, 'ghost.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    fig_crops()
    fig_window()
    fig_ghost()
    print('figures ->', os.path.relpath(OUT))
