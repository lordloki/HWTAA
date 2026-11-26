# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""make_tables.py: LaTeX tables of the paper from results/*.json and cost_model.json.

Writes ../tables/<lang>/*.tex (tabular bodies, language dependent labels) and
../tables/numbers.tex (\\newcommand macros for the numbers quoted in the text, shared by both
languages, so the prose never holds a number the bench did not produce).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = os.path.join(HERE, '..', 'tables')

GROUPS = [('motion', ['pan', 'wires', 'floor', 'fastfloor', 'deform', 'smoke', 'noise', 'zoomin',
                      'zoomout']),
          ('stop', ['panstop', 'pausestop']),
          ('still', ['shadow', 'static'])]
SCENES = [s for _, g in GROUPS for s in g]
HW = ['hwtaa_native', 'hwtaa_ultra', 'hwtaa_quality', 'hwtaa_balanced', 'hwtaa_performance']
METHODS = ['noaa', 'taa_point', 'taa_gauss'] + HW
# Render pixels per output pixel.
SCALE = {'noaa': 1.0, 'taa_point': 1.0, 'taa_gauss': 1.0, 'hwtaa_native': 1.0,
         'hwtaa_ultra': 1 / 1.3, 'hwtaa_quality': 1 / 1.5, 'hwtaa_balanced': 1 / 1.7,
         'hwtaa_performance': 0.5}
FRAMES = {'hwtaa_native': 8, 'hwtaa_ultra': 14, 'hwtaa_quality': 14, 'hwtaa_balanced': 14,
          'hwtaa_performance': 12}

L = {
    'en': {
        'noaa': 'No AA', 'taa_point': 'TAA\\textsubscript{pt}', 'taa_gauss': 'TAA\\textsubscript{G}',
        'hwtaa_native': 'Native', 'hwtaa_ultra': 'U.\\,Qual.', 'hwtaa_quality': 'Quality',
        'hwtaa_balanced': 'Balanced', 'hwtaa_performance': 'Perf.',
        'motion': 'Motion', 'stop': 'Motion that stops', 'still': 'Static camera',
        'mean': 'mean', 'all': 'All scenes, mean', 'scene': 'Scene', 'method': 'Method',
        'render': 'Render px', 'psnr_motion': 'PSNR motion', 'psnr_all': 'PSNR all',
        'flip': '\\FLIP', 'ssim': 'SSIM', 'flicker': 'Flicker', 'wins': 'Wins',
        'abl': {'history_only': 'History only (no window)', 'window_only': 'Window only (no history)',
                'no_missing_fill': 'No missing-weight fill', 'no_locks': 'No luma locks',
                'no_thin_front': 'No thin-foreground rule', 'no_deformed_window': 'No window on deformed meshes',
                'no_reactive': 'No reactive mask'},
        'component': 'Variant', 'nscenes': '{} scenes\\textsuperscript{{*}}', 'dnat': '$\\Delta$ Native', 'dq': '$\\Delta$ Quality',
        'worst': 'Worst case', 'target': 'Targeted region',
        'region': {'ghost': 'uncovered', 'wire': 'cables', 'sky': 'sky', 'disk': 'deformed disk',
                   'trail': 'smoke trail', 'floor': 'floor', 'flicker': 'flicker'},
        'res': 'Output', 'mode': 'Mode', 'n': '$N$', 'still_ms': 'Still', 'moving_ms': 'Moving',
        'fsr2': 'FSR 2.2', 'fsr31': 'FSR 3.1', 'mem': 'Memory', 'native_aa': 'Native AA',
        'ultra': 'Ultra Quality', 'quality': 'Quality', 'balanced': 'Balanced', 'performance': 'Performance',
        'native': 'Native',
        'scenes': {'pan': 'pan', 'wires': 'cables', 'floor': 'floor', 'fastfloor': 'fast floor',
                   'deform': 'deform', 'smoke': 'smoke', 'noise': 'noise', 'zoomin': 'zoom in',
                   'zoomout': 'zoom out', 'panstop': 'pan-stop', 'pausestop': 'pan-idle',
                   'shadow': 'shadow', 'static': 'static'},
    },
    'es': {
        'noaa': 'Sin AA', 'taa_point': 'TAA\\textsubscript{p}', 'taa_gauss': 'TAA\\textsubscript{G}',
        'hwtaa_native': 'Nativo', 'hwtaa_ultra': 'U.\\,Cal.', 'hwtaa_quality': 'Calidad',
        'hwtaa_balanced': 'Equil.', 'hwtaa_performance': 'Rend.',
        'motion': 'Movimiento', 'stop': 'Movimiento que se detiene', 'still': 'C\\\'amara fija',
        'mean': 'media', 'all': 'Todas las escenas, media', 'scene': 'Escena', 'method': 'M\\\'etodo',
        'render': 'P\\\'ix. render', 'psnr_motion': 'PSNR mov.', 'psnr_all': 'PSNR total',
        'flip': '\\FLIP', 'ssim': 'SSIM', 'flicker': 'Parpadeo', 'wins': 'Victorias',
        'abl': {'history_only': 'Solo historial (sin ventana)', 'window_only': 'Solo ventana (sin historia)',
                'no_missing_fill': 'Sin relleno del peso ausente', 'no_locks': 'Sin bloqueos de luma',
                'no_thin_front': 'Sin regla de primer plano fino', 'no_deformed_window': 'Sin ventana en mallas deformadas',
                'no_reactive': 'Sin m\\\'ascara reactiva'},
        'component': 'Variante', 'nscenes': '{} escenas\\textsuperscript{{*}}', 'dnat': '$\\Delta$ Nativo', 'dq': '$\\Delta$ Calidad',
        'worst': 'Peor caso', 'target': 'Regi\\\'on afectada',
        'region': {'ghost': 'descubierto', 'wire': 'cables', 'sky': 'cielo', 'disk': 'disco deformado',
                   'trail': 'estela de humo', 'floor': 'suelo', 'flicker': 'parpadeo'},
        'res': 'Salida', 'mode': 'Modo', 'n': '$N$', 'still_ms': 'Quieto', 'moving_ms': 'En mov.',
        'fsr2': 'FSR 2.2', 'fsr31': 'FSR 3.1', 'mem': 'Memoria', 'native_aa': 'Native AA',
        'ultra': 'Ultra Calidad', 'quality': 'Calidad', 'balanced': 'Equilibrado', 'performance': 'Rendimiento',
        'native': 'Nativo',
        'scenes': {'pan': 'paneo', 'wires': 'cables', 'floor': 'suelo', 'fastfloor': 'suelo r\\\'apido',
                   'deform': 'deformaci\\\'on', 'smoke': 'humo', 'noise': 'ruido', 'zoomin': 'zoom +',
                   'zoomout': 'zoom $-$', 'panstop': 'paneo-parada', 'pausestop': 'paneo-reposo',
                   'shadow': 'sombra', 'static': 'est\\\'atica'},
    },
}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float('nan')


def best_taa(q, scene, key='psnr'):
    return max(q[scene]['taa_point'][key], q[scene]['taa_gauss'][key])


# ---- Cost model (cost_model.json): times on the RX 590 / GTX 1060 class, MB. ----
CM = json.load(open(os.path.join(HERE, 'cost_model.json')))
RESOLUTIONS = [('1080p', 1920, 1080), ('1440p', 2560, 1440), ('4K', 3840, 2160)]
MODES = [('native', 1.0, 8), ('ultra', 1 / 1.3, 14), ('quality', 1 / 1.5, 14),
         ('balanced', 1 / 1.7, 14), ('performance', 0.5, 12)]
# AMD's published upscaler times (ms) and memory (MB). FSR 2.2: measured, RX 590 (the GPU the
# cost model is calibrated on). FSR 3.1: RX 5700 XT, a faster GPU, for reference only.
FSR2_MS = {'1080p': {'quality': 1.3, 'balanced': 1.2, 'performance': 1.1},
           '1440p': {'quality': 2.3, 'balanced': 2.1, 'performance': 1.9},
           '4K': {'quality': 5.4, 'balanced': 4.9, 'performance': 4.4}}
FSR2_MB = {'1080p': {'quality': 115, 'balanced': 105, 'performance': 101},
           '1440p': {'quality': 207, 'balanced': 189, 'performance': 172},
           '4K': {'quality': 448, 'balanced': 407, 'performance': 376}}
FSR31_MS = {'1080p': {'native': 1.7, 'quality': 1.2, 'performance': 1.0},
            '1440p': {'native': 3.0, 'quality': 2.0, 'performance': 1.7},
            '4K': {'native': 6.5, 'quality': 4.4, 'performance': 3.7}}
FSR31_MB = {'1080p': {'quality': 75, 'balanced': 65, 'performance': 61},
            '1440p': {'quality': 134, 'balanced': 118, 'performance': 101},
            '4K': {'quality': 292, 'balanced': 256, 'performance': 226}}


def render_extent(w, h, s):
    return int(w * s + 0.5), int(h * s + 0.5)


def cost_ms(w, h, s, n, moving):
    r, st = CM['resolve'], CM['store']
    k = CM['calibration'] * CM['waves_per_simd_1080p'] / CM['clock_hz'] * 1000.0
    pix = (w * h) / (1920 * 1080)
    rw, rh = render_extent(w, h, s)
    rpix = (rw * rh) / (1920 * 1080)
    core = r['moving_base_invthr'] + (n - 1) * r['per_window_frame_invthr'] if moving else r['still_invthr']
    alu = k * (core * pix + st['invthr'] * rpix)
    t = CM['traffic_bytes']
    per_texel = (t['per_render_texel']['store'] + t['per_render_texel']['resolve_inputs'] +
                 (t['per_render_texel']['per_window_frame'] * (n - 1) if moving else
                  t['per_render_texel']['previous_layer']))
    traffic = (rw * rh * per_texel + w * h * t['per_output_pixel']['history_and_locks'])
    mem_ms = traffic / t['bandwidth_bytes_per_s'] * 1000.0
    return max(alu, mem_ms), alu, mem_ms


def memory_mb(w, h, s, n):
    m = CM['memory_bytes']
    rw, rh = render_extent(w, h, s)
    rt = rw * rh
    return (rt * (m['ring_per_render_texel_per_frame'] * n + m['object_id_per_render_texel'] +
                  m['reactive_per_render_texel']) + w * h * m['fixed_per_output_pixel']) / 1e6


def f1(x):
    return f'{x:.1f}'


def write(lang, name, text):
    d = os.path.join(OUT, lang)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, name), 'w').write(text)


def table_quality(q, lang):
    lab = L[lang]
    cols = 'l' + 'r' * len(METHODS)
    rows = ['\\begin{tabular}{' + cols + '}', '\\toprule',
            ' & \\multicolumn{3}{c}{} & \\multicolumn{5}{c}{HWTAA} \\\\',
            '\\cmidrule(lr){5-9}',
            lab['scene'] + ' & ' + ' & '.join(lab[m] for m in METHODS) + ' \\\\', '\\midrule']
    for gname, scenes in GROUPS:
        rows.append(f'\\multicolumn{{{len(METHODS) + 1}}}{{l}}{{\\textit{{{lab[gname]}}}}} \\\\')
        for s in scenes:
            vals = [q[s][m]['psnr'] for m in METHODS]
            b = max(vals)
            cells = [('\\textbf{' + f'{v:.2f}' + '}') if v == b else f'{v:.2f}' for v in vals]
            rows.append('\\quad ' + lab['scenes'][s] + ' & ' + ' & '.join(cells) + ' \\\\')
        vals = [mean(q[s][m]['psnr'] for s in scenes) for m in METHODS]
        rows.append('\\quad\\textit{' + lab['mean'] + '} & ' +
                    ' & '.join(f'\\textit{{{v:.2f}}}' for v in vals) + ' \\\\')
    rows.append('\\midrule')
    vals = [mean(q[s][m]['psnr'] for s in SCENES) for m in METHODS]
    rows.append(lab['all'] + ' & ' + ' & '.join(f'{v:.2f}' for v in vals) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    write(lang, 'quality.tex', '\n'.join(rows) + '\n')


def table_summary(q, lang):
    lab = L[lang]
    motion = GROUPS[0][1]
    rows = ['\\begin{tabular}{lrrrrrrr}', '\\toprule',
            ' & '.join([lab['method'], lab['render'], lab['psnr_motion'] + ' $\\uparrow$',
                        lab['psnr_all'] + ' $\\uparrow$', lab['flip'] + ' $\\downarrow$',
                        lab['ssim'] + ' $\\uparrow$', lab['flicker'] + ' $\\downarrow$',
                        lab['wins']]) + ' \\\\', '\\midrule']
    for m in METHODS:
        if m == 'hwtaa_native':
            rows.append('\\midrule')
        name = lab[m] if not m.startswith('hwtaa') else 'HWTAA ' + lab[m]
        wins = '' if not m.startswith('hwtaa') else \
            f"{sum(q[s][m]['psnr'] > best_taa(q, s) for s in SCENES)}/{len(SCENES)}"
        rows.append(' & '.join([
            name, f'{SCALE[m] ** 2 * 100:.0f}\\,\\%',
            f"{mean(q[s][m]['psnr'] for s in motion):.2f}",
            f"{mean(q[s][m]['psnr'] for s in SCENES):.2f}",
            f"{mean(q[s][m]['flip'] for s in SCENES):.4f}",
            f"{mean(q[s][m]['ssim'] for s in SCENES):.3f}",
            f"{mean(q[s][m]['flicker'] for s in SCENES):.1f}", wins]) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    write(lang, 'summary.tex', '\n'.join(rows) + '\n')


# Per ablation: the scenes it acts on (its PSNR change is their mean; elsewhere it changes
# nothing measurable) and the region metric it targets (scene, metric key).
ABL = {'history_only': (['pan', 'wires', 'floor', 'fastfloor', 'deform', 'smoke'], ('pan', 'ghost')),
       'window_only': (['pausestop', 'static'], ('static', 'flicker')),
       'no_missing_fill': (['pan', 'wires', 'floor', 'fastfloor', 'deform'], ('pan', 'ghost')),
       'no_locks': (['pausestop', 'static', 'floor', 'fastfloor'], ('fastfloor', 'sky')),
       'no_thin_front': (['wires'], ('wires', 'wire')),
       'no_deformed_window': (['deform'], ('deform', 'disk')),
       'no_reactive': (['smoke'], ('smoke', 'trail'))}


def ablation_rows(a):
    """Per ablation and mode: mean PSNR change over the scenes it acts on, relative change of the
    targeted region error, and the largest PSNR change anywhere else (to back 'nothing else')."""
    out = {}
    for name, (scenes, (ts, tk)) in ABL.items():
        r = {}
        for mode in ('native', 'quality'):
            ds = [a[s][mode][name]['psnr'] - a[s][mode]['full']['psnr'] for s in scenes
                  if s in a and name in a[s].get(mode, {})]
            r[mode] = mean(ds)
            rest = [abs(a[s][mode][name]['psnr'] - a[s][mode]['full']['psnr']) for s in a
                    if s not in scenes and name in a[s].get(mode, {})]
            r[mode + '_rest'] = max(rest) if rest else 0.0
            if ts in a and name in a[ts].get(mode, {}):
                full, var = a[ts][mode]['full'][tk], a[ts][mode][name][tk]
                r[mode + '_target'] = 100.0 * (var / full - 1.0)
        out[name] = r
    return out


def table_ablation(a, lang):
    lab = L[lang]
    rows = ['\\begin{tabular}{lrrlrr}', '\\toprule',
            lab['component'] + ' & ' + lab['dnat'] + ' & ' + lab['dq'] +
            ' & \\multicolumn{3}{c}{' + lab['target'] + '} \\\\',
            '\\cmidrule(lr){4-6}',
            ' & dB & dB & & ' + lab['native'] + ' & ' + lab['quality'] + ' \\\\', '\\midrule']
    def pct(x):
        return f'${x:+.0f}$\\,\\%' if x == x else '--'
    for name, r in ablation_rows(a).items():
        scenes, (ts, tk) = ABL[name]
        rows.append(' & '.join([
            lab['abl'][name], f"${r['native']:+.2f}$", f"${r['quality']:+.2f}$",
            lab['region'][tk] + ' (' + lab['scenes'][ts] + ')',
            pct(r.get('native_target', float('nan'))),
            pct(r.get('quality_target', float('nan')))]) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    write(lang, 'ablation.tex', '\n'.join(rows) + '\n')


def table_cost(lang):
    lab = L[lang]
    rows = ['\\begin{tabular}{llrrrrrrrr}', '\\toprule',
            ' & & & \\multicolumn{2}{c}{HWTAA (ms)} & \\multicolumn{2}{c}{FSR (ms)} & '
            '\\multicolumn{3}{c}{' + lab['mem'] + ' (MB)} \\\\',
            '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}',
            ' & '.join([lab['res'], lab['mode'], lab['n'], lab['still_ms'], lab['moving_ms'],
                        '2.2\\textsuperscript{a}', '3.1\\textsuperscript{b}', 'HWTAA',
                        '2.2', '3.1']) + ' \\\\', '\\midrule']
    for ri, (res, w, h) in enumerate(RESOLUTIONS):
        if ri:
            rows.append('\\midrule')
        for mode, s, n in MODES:
            still = cost_ms(w, h, s, n, False)[0]
            moving = cost_ms(w, h, s, n, True)[0]
            def get(t, unit=f1):
                v = t[res].get(mode)
                return unit(v) if v is not None else '--'
            rows.append(' & '.join([
                res if mode == 'native' else '', lab[mode], str(n), f1(still), f1(moving),
                get(FSR2_MS), get(FSR31_MS), f'{memory_mb(w, h, s, n):.0f}',
                get(FSR2_MB, str), get(FSR31_MB, str)]) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    write(lang, 'cost.tex', '\n'.join(rows) + '\n')


def numbers(q, a, w, g):
    """Macros for the numbers quoted in the prose."""
    m = {}
    motion = GROUPS[0][1]
    def put(name, value, fmt='{:.2f}'):
        m[name] = fmt.format(value)
    for meth, tag in [('noaa', 'NoAA'), ('taa_point', 'TAApt'), ('taa_gauss', 'TAAg'),
                      ('hwtaa_native', 'Nat'), ('hwtaa_ultra', 'UQ'), ('hwtaa_quality', 'Q'),
                      ('hwtaa_balanced', 'B'), ('hwtaa_performance', 'P')]:
        put('PsnrMotion' + tag, mean(q[s][meth]['psnr'] for s in motion))
        put('PsnrAll' + tag, mean(q[s][meth]['psnr'] for s in SCENES))
        put('Flicker' + tag, mean(q[s][meth]['flicker'] for s in SCENES), '{:.1f}')
        put('Flip' + tag, mean(q[s][meth]['flip'] for s in SCENES), '{:.4f}')
        if meth.startswith('hwtaa'):
            wins = sum(q[s][meth]['psnr'] > best_taa(q, s) for s in SCENES)
            put('Wins' + tag, wins, '{:d}')
            d = [q[s][meth]['psnr'] - best_taa(q, s) for s in SCENES]
            put('GainMotion' + tag, mean(q[s][meth]['psnr'] - best_taa(q, s) for s in motion))
            put('GainAll' + tag, mean(d))
            put('WorstGain' + tag, min(d))
            put('WorstScene' + tag, L['en']['scenes'][SCENES[d.index(min(d))]], '{}')
            put('WorstSceneEs' + tag, L['es']['scenes'][SCENES[d.index(min(d))]], '{}')
    put('NumScenes', len(SCENES), '{:d}')
    best_taa_all = mean(max(q[s]['taa_point']['psnr'], q[s]['taa_gauss']['psnr']) for s in SCENES)
    put('PsnrAllBestTAA', best_taa_all)
    # Ablations.
    for name, r in ablation_rows(a).items():
        tag = ''.join(p.capitalize() for p in name.split('_'))
        put('Abl' + tag + 'Nat', r['native'], '{:+.2f}')
        put('Abl' + tag + 'Q', r['quality'], '{:+.2f}')
        if 'native_target' in r:
            put('Abl' + tag + 'TgtNat', r['native_target'], '{:+.0f}')
        if 'quality_target' in r:
            put('Abl' + tag + 'TgtQ', r['quality_target'], '{:+.0f}')
            put('Abl' + tag + 'TgtQAbs', abs(r['quality_target']), '{:.0f}')
        if 'native_target' in r:
            put('Abl' + tag + 'TgtNatAbs', abs(r['native_target']), '{:.0f}')
        put('Abl' + tag + 'Rest', max(r['native_rest'], r['quality_rest']), '{:.2f}')
    for name in ABL:
        tag = ''.join(p.capitalize() for p in name.split('_'))
        for sc in a:
            for mode, mt in (('native', 'Nat'), ('quality', 'Q')):
                if name in a[sc].get(mode, {}):
                    put('Abl' + tag + sc.capitalize() + mt,
                        a[sc][mode][name]['psnr'] - a[sc][mode]['full']['psnr'], '{:+.2f}')
                    put('Abl' + tag + sc.capitalize() + mt + 'Flicker',
                        100.0 * (a[sc][mode][name]['flicker'] / a[sc][mode]['full']['flicker'] - 1.0), '{:+.0f}')
    # Resolve cost shares (ACO estimates) and time per window frame at 1080p.
    r = CM['resolve']
    k = CM['calibration'] * CM['waves_per_simd_1080p'] / CM['clock_hz'] * 1000.0
    for mode, s_, n in MODES:
        tag = {'native': 'Nat', 'ultra': 'UQ', 'quality': 'Q', 'balanced': 'B', 'performance': 'P'}[mode]
        put('HistShare' + tag, 100.0 * r['still_invthr'] / (r['moving_base_invthr'] + (n - 1) * r['per_window_frame_invthr']), '{:.0f}')
    put('MsPerFrameHd', k * r['per_window_frame_invthr'], '{:.2f}')
    put('MsPerFrameUhd', 4 * k * r['per_window_frame_invthr'], '{:.2f}')
    # Ghost after uncovering.
    if g:
        for mode, mt in (('native', 'Nat'), ('quality', 'Q'), ('performance', 'P')):
            if mode in g:
                for meth, tg in (('hwtaa', 'Hw'), ('no_missing_fill', 'NoFill'),
                                 ('history_only', 'Hist'), ('taa', 'Taa')):
                    if meth in g[mode]:
                        put('Ghost' + tg + mt + 'One', g[mode][meth]['1'], '{:.1f}')
                if 'hwtaa' in g[mode] and 'no_missing_fill' in g[mode]:
                    put('GhostCut' + mt, 100.0 * (1.0 - g[mode]['hwtaa']['1'] / g[mode]['no_missing_fill']['1']), '{:.0f}')
                    late = [g[mode]['hwtaa'][str(x)] / g[mode]['history_only'][str(x)] for x in range(2, 9)]
                    put('GhostVsHist' + mt, 100.0 * (1.0 - sum(late) / len(late)), '{:.0f}')
    # Costs at 1080p.
    for mode, s, n in MODES:
        tag = {'native': 'Nat', 'ultra': 'UQ', 'quality': 'Q', 'balanced': 'B', 'performance': 'P'}[mode]
        for res, ww, hh in RESOLUTIONS:
            rt = {'1080p': 'Hd', '1440p': 'Qhd', '4K': 'Uhd'}[res]
            put('MsStill' + tag + rt, cost_ms(ww, hh, s, n, False)[0], '{:.1f}')
            put('MsMoving' + tag + rt, cost_ms(ww, hh, s, n, True)[0], '{:.1f}')
            put('Mem' + tag + rt, memory_mb(ww, hh, s, n), '{:.0f}')
    lines = ['% Generated by bench/make_tables.py: do not edit.']
    for k, v in sorted(m.items()):
        lines.append(f'\\newcommand{{\\n{k}}}{{{v}}}')
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'numbers.tex'), 'w').write('\n'.join(lines) + '\n')
    return m


if __name__ == '__main__':
    q, a, w, g = load('quality.json'), load('ablations.json'), load('window.json'), load('ghost.json')
    missing = [s for s in SCENES if s not in q]
    if missing:
        print('quality.json lacks', missing)
    for lang in L:
        if not missing:
            table_quality(q, lang)
            table_summary(q, lang)
        if a:
            table_ablation(a, lang)
        table_cost(lang)
    if not missing and a:
        for k, v in sorted(numbers(q, a, w, g).items()):
            print(f'{k:28s} {v}')
