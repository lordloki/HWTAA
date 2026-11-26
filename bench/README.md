# HWTAA evaluation bench

Everything in Section 5 of the paper comes from these scripts. They run the production GLSL
shaders of eevee-game (`shaders/`) through an OpenGL 4.3 context on analytic scenes
rendered on the CPU, and compare them with a 6 x 6 supersampled reference. Only the shaders'
engine-specific declarations are replaced (see `harness/`).

## Requirements

Python 3.10+, `numpy`, `moderngl` (EGL or any OpenGL 4.3 context; Mesa llvmpipe works),
`scikit-image`, `flip-evaluator`, `matplotlib`; LaTeX (`latexmk`, `pdflatex`) for the papers.

    pip install numpy moderngl scikit-image flip-evaluator matplotlib

The SSIM package is `scikit-image` (it is imported as `skimage`; the `skimage` package on PyPI is
an empty placeholder that fails to install).

**Context.** Linux uses a headless EGL context (Mesa llvmpipe works without a GPU); Windows and
other platforms use the platform default (WGL on Windows) with the installed GPU driver.
`HWTAA_GL_BACKEND` overrides it (`egl`, `x11`, `wgl`). macOS is not supported: its OpenGL stops at
4.1, without compute shaders. A hardware driver gives slightly different numbers than llvmpipe
(the paper's were produced with llvmpipe), and `run_all.sh` overwrites `results/`, the tables and
the papers with them.

**Windows (Git Bash).** `run_all.sh` uses `python` when `python3` is missing or is the Microsoft
Store stub; set `PYTHON` to choose another interpreter. `latexmk` needs Perl (MiKTeX installs it
on demand); without it, run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `en/` and `es/`.

## Layout

| Path | Content |
|---|---|
| `shaders/` | The four production shaders: store, resolve, reactive mask, shared library. |
| `harness/` | GL harness (`gpu_taau.py`), scenes (`persp_scene.py`, `taau_sim2.py`), TAA baseline (`taa_good.py`), GLSL prelude (`common.glsl`, `*_res.glsl`: type aliases, constants and resource bindings standing in for the engine's create infos; the harness strips each shader's `#include` and create-info lines and prepends them), textbook TAA for timing. The CPU side of the module (window matrices, motion chains, jitter, ring) is ported to Python in `gpu_taau.py`. |
| `hwtaa_bench.py` | `quality`, `ablations`, `window`, `ghost`, `frames`: write `results/*.json` and `results/frames.npz`. Ablations patch a copy of the resolve shader in `build/`. |
| `make_tables.py` | LaTeX tables in `../tables/{en,es}/` and `../tables/numbers.tex`, the macros for every number quoted in the text. |
| `make_figures.py` | `../figures/{en,es}/`: crops, window length, disocclusion. |
| `cost_model.json` | ACO compiler statistics of the shaders and the calibration that turns them into time on an RX 590 (GTX 1060 class), the traffic floor and the memory per texel. `make_tables.py` computes the cost table from it. |
| `gpu_timing.py` | Measures the store and resolve passes at 1080p, 1440p and 4K on a real GPU (`--res`, `--modes`): fills the ring with real frames, then repeats the last frame's dispatches back to back (full clocks, no texture uploads) and reports the median of 100 GL timer queries; checks that the repeated dispatch reproduces the frame. llvmpipe returns 0 for timer queries. |
| `gpu_profile.py` | Where the resolve's time goes on a real GPU: driver compiler statistics (NVIDIA, needs PyOpenGL; binary saved to `profile_*.bin`) and resolve time with the window length forced to 1, 2, 4, 8 and the mode's own. |
| `variants.py`, `gpu_variants.py` | Performance variants of the resolve (A uniform buffer for the window matrices, B prefetch of the next frame's ring samples, C half floats in shared memory) and their timing on a real GPU against the base shader, with an output check; also shorter windows (D). |
| `run_all.sh` | Everything, in order. |

## Notes

- Output 192 x 108, frames 30 to 59 measured, after the tone map c / (1 + c).
- TAA baseline: Halton(2,3) over 8 frames, closest-sample motion with exact motion vectors,
  Catmull-Rom history, variance clip (gamma 1) in tone-mapped YCoCg, blend 0.1; point
  (`taa_point`) or Gaussian (`taa_gauss`) reconstruction of the current frame.
- The upscaling modes are compared with TAA at native resolution.
- Times in the paper are static estimates (see `cost_model.json`), not measurements.
