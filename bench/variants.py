# SPDX-FileCopyrightText: 2026 UPBGE Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Performance variants of the resolve shader, as source patches on the composed GLSL (harness
prelude + production shader). Each patch asserts its anchors, so a changed shader fails loudly.

    B  prefetch: the next frame's 2x2 ring samples are fetched while this frame's are processed.
    C  shared memory in half floats: tile colors, statistics and the fill color, 4624 -> 2512
       bytes per group.
    W  cooperative window: each ring texel of the group's footprint reprojected, tested and
       decoded once per frame instead of once per pixel that reads it (see below).
"""


def _patch(src, old, new):
    assert src.count(old) == 1, f'anchor not found once: {old[:60]!r}'
    return src.replace(old, new)


_FETCH = '''
/* Variant B: where the pixel's surface was `age` frames ago, and its 2x2 ring samples, fetched
 * one frame ahead of their use. False when the frame is out of the window or off screen. */
bool window_fetch(int age,
                  float4 world_h,
                  uint slot,
                  int chain_base,
                  bool is_deformed,
                  float2 render_half,
                  float2 ring_offset,
                  out float2 ring_lo,
                  out int2 t0,
                  out uint2 r0,
                  out uint2 r1,
                  out uint2 r2,
                  out uint2 r3)
{
  ring_lo = float2(0.0);
  t0 = int2(0);
  r0 = uint2(0u);
  r1 = uint2(0u);
  r2 = uint2(0u);
  r3 = uint2(0u);
  if (age >= window_frames) {
    return false;
  }
  const float4x4 viewproj_k = age_matrix(age, TEMPORAL_AGE_VIEWPROJ);
  const float4 clip_k = viewproj_k * ((slot != 0u) ?
                                          float4(chain_apply(chain_base + age * 3, world_h), 1.0) :
                                          world_h);
  if (clip_k.w <= 0.0 && !is_deformed) {
    return false;
  }
  const float2 ndc_k = !is_deformed ? clip_k.xy / clip_k.w :
                                      ((age < DEFORM_WINDOW_FRAMES) ?
                                           world_h.xy + world_h.zw * float(age) :
                                           float2(4.0));
  const float4 jitter_k = age_data(age, TEMPORAL_AGE_JITTER);
  ring_lo = ndc_k * render_half + (ring_offset + jitter_k.xy);
  t0 = int2(floor(ring_lo));
  if (any(lessThan(t0, int2(-1))) || any(greaterThanEqual(t0, render_size))) {
    return false;
  }
  const int2 lo = max(t0, int2(0));
  const int2 hi = min(t0 + 1, render_size - 1);
  const int layer = int(jitter_k.z);
  r0 = texelFetch(ring_tx, int3(lo.x, hi.y, layer), 0).xy;
  r1 = texelFetch(ring_tx, int3(hi.x, hi.y, layer), 0).xy;
  r2 = texelFetch(ring_tx, int3(hi.x, lo.y, layer), 0).xy;
  r3 = texelFetch(ring_tx, int3(lo.x, lo.y, layer), 0).xy;
  return true;
}

'''

_LOOP_HEAD = '''    for (int age = 1; age < window_frames; age++) {
      /* Where the pixel's surface was `age` frames ago. */'''
_LOOP_FETCH_END = '''      const uint2 ring_3 = texelFetch(ring_tx, int3(lo.x, lo.y, layer), 0).xy;
'''
_LOOP_PREFETCH = '''    float2 next_ring_lo;
    int2 next_t0;
    uint2 next_0, next_1, next_2, next_3;
    bool next_ok = window_fetch(1, world_h, slot, chain_base, is_deformed, render_half, ring_offset,
                                next_ring_lo, next_t0, next_0, next_1, next_2, next_3);
    for (int age = 1; age < window_frames; age++) {
      const bool frame_ok = next_ok;
      const float2 ring_lo = next_ring_lo;
      const int2 t0 = next_t0;
      const uint2 ring_0 = next_0;
      const uint2 ring_1 = next_1;
      const uint2 ring_2 = next_2;
      const uint2 ring_3 = next_3;
      next_ok = window_fetch(age + 1, world_h, slot, chain_base, is_deformed, render_half,
                             ring_offset, next_ring_lo, next_t0, next_0, next_1, next_2, next_3);
      if (!frame_ok) {
        continue;
      }
      const int2 lo = max(t0, int2(0));
      const int2 hi = min(t0 + 1, render_size - 1);
'''


def prefetch(src):
    i = src.index(_LOOP_HEAD)
    j = src.index(_LOOP_FETCH_END, i) + len(_LOOP_FETCH_END)
    assert src.count(_LOOP_HEAD) == 1 and j - i < 4000
    src = src[:i] + _LOOP_PREFETCH + src[j:]
    return _patch(src, '\nvoid main()\n', _FETCH + 'void main()\n')


_HALF_DECL = '''shared uint2 tile_work_h[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
shared uint2 tile_linear_h[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
shared uint tile_info[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
/* Per thread, from step 1 to the end of step 6: see `other` in step 1. */
shared uint2 tile_other_h[TEMPORAL_GROUP_SIZE * TEMPORAL_GROUP_SIZE];
uint2 half4_pack(float4 v)
{
  return uint2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
float4 half4_unpack(uint2 p)
{
  return float4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
}'''


def half_tiles(src):
    import re
    src = _patch(src, '''shared float4 tile_work[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
shared float3 tile_linear[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
shared uint tile_info[TEMPORAL_TILE_MAX * TEMPORAL_TILE_MAX];
/* Per thread, from step 1 to the end of step 6: see `other` in step 1. */
shared float4 tile_other[TEMPORAL_GROUP_SIZE * TEMPORAL_GROUP_SIZE];''', _HALF_DECL)
    src = _patch(src, '  tile_linear[index] = color;\n',
                 '  tile_linear_h[index] = half4_pack(float4(color, 0.0));\n')
    src = _patch(src, '''  tile_work[index] = float4(rgb_to_ycocg(color * (1.0 / (1.0 + max3(color)))),
                            ring_median_decode(unpackUnorm4x8(ring.y)));''',
                 '''  tile_work_h[index] = half4_pack(float4(rgb_to_ycocg(color * (1.0 / (1.0 + max3(color)))),
                                       ring_median_decode(unpackUnorm4x8(ring.y))));''')
    src = _patch(src, '  tile_other[gl_LocalInvocationIndex] = other;',
                 '  tile_other_h[gl_LocalInvocationIndex] = half4_pack(other);')
    src = _patch(src, 'const float4 other = tile_other[gl_LocalInvocationIndex];',
                 'const float4 other = half4_unpack(tile_other_h[gl_LocalInvocationIndex]);')
    # Reads: one component unpacks one word.
    n = len(re.findall(r'tile_(work|linear)\[', src))
    src, k1 = re.subn(r'tile_work\[([^\]]+)\]\.x\b', r'unpackHalf2x16(tile_work_h[\1].x).x', src)
    src, k2 = re.subn(r'tile_work\[([^\]]+)\]\.w\b', r'unpackHalf2x16(tile_work_h[\1].y).y', src)
    src, k3 = re.subn(r'tile_work\[([^\]]+)\]\.xyz\b', r'half4_unpack(tile_work_h[\1]).xyz', src)
    src, k4 = re.subn(r'tile_linear\[([^\]]+)\]', r'half4_unpack(tile_linear_h[\1]).rgb', src)
    assert k1 + k2 + k3 + k4 == n and not re.search(r'tile_(work|linear|other)\[', src), (n, k1, k2, k3, k4)
    return src


PATCHES = {'B': prefetch, 'C': half_tiles}
VARIANTS = ['base', 'B', 'C']


def apply(src, variant):
    for letter in ('' if variant == 'base' else variant):
        src = PATCHES[letter](src)
    return src


# ---- Variant W: cooperative window. ----
# Per frame of the window, the group's pixels that use it and do not follow their own moving
# object publish the bounding box of their 2x2 ring footprints (shared atomics); when it fits
# in TILE_STRIDE x TILE_STRIDE texels, the group reprojects, tests and decodes every texel of
# the box once, into tile_work and tile_linear (dead after step 5), and each pixel gathers its
# four records. Pixels of moving objects (own chain), and every pixel of a frame whose box does
# not fit, keep the per pixel path. Same samples, same tests, same weights as the base shader.

_W_SHARED = '''
/* Variant W: footprint box of the group per window frame, two sets alternated by frame (min x,
 * min y, max x, max y), and whether any pixel of the group uses the window. */
shared int win_box[8];
shared uint group_window;
/* Record flags, above the 16 key bits. */
#define WIN_VALID 0x10000u
#define WIN_SAME 0x20000u
#define WIN_FRONT 0x40000u
#define WIN_TRANSPARENT 0x80000u

/* One precomputed sample of the window (variant W). True when it counts. */
bool window_record(float4 record,
                   float3 linear_color,
                   float2 g_now,
                   uint key_closest,
                   bool thin_front,
                   inout float3 window_sum,
                   inout float window_weight)
{
  const uint flags = floatBitsToUint(record.z);
  if ((flags & WIN_VALID) == 0u) {
    return false;
  }
  const float2 g = record.xy - g_now;
  const float2 g_squared = g * g;
  if (max(g_squared.x, g_squared.y) >= WINDOW_UNIT_SQUARED) {
    return false;
  }
  const bool same = (flags & WIN_SAME) != 0u;
  const bool in_front = thin_front && (flags & TEMPORAL_KEY_MASK) == key_closest &&
                        (flags & WIN_FRONT) != 0u;
  if (!(same || in_front)) {
    return false;
  }
  const float weight = exp2(-(g_squared.x + g_squared.y));
  window_sum += linear_color * weight;
  window_weight += weight;
  return true;
}
'''

_W_LOOP_TOP = '''    /* Tile position of a sample at g (window units, this frame): pixel independent. */
    const float2 tile_shift = jitter_px - float2(tile_origin);
    float missing_weight = 0.0;
    for (int age = 1; age < window_frames; age++) {
      const int set = (age & 1) * 4;
      /* Where the pixel's surface was `age` frames ago. */
      bool frame_ok = false;
      float2 ring_lo = float2(0.0);
      int2 t0 = int2(0);
      if (use_window) {
        const float4x4 viewproj_k = age_matrix(age, TEMPORAL_AGE_VIEWPROJ);
        const float4 clip_k = viewproj_k *
                              ((slot != 0u) ?
                                   float4(chain_apply(chain_base + age * 3, world_h), 1.0) :
                                   world_h);
        const float2 ndc_k = !is_deformed ? clip_k.xy / clip_k.w :
                                            ((age < DEFORM_WINDOW_FRAMES) ?
                                                 world_h.xy + world_h.zw * float(age) :
                                                 float2(4.0));
        ring_lo = ndc_k * render_half + (ring_offset + age_data(age, TEMPORAL_AGE_JITTER).xy);
        t0 = int2(floor(ring_lo));
        frame_ok = (clip_k.w > 0.0 || is_deformed) && all(greaterThanEqual(t0, int2(-1))) &&
                   all(lessThan(t0, render_size));
      }
      const int2 lo = max(t0, int2(0));
      const int2 hi = min(t0 + 1, render_size - 1);
      const bool coop_pixel = frame_ok && !own_chain;
      if (coop_pixel) {
        atomicMin(win_box[set], lo.x);
        atomicMin(win_box[set + 1], lo.y);
        atomicMax(win_box[set + 2], hi.x);
        atomicMax(win_box[set + 3], hi.y);
      }
      if (gl_LocalInvocationIndex == 0u) {
        /* The other set, read before the last barrier of the previous frame. */
        win_box[4 - set] = 1 << 30;
        win_box[5 - set] = 1 << 30;
        win_box[6 - set] = -(1 << 30);
        win_box[7 - set] = -(1 << 30);
      }
      barrier();
      const int2 box_lo = int2(win_box[set], win_box[set + 1]);
      const int2 box_size = int2(win_box[set + 2], win_box[set + 3]) - box_lo + 1;
      const bool coop = box_size.x > 0 && all(lessThanEqual(box_size, int2(TILE_STRIDE)));
      const int layer = int(age_data(age, TEMPORAL_AGE_JITTER).z);
      if (coop) {
        const float4x4 window_k = age_matrix(age, TEMPORAL_AGE_REPROJECT);
        const float far_distance = age_data(age, TEMPORAL_AGE_DEPTH_PARAMS).w;
        const int box_len = box_size.x * box_size.y;
        const float box_width_inv = 1.0 / float(box_size.x);
        for (int i = int(gl_LocalInvocationIndex); i < box_len;
             i += TEMPORAL_GROUP_SIZE * TEMPORAL_GROUP_SIZE)
        {
          const int row = int(floor((float(i) + 0.5) * box_width_inv));
          const int2 local = int2(i - row * box_size.x, row);
          const int2 texel = box_lo + local;
          const uint2 ring = texelFetch(ring_tx, int3(texel, layer), 0).xy;
          const float inverse = info_inverse_depth(ring.x);
          float z = inverse;
          if (window_ortho) {
            z = min(1.0 / max(z, 1e-30), far_distance);
          }
          const float4 clip = window_k[3] + window_k[0] * float(texel.x) +
                              window_k[1] * float(texel.y) + window_k[2] * z;
          uint flags = info_key(ring.x);
          if ((ring.x & TEMPORAL_INFO_TRANSPARENT_BIT) != 0u) {
            flags |= WIN_TRANSPARENT;
          }
          float2 g_abs = float2(0.0);
          if (clip.w > 0.0) {
            g_abs = clip.xy / clip.w;
            flags |= WIN_VALID;
            const float2 tile_pos = g_abs * tile_per_g + tile_shift;
            if (all(greaterThanEqual(tile_pos, float2(0.0))) &&
                all(lessThan(tile_pos, float2(tile_dim))))
            {
              const uint info_now =
                  tile_info[uint(trunc(tile_pos.y) * float(TILE_STRIDE) + tile_pos.x)];
              if (info_key(info_now) == info_key(ring.x) && !info_is_unstable(info_now)) {
                flags |= WIN_SAME;
              }
              if (!info_is_unstable(ring.x) &&
                  inverse > info_inverse_depth(info_now) * (1.0 + DEPTH_TOLERANCE))
              {
                flags |= WIN_FRONT;
              }
            }
          }
          const float3 squares = ring_color_squares(ring.y);
          const int index = local.y * TILE_STRIDE + local.x;
          tile_work[index] = float4(g_abs, uintBitsToFloat(flags), uintBitsToFloat(ring.y));
          tile_linear[index] = squares * ring_color_inverse_scale(squares);
        }
      }
      barrier();
      if (!frame_ok) {
        continue;
      }
      if (coop_pixel && coop) {
        const int2 step_i = hi - lo;
        const int i_3 = (lo.y - box_lo.y) * TILE_STRIDE + (lo.x - box_lo.x);
        const int i_2 = i_3 + step_i.x;
        const int i_0 = i_3 + step_i.y * TILE_STRIDE;
        const int i_1 = i_0 + step_i.x;
        const float4 record_0 = tile_work[i_0];
        const float4 record_1 = tile_work[i_1];
        const float4 record_2 = tile_work[i_2];
        const float4 record_3 = tile_work[i_3];
        const uint flags_0 = floatBitsToUint(record_0.z);
        const uint flags_1 = floatBitsToUint(record_1.z);
        const uint flags_2 = floatBitsToUint(record_2.z);
        const uint flags_3 = floatBitsToUint(record_3.z);
        const bool transparent = ((flags_0 | flags_1 | flags_2 | flags_3) & WIN_TRANSPARENT) != 0u;
        if (is_volatile || reactive_now || transparent) {
          const bool2 nearest_hi = greaterThan(int2(floor(ring_lo + 0.5)), t0);
          const float nearest = nearest_hi.y ? (nearest_hi.x ? record_1.w : record_0.w) :
                                               (nearest_hi.x ? record_2.w : record_3.w);
          if (!luma_within(ring_median_decode(unpackUnorm4x8(floatBitsToUint(nearest))) + 0.01,
                           neighborhood_median + 0.01,
                           MEDIAN_KILL_RATIO))
          {
            continue;
          }
        }
        const float2 lo_f = float2(lo);
        const float2 own_00 = (lo_f - ring_lo) * g_texel;
        const float2 own_step = g_texel * float2(step_i);
        const float2 own_01 = own_00 + float2(0.0, own_step.y);
        const float2 own_10 = own_00 + float2(own_step.x, 0.0);
        const float2 own_11 = own_00 + own_step;
        const bool counted_0 = window_record(
            record_0, tile_linear[i_0], g_now, key_closest, thin_front, window_sum, window_weight);
        const bool counted_1 = window_record(
            record_1, tile_linear[i_1], g_now, key_closest, thin_front, window_sum, window_weight);
        const bool counted_2 = window_record(
            record_2, tile_linear[i_2], g_now, key_closest, thin_front, window_sum, window_weight);
        const bool counted_3 = window_record(
            record_3, tile_linear[i_3], g_now, key_closest, thin_front, window_sum, window_weight);
        const uint key_0 = flags_0 & TEMPORAL_KEY_MASK;
        const uint key_1 = flags_1 & TEMPORAL_KEY_MASK;
        const uint key_2 = flags_2 & TEMPORAL_KEY_MASK;
        const uint key_3 = flags_3 & TEMPORAL_KEY_MASK;
        const bool other_seen = (counted_0 && key_0 != key_closest) ||
                                (counted_1 && key_1 != key_closest) ||
                                (counted_2 && key_2 != key_closest) ||
                                (counted_3 && key_3 != key_closest);
        if ((own_chain || geometric_edge) && !other_seen) {
          missing_weight += missing_sample(key_0, own_01, key_closest, counted_0) +
                            missing_sample(key_1, own_11, key_closest, counted_1) +
                            missing_sample(key_2, own_10, key_closest, counted_2) +
                            missing_sample(key_3, own_00, key_closest, counted_3);
        }
        continue;
      }
      /* Per pixel path: own moving object, or a box that does not fit. */
      const uint2 ring_0 = texelFetch(ring_tx, int3(lo.x, hi.y, layer), 0).xy;
      const uint2 ring_1 = texelFetch(ring_tx, int3(hi.x, hi.y, layer), 0).xy;
      const uint2 ring_2 = texelFetch(ring_tx, int3(hi.x, lo.y, layer), 0).xy;
      const uint2 ring_3 = texelFetch(ring_tx, int3(lo.x, lo.y, layer), 0).xy;
'''


def coop_window(src):
    # No early return: barriers follow. Threads past the display edge compute their edge pixel
    # and store nothing.
    src = _patch(src, '  const int2 out_px = int2(gl_GlobalInvocationID.xy);\n',
                 '  const bool in_display = all(lessThan(int2(gl_GlobalInvocationID.xy), display_size));\n'
                 '  const int2 out_px = min(int2(gl_GlobalInvocationID.xy), display_size - 1);\n')
    src = _patch(src, '''  if (any(greaterThanEqual(out_px, display_size))) {
    return;
  }
''', '')
    src = _patch(src, '''  if (i_0 < tile_len) {
    tile_store(local_0, color_0, ring_0);''', '''  if (gl_LocalInvocationIndex == 0u) {
    group_window = 0u;
    win_box[0] = 1 << 30;
    win_box[1] = 1 << 30;
    win_box[2] = -(1 << 30);
    win_box[3] = -(1 << 30);
    win_box[4] = 1 << 30;
    win_box[5] = 1 << 30;
    win_box[6] = -(1 << 30);
    win_box[7] = -(1 << 30);
  }
  if (i_0 < tile_len) {
    tile_store(local_0, color_0, ring_0);''')
    src = _patch(src, '    imageStore(lock_out_img, out_px, float4(lock_life, median_encode(lock_luma), 0.0, 0.0));',
                 '    if (in_display) {\n      imageStore(lock_out_img, out_px, float4(lock_life, median_encode(lock_luma), 0.0, 0.0));\n    }')
    src = _patch(src, '  imageStore(history_out_img, out_px, float4(result, confidence_new));',
                 '  if (in_display) {\n    imageStore(history_out_img, out_px, float4(result, confidence_new));\n  }')
    src = _patch(src, '  const bool use_history = window_mix < 1.0;\n',
                 '  const bool use_history = window_mix < 1.0;\n  if (window_mix > 0.0) {\n    atomicOr(group_window, 1u);\n  }\n')
    src = _patch(src, '\nvoid main()\n', _W_SHARED + '\nvoid main()\n')
    # Step 6.
    a0 = '  /* ---- 6. Sample window. ---- */\n  if (window_mix > 0.0) {\n'
    a1 = '    float missing_weight = 0.0;\n    for (int age = 1; age < window_frames; age++) {\n'
    a2 = '      /* Moving shadows, and transparent surfaces over the pixel now or over the samples then:\n'
    a3 = '    /* The window reads each past frame where the pixel\'s closest surface was.'
    a4 = '  if (in_display) {\n    imageStore(history_out_img'
    i0, i1, i2, i3, i4 = (src.index(a) for a in (a0, a1, a2, a3, a4))
    assert i0 < i1 < i2 < i3 < i4 and all(src.count(a) == 1 for a in (a0, a1, a2, a3))
    head = src[i0 + len(a0):i1]
    body = src[i2:i3]          # the per pixel frame body, through the end of the frame loop
    tail = src[i3:i4]          # missing weight and output, through the end of the window block
    assert tail.rstrip().endswith('}\n  }') or tail.rstrip().endswith('}')
    tail_inner = tail.rstrip()[:-1].rstrip() + '\n'   # without the window block's brace
    step6 = ('  /* ---- 6. Sample window (variant W: cooperative). ---- */\n'
             '  barrier();\n'
             '  if (group_window != 0u) {\n'
             '    const bool use_window = window_mix > 0.0;\n' + head + _W_LOOP_TOP + body +
             '    if (use_window) {\n' + tail_inner + '    }\n  }\n\n')
    return src[:i0] + step6 + src[i4:]


PATCHES['W'] = coop_window
VARIANTS.append('W')
