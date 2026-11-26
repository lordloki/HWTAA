/* SPDX-FileCopyrightText: 2026 UPBGE Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

layout(binding = 0) uniform usampler2DArray ring_tx;
layout(binding = 1) uniform sampler2D depth_tx;
layout(binding = 2) uniform usampler2D object_id_tx;
layout(binding = 3) uniform sampler2D vector_tx;
layout(binding = 4) uniform sampler2D history_tx;
layout(binding = 5) uniform sampler2D reactive_tx;
layout(binding = 6) uniform sampler2D lock_tx;
layout(binding = 7) uniform sampler2D color_tx;
layout(std430, binding = 0) readonly buffer motion_buf_b { uint motion_buf[]; };
layout(std430, binding = 1) readonly buffer chain_buf_b { vec4 chain_buf[]; };
layout(std140, binding = 0) uniform window_buf_b { vec4 window_buf[TEMPORAL_WINDOW_MAX * TEMPORAL_AGE_STRIDE]; };
layout(binding = 0, rgba16f) writeonly uniform image2D history_out_img;
layout(binding = 1, rg8) writeonly uniform image2D lock_out_img;
uniform mat4 curr_viewprojinv;
uniform mat4 prev_reproject;
uniform vec4 curr_depth_params;
uniform vec2 render_scale;
uniform vec2 jitter_px;
uniform ivec2 render_size;
uniform ivec2 display_size;
uniform int tile_dim;
uniform int motion_len;
uniform int ring_layer;
uniform int window_frames;
uniform int chain_stride;
uniform float max_confidence;
uniform float lock_decay;
uniform bool history_valid;
uniform bool window_volatile;
uniform bool frequency_split;
#ifndef WINDOW_ORTHO
#define WINDOW_ORTHO false
#endif
const bool window_ortho = WINDOW_ORTHO;
