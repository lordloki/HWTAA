/* SPDX-FileCopyrightText: 2026 UPBGE Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

layout(binding = 0) uniform sampler2D color_tx;
layout(binding = 1) uniform sampler2D depth_tx;
layout(binding = 2) uniform usampler2D object_id_tx;
layout(std430, binding = 0) readonly buffer motion_buf_b { uint motion_buf[]; };
layout(binding = 0, rg32ui) writeonly uniform uimage2DArray ring_img;
uniform vec4 depth_params;
uniform ivec2 render_size;
uniform int motion_len;
uniform int ring_layer;
layout(binding = 3) uniform sampler2D reactive_tx;
