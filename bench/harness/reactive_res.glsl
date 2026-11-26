/* SPDX-FileCopyrightText: 2026 UPBGE Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

layout(binding = 0) uniform sampler2D color_tx;
layout(binding = 0, rgba8) uniform image2D opaque_img;
layout(binding = 1, r8) writeonly uniform image2D reactive_img;
uniform ivec2 render_size;
uniform bool capture;
