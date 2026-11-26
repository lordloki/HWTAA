/* SPDX-FileCopyrightText: 2026 UPBGE Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define uint4 uvec4
#define uint2 uvec2
#define bool2 bvec2
#define float4x4 mat4
#define TEMPORAL_GROUP_SIZE 8
#define TEMPORAL_TILE_MAX 10
#define TEMPORAL_RECONSTRUCTION_SHARPNESS 4.0
#define TEMPORAL_KEY_MASK 0xFFFFu
#define TEMPORAL_SLOT_SHIFT 16
#define TEMPORAL_SLOT_MASK 0x1FFFu
#define TEMPORAL_MOVED_BIT 0x20000000u
#define TEMPORAL_VOLATILE_BIT 0x40000000u
#define TEMPORAL_DEFORM_BIT 0x80000000u
#define TEMPORAL_INFO_UNSTABLE_BIT 0x80000000u
#define TEMPORAL_INFO_TRANSPARENT_BIT 0x00010000u
#define TEMPORAL_REACTIVE_THRESHOLD 0.1
#define TEMPORAL_REACTIVE_SCALE 4.0
#define TEMPORAL_WINDOW_MAX 16
#define TEMPORAL_AGE_STRIDE 10
#define TEMPORAL_AGE_REPROJECT 0
#define TEMPORAL_AGE_VIEWPROJ 4
#define TEMPORAL_AGE_DEPTH_PARAMS 8
#define TEMPORAL_AGE_JITTER 9
layout(local_size_x = TEMPORAL_GROUP_SIZE, local_size_y = TEMPORAL_GROUP_SIZE) in;
