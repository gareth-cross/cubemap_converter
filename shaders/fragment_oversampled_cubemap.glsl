#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

// Fragment coordinate in pixels.
// Note that we swap the origin to the top left, since the valid mask is read in flipped vertically.
layout (origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

// Rotation matrix from typical camera to DirectX Cubemap (what we exported).
uniform mat3 cubemap_R_camera;

// The remap table.
uniform sampler2D remap_table;

// The oversampled cubemap represented as a texture array.
uniform sampler2DArray input_cube;

// The valid mask (corresponds to the remap table).
uniform sampler2D valid_mask;

// The oversampled cube FOV in radians.
uniform float oversampled_fov;

// Size of a single cubemap face in pixels.
uniform int cubemap_dim;

// True if we are processing depth - false if we are processing color.
uniform bool is_depth;

// The clip plane in Unreal Engine in meters.
uniform float ue_clip_plane_meters;

// Transform vector `v` from cube coordinates to face coordinates.
vec3 TransformToFaceFromCube(in int face, in vec3 v) {
    switch (face) {
        case 0:     // Positive X
            return vec3(-v.z, v.y, v.x);
        case 1:     // Negative X
            return vec3(v.z, v.y, -v.x);
        case 2:     // Positive Y
            return vec3(v.x, -v.z, v.y);
        case 3:     // Negative Y
            return vec3(v.x, v.z, -v.y);
        case 4:     // Positive Z
            return vec3(v.x, v.y, v.z);
        case 5:     // Negative Z
            return vec3(-v.x, v.y, -v.z);
    }
    return vec3(0.0, 0.0, 0.0);
}

// Given normalized inverse depth, compute normalized invers range.
// The argument is in [0, 1] (converted from [0, 65535] for us by the texture unit).
float InverseRangeFromInverseDepth(in float inv_depth_normalized, in vec3 v_face) {
    // Scale inverse depth into units of meters.
    float inv_depth_meters = inv_depth_normalized / ue_clip_plane_meters;

    // Inverse depth has been specified wrt the image plane of the cube face.
    // We convert to inverse range in the target camera model.
    //
    // To convert from inverse depth to inverse range we do:
    //   v_face * range = p_face * depth ---> v_unit / inv_range = p_face / inv_depth
    // Where `p_face` is a point in the face of the cube we sampled from. The z-value of
    // `p_face` is 1 (specified in face-local coordinates), so we can write:
    //
    // Then:
    // v_face.z / inv_range = 1 / inv_depth
    // v_face.z * inv_depth = inv_range
    // Where we have to be careful to take the absolute value of the max of (x,y,z) in order to keep range positive.
    float inv_range_meters = inv_depth_meters * v_face.z;

    // Normalize it back into the range of [0 (infinity), 1 / ue_clip_plane].
    // TODO: We could choose a new clip value here if we wanted to.
    float inv_range_normalized = min(inv_range_meters * ue_clip_plane_meters, 1.0f);
    return inv_range_normalized;
}

// TODO: This program might be a bit faster if split it into two shaders for RGB and range.
void main() {
    // Lookup the unit vector:
    vec3 v_cam = normalize(texture(remap_table, TexCoords).xyz);
    vec3 v_cube = cubemap_R_camera * v_cam;

    // Read from the valid mask:
    float is_valid = float(texelFetch(valid_mask, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).x > 0.0);

    // Oversampled image-plane width (normalized units, halved):
    float oversampled_half_size = tan(oversampled_fov * 0.5f);
    float overlap_blend_distance = oversampled_half_size - 1.0f;

    // Intersect ray into the face:
    float total_weight = 0.0f;
    vec3 color_rgb = vec3(0.0f, 0.0f, 0.0f);
    for (int face = 0; face < 6; ++face) {
        vec3 v_face = TransformToFaceFromCube(face, v_cube);

        // Check if we can project into the cube face:
        if (v_face.z <= 0.0f) {
            continue;
        }
        vec2 p_face = v_face.xy / v_face.z;

        // Convert to normalized units if we can:
        if (abs(p_face.x) > oversampled_half_size || abs(p_face.y) > oversampled_half_size) {
            continue;
        }
        vec2 uv = clamp((p_face + oversampled_half_size) / (2.0f * oversampled_half_size), 0.0f, 1.0f);

        // We flip y, since images are read in normal (top to bottom) vertical order, instead of OpenGL convention.
        uv = vec2(uv.x, 1.0 - uv.y);

        // Check if we are processing color or depth.
        if (!is_depth) {
            // Sample w/ bilinear interpolation. `face` is passed as a whole integer, cast to float.
            vec3 sampled_rgb = texture(input_cube, vec3(uv, float(face))).xyz;

            // Compute blend weights in X & Y:
            vec2 blend_weights = 1.0 - smoothstep(vec2(1.0f, 1.0f), vec2(oversampled_half_size, oversampled_half_size),  p_face);
            float weight_product = blend_weights.x * blend_weights.y;
            // Add contribution:
            color_rgb += sampled_rgb * weight_product;
            total_weight += weight_product;
        } else {
            // Sample the four values we would normally use for interpolation, and take the maximum:
            int max_pixel_value = cubemap_dim - 1;
            ivec2 p00 = ivec2(floor(uv * max_pixel_value));
            ivec2 p11 = ivec2(ceil(uv * max_pixel_value));
            float v00 = texelFetch(input_cube, ivec3(p00.x, p00.y, face), 0).x;
            float v10 = texelFetch(input_cube, ivec3(p11.x, p00.y, face), 0).x;
            float v01 = texelFetch(input_cube, ivec3(p00.x, p11.y, face), 0).x;
            float v11 = texelFetch(input_cube, ivec3(p11.x, p11.y, face), 0).x;
            float v_max = max(max(v00, v01), max(v10, v11));

            // Take the maximum inverse range (ie. closest object).
            float inv_range = InverseRangeFromInverseDepth(v_max, v_face);
            color_rgb.x = max(color_rgb.x, inv_range);
        }
    }
    if (!is_depth) {
        FragColor = vec4(color_rgb * is_valid / total_weight, 1.0f);
    } else {
        FragColor = vec4(color_rgb * is_valid, 1.0f);
    }
}
