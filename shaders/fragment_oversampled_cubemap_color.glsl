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

void main() {
    // Lookup the unit vector:
    vec3 v_cam = normalize(texture(remap_table, TexCoords).xyz);
    vec3 v_cube = cubemap_R_camera * v_cam;

    // Read from the valid mask:
    float is_valid = float(texelFetch(valid_mask, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).r > 0.0);

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
        vec2 uv = (p_face + oversampled_half_size) / (2.0f * oversampled_half_size);

        // We flip y, since images are read in normal (top to bottom) vertical order, instead of OpenGL convention.
        uv = vec2(uv.x, 1.0 - uv.y);

        // Compute blend weights in X & Y:
        vec2 blend_weights = 1.0 - smoothstep(vec2(1.0f, 1.0f), vec2(oversampled_half_size, oversampled_half_size),  p_face);
        float weight_product = blend_weights.x * blend_weights.y;

        // Add contribution:
        color_rgb += texture(input_cube, vec3(uv, float(face))).xyz * weight_product;
        total_weight += weight_product;
    }
    FragColor = vec4(color_rgb * is_valid / total_weight, 1.0f);
}
