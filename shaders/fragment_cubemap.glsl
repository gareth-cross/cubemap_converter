#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

// Fragment coordinate in pixels.
// Note that we swap the origin to the top left, since the valid mask is read in flipped vertically.
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

// Rotation matrix from typical camera to DirectX Cubemap (what we exported).
uniform mat3 cubemap_R_camera;

// The remap table.
uniform sampler2D remap_table;

// The cubemap.
uniform samplerCube input_cube;

// The valid mask (corresponds to the remap table).
uniform sampler2D valid_mask;

// True if we are processing the inverse-depth map.
uniform bool is_depth;

void main() {
    // Lookup the unit vector:
    vec3 v_cam = normalize(texture(remap_table, TexCoords).xyz);
    vec3 v_cube = cubemap_R_camera * v_cam;

    // Sample the cube-map:
    vec3 color = texture(input_cube, v_cube).rgb;

    // Read from the valid mask:
    float is_valid = float(texelFetch(valid_mask, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0).r > 0.0);

    if (!is_depth) {
        FragColor = vec4(color * is_valid, 1.0f);
    } else {
        // This is not real color, but inverse depth. The texture unit has normalized from [0, 65535] --> [0, 1].
        float inv_depth_normalized = color.x;

        // TODO: Make this a parameter. This is the near clipping plane in unreal engine.
        const float ue_near_clip_plane_meters = 0.1f;

        // Scale inverse depth into units of meters.
        float inv_depth_meters = inv_depth_normalized / ue_near_clip_plane_meters;

        // Inverse depth has been specified wrt the image plane of the cube face.
        // We convert to inverse range in the target camera model.
        // The largest element of `v_cube` determines what face of the cubemap we read from. For example,
        // [0.1, 0.2, 0.6] reads from +z, while [-0.7, 0.1, 0.1] would read from -x.

        // To convert from inverse depth to inverse range we do:
        // v_cube * range = p_face * depth ---> v_unit / inv_range = p_face / inv_depth
        // Where `p_face` is a point in the face of the cube we sampled from. Notably, one of the elements of p_face will
        // always be +1 or -1 (the element corresponding to the plane of the face).
        //
        // Then:
        // max(v_cube.xyz) / inv_range = ± 1 / inv_depth
        // max(v_cube.xyz) * inv_depth = ± inv_range
        // Where we have to be careful to take the absolute value of the max of (x,y,z) in order to keep range positive.
        float max_axis = max(max(abs(v_cube.x), abs(v_cube.y)), abs(v_cube.z));
        float inv_range_meters = inv_depth_meters * max_axis;

        // Normalize it back into the range of [0 (infinity), 1 / ue_clip_plane].
        float inv_range_normalized = min(inv_range_meters * ue_near_clip_plane_meters, 1.0f);
        FragColor = vec4(inv_range_normalized * is_valid, 0.0f, 0.0f, 1.0f);
    }
}
