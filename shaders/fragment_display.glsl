#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

// Texture we are going to display.
uniform sampler2D image;

// Viewport dims in pixels.
uniform vec2 viewport_dims;

// Image dims in pixels.
uniform vec2 image_dims;

void main() {
  // determine scale factor to fit the image in the viewport
  float scale_factor = min(viewport_dims.x / image_dims.x, viewport_dims.y / image_dims.y);

  // scale image dimensions to fit
  vec2 scaled_image_dims = image_dims * scale_factor;

  // compute offset to center:
  vec2 image_origin = (viewport_dims - scaled_image_dims) / vec2(2.0, 2.0);

  // compute coords inside the viewport bounding box [0 -> viewport_dims]
  vec2 viewport_coords = TexCoords * viewport_dims;

  // transform viewport coordinates into image coords (normalized)
  vec2 image_coords = (viewport_coords - image_origin) / scaled_image_dims;

  // are they inside the box?
  bvec2 inside_upper_bound = lessThan(image_coords, vec2(1.0, 1.0));
  bvec2 inside_lower_bound = greaterThan(image_coords, vec2(0.0, 0.0));
  float mask = float(inside_upper_bound.x && inside_upper_bound.y && inside_lower_bound.x && inside_lower_bound.y);

  vec3 rgb = texture(image, image_coords).xyz;
  FragColor = vec4(rgb.x * mask, rgb.y * mask, rgb.z * mask, 1.0f);
}
