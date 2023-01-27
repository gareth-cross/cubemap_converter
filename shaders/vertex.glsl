#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

// Projection matrix of the viewport.
uniform mat4 projection;

out vec2 TexCoords;

void main() {
  gl_Position = projection * vec4(pos.x, pos.y, pos.z, 1.0);
  TexCoords = uv;
}
