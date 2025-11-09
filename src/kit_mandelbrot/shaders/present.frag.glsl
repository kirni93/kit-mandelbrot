#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
  float s = texture(tex, v_uv).r;      // 0..1 stability
  f_color = vec4(vec3(s), 1.0);        // grayscale for now
}

