#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;

// color constants
//const vec3 COLOR_MIDNIGHTBLUE = vec3(0.098, 0.098, 0.439);
const vec3 COLOR_MIDNIGHTBLUE = vec3(0.008, 0.008, 0.12);
const vec3 COLOR_WHITE = vec3(1.0, 1.0, 1.0);
const vec3 COLOR_YELLOW = vec3(1.0, 1.0, 0.0);
const vec3 COLOR_RED = vec3(1.0, 0.0, 0.0);
const vec3 COLOR_BLACK = vec3(0.0, 0.0, 0.0);

vec3 colormap(float stability) {
  if(stability < 0.5){
    return mix(COLOR_MIDNIGHTBLUE, COLOR_WHITE, stability / 0.5);
  }
  else if(stability < 0.65){
    return mix(COLOR_WHITE, COLOR_YELLOW, (stability - 0.5) / 0.15);
  }  
  else if(stability < 0.8){
    return mix(COLOR_YELLOW, COLOR_RED, (stability - 0.65) / 0.15);
  }
  else {
    return mix(COLOR_RED, COLOR_BLACK, (stability - 0.8) / 0.2);
  }
}

void main() {
  float s = texture(tex, v_uv).r;      // 0..1 stability
  vec3 color = colormap(s);
  f_color = vec4(color, 1.0);
}

