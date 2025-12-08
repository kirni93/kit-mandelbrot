#version 330

uniform sampler2D tex;
uniform int max_iter;
uniform int use_smooth;
in vec2 v_uv;
out vec4 f_color;

// color constants
//const vec3 COLOR_MIDNIGHTBLUE = vec3(0.098, 0.098, 0.439);
const vec3 COLOR_MIDNIGHTBLUE = vec3(0.010, 0.010, 0.20);
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

float stability_from_data(vec4 raw, int maxIter, bool do_smooth) {
  float s_basic = raw.r;          // i / max_iter from compute pass
  float iter_f  = raw.g;
  vec2  z       = vec2(raw.b, raw.a);

  if (iter_f >= float(maxIter)) {
      return s_basic;
  }

  if(!do_smooth){
    return s_basic;
  }

  float mag = length(z);

  if (mag > 0.0) {
      float smooth_iter = iter_f + 1.0 - log(log(mag)) / log(2.0);
      float s_smooth = smooth_iter / float(maxIter);
      return clamp(s_smooth, 0.0, 1.0);
  } else {
      return s_basic;
  }
}

void main() {
  vec4 raw = texture(tex, v_uv);

  bool do_smooth = (use_smooth != 0);

  float s = stability_from_data(raw, max_iter, do_smooth); 
  vec3 color = colormap(s);

  f_color = vec4(color, 1.0);
}
