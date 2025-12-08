#version 330 core

// Same input as your present shader: UV of the fullscreen quad
in vec2 v_uv;

// We render into an RGBA32F texture, so we output a vec4 (4 components).
// Moderngl will attach this as a color attachment with 4 components.
out vec4 f_color;

// Viewport in the complex plane
uniform float re_min;
uniform float re_max;
uniform float imag_min;
uniform float imag_max;

// Mandelbrot controls
uniform int max_iter;

// Map UV (0..1,0..1) to complex plane coordinates
vec2 plane_coords(vec2 uv) {
    float cre = mix(re_min,  re_max,  uv.x);
    float cim = mix(imag_min, imag_max, uv.y);
    return vec2(cre, cim);
}

// Compute raw + smooth iteration; return a 0..1 "stability" value
float mandelbrot_stability(vec2 c, int maxIter) {
    vec2 z = vec2(0.0);
    int i = 0;

    for (; i < maxIter; ++i) {
        float x = z.x;
        float y = z.y;
        float x2 = x * x;
        float y2 = y * y;

        if (x2 + y2 > 4.0) {
            break;
        }

        z = vec2(x2 - y2 + c.x,
                 2.0 * x * y + c.y);
    }
    // Inside the set → fully stable
    if (i == maxIter) {
        return 1.0;
    }
    float base = float(i) / float(maxIter);

    return base;
}

void main() {
  vec2 c = plane_coords(v_uv);
  float s = mandelbrot_stability(c, max_iter);
  f_color = vec4(s, 0.0, 0.0, 1.0);
}
