#version 330 core

// Same input as your present shader: UV of the fullscreen quad
in vec2 v_uv;

// We render into an R32F texture, so a single float output is fine.
// Moderngl will attach this as a color attachment with 1 component.
out vec4 f_color;

// Viewport in the complex plane
uniform float re_min;
uniform float re_max;
uniform float imag_min;
uniform float imag_max;

// Mandelbrot controls
uniform int max_iter;
uniform int smooth_stability; // 0 = off, 1 = on

// Map UV (0..1,0..1) to complex plane coordinates
vec2 plane_coords(vec2 uv) {
    float cre = mix(re_min,  re_max,  uv.x);
    float cim = mix(imag_min, imag_max, uv.y);
    return vec2(cre, cim);
}

// Compute raw + smooth iteration; return a 0..1 "stability" value
float mandelbrot_stability(vec2 c, int maxIter, bool do_smooth) {
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

    // Basic normalized escape: 0..1
    float base = float(i) / float(maxIter);

    if (!do_smooth) {
        return base;
    }

    // Smooth escape-time coloring
    float mag = length(z);
    if (mag > 0.0) {
        float smooth_iter = float(i) + 1.0 - log(log(mag)) / log(2.0);
        return clamp(smooth_iter / float(maxIter), 0.0, 1.0);
    } else {
        return base;
    }
}

void main() {
  vec2 c = plane_coords(v_uv);
  bool do_smooth = (smooth_stability != 0);
  float s = mandelbrot_stability(c, max_iter, do_smooth);
  f_color = vec4(s, 0.0, 0.0, 1.0);
}
