// Catmull-Rom 2D reprojection — exact UE4 Bicubic2DCatmullRom math.
//
// slangc compiles this implementation to CUDA with automatic backward
// (bwd_diff) for gradient validation.
//
// The UE4 algorithm (Engine/Shaders/PostProcessTemporalAA.usf,
// Bicubic2DCatmullRom):
//   1. Convert UV (pixel coords) to texel center + fractional offset f.
//   2. Compute 4 Catmull-Rom weights (Keys cubic, A = -0.5):
//        w0 = f² − 0.5(f³ + f)
//        w1 = 1.5f³ − 2.5f² + 1
//        w3 = 0.5(f³ − f²)
//        w2 = 1 − w0 − w1 − w3
//   3. UE collapses the 4×4 = 16 point samples into 3×3 = 9 bilinear taps for
//      performance. We expand back to the full 16-tap separable form using the
//      SAME weight formulas; the result is identical to the 9-tap bilinear
//      trick (bilinear filtering is exactly a weighted sum of the same texels).
//
// The autodiff backward of this function is the exact analytical gradient of
// the Catmull-Rom reprojection w.r.t. both the motion vector and the taps.

// UE4 Catmull-Rom weights (Keys cubic, A = -0.5).
[Differentiable]
void catmullRomWeights(float f, out float w0, out float w1, out float w2, out float w3) {
    float f2 = f * f;
    float f3 = f2 * f;
    w0 = f2 - 0.5f * (f3 + f);
    w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
    w3 = 0.5f * (f3 - f2);
    w2 = 1.0f - w0 - w1 - w3;
}

// 1D 4-tap Catmull-Rom interpolation.
[Differentiable]
float cr1d(float f, float h0, float h1, float h2, float h3) {
    float w0, w1, w2, w3;
    catmullRomWeights(f, w0, w1, w2, w3);
    return w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3;
}

// Full per-pixel reprojection: given target pixel (x, y) and motion (mvx, mvy),
// interpolate the already-gathered-and-clamped 4×4 neighborhood `taps`.
// taps layout: row-major, taps[j*4 + i] is column i (x) of row j (y).
// Differentiable w.r.t. mvx, mvy, and taps; x, y are integer (no_diff).
[Differentiable]
float catmullRomReproject(
    float mvx, float mvy,
    no_diff int x, no_diff int y,
    float taps[16]
) {
    float px = float(x) + mvx;
    float py = float(y) + mvy;
    int ix = int(floor(px));
    int iy = int(floor(py));
    float tx = px - float(ix);
    float ty = py - float(iy);

    float row0 = cr1d(tx, taps[0], taps[1], taps[2], taps[3]);
    float row1 = cr1d(tx, taps[4], taps[5], taps[6], taps[7]);
    float row2 = cr1d(tx, taps[8], taps[9], taps[10], taps[11]);
    float row3 = cr1d(tx, taps[12], taps[13], taps[14], taps[15]);
    return cr1d(ty, row0, row1, row2, row3);
}

// Per-invocation result: forward value + gradients w.r.t. mv and taps.
struct CatmullRomGrads {
    float forward_val;
    float grad_mvx;
    float grad_mvy;
    float grad_taps[16];
};

RWStructuredBuffer<CatmullRomGrads> results;
// Packed inputs, 20 floats per invocation:
//   [0]=mvx [1]=mvy [2..17]=taps[16] [18]=x [19]=y
StructuredBuffer<float> inputs;
uniform int count;

[shader("compute")]
[numthreads(64, 1, 1)]
void crReference(uint3 tid: SV_DispatchThreadID) {
    if (int(tid.x) >= count) {
        return;
    }
    int base = int(tid.x) * 20;
    float mvx_val = inputs[base + 0];
    float mvy_val = inputs[base + 1];
    float tap_vals[16];
    [ForceUnroll]
    for (int i = 0; i < 16; i++) {
        tap_vals[i] = inputs[base + 2 + i];
    }
    int x = int(inputs[base + 18]);
    int y = int(inputs[base + 19]);

    // Forward.
    float fwd = catmullRomReproject(mvx_val, mvy_val, x, y, tap_vals);

    // Backward with grad_output = 1.
    var dp_mvx = diffPair(mvx_val);
    var dp_mvy = diffPair(mvy_val);
    var dp_taps = diffPair(tap_vals);
    bwd_diff(catmullRomReproject)(dp_mvx, dp_mvy, x, y, dp_taps, 1.0f);

    CatmullRomGrads g;
    g.forward_val = fwd;
    g.grad_mvx = dp_mvx.d;
    g.grad_mvy = dp_mvy.d;
    [ForceUnroll]
    for (int j = 0; j < 16; j++) {
        g.grad_taps[j] = dp_taps.d[j];
    }
    results[tid.x] = g;
}
