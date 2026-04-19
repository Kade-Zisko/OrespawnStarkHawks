#pragma once
#include <math.h>
#include <Preferences.h>
#include "hive_model_v2_weights.h"

// ── tunables ──────────────────────────────────────────────────────────────────
// Sampling must be hourly for windows to map correctly to time.
// 3600000 ms = 1 hour; lower for bench testing.
#define HVM2_SAMPLE_MS       3600000UL

// Consecutive stressed readings required before triggering alert.
#define HVM2_CONSEC_NEEDED   3

// Decision threshold (0.5 = model default; tune with test_ei_model.py)
#define HVM2_THRESHOLD       0.5f

// Readings needed to complete the per-hive baseline (21 days × 24/day).
#define HVM2_BASELINE_N      504

// ── feature window sizes (in readings, must match training) ───────────────────
#define HVM2_WIN_6H   6
#define HVM2_WIN_24H  24
#define HVM2_WIN_48H  48
#define HVM2_HIST_CAP 52   // ≥ WIN_48H + 1

// ── ring buffer ───────────────────────────────────────────────────────────────
struct HiveHistory {
    float temp[HVM2_HIST_CAP];
    float hum[HVM2_HIST_CAP];
    int   head = 0;
    int   len  = 0;

    void push(float t, float h) {
        temp[head] = t;
        hum[head]  = h;
        head = (head + 1) % HVM2_HIST_CAP;
        if (len < HVM2_HIST_CAP) len++;
    }

    // i=0 → most recent, i=1 → one step ago, …
    float getTemp(int i) const { return temp[(head - 1 - i + HVM2_HIST_CAP) % HVM2_HIST_CAP]; }
    float getHum(int i)  const { return hum[ (head - 1 - i + HVM2_HIST_CAP) % HVM2_HIST_CAP]; }
};

// ── rolling stats (over last n readings, index 0 = most recent) ───────────────
struct RollStats { float mean, std_val, min_val, max_val; };

static RollStats rollStats(float *arr, int n) {
    float sum = 0, sq = 0, mn = arr[0], mx = arr[0];
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        sq  += arr[i] * arr[i];
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }
    float m   = sum / n;
    float var = sq / n - m * m;
    return {m, sqrtf(var > 0 ? var : 0.0f), mn, mx};
}

// ── feature engineering ───────────────────────────────────────────────────────
// Returns false if history too short (need ≥ WIN_48H + 1 readings).
// feat[] order must match FEATURE_COLS in train_classifier_v2.py exactly.
static bool buildFeatures(const HiveHistory &hist, float feat[HVM2_N_IN]) {
    if (hist.len < HVM2_WIN_48H + 1) return false;

    float temps[HVM2_WIN_48H + 1], hums[HVM2_WIN_48H + 1];
    for (int i = 0; i <= HVM2_WIN_48H; i++) {
        temps[i] = hist.getTemp(i);
        hums[i]  = hist.getHum(i);
    }

    int n24 = (hist.len >= HVM2_WIN_24H) ? HVM2_WIN_24H : hist.len;
    RollStats ts = rollStats(temps, n24);
    RollStats hs = rollStats(hums,  n24);

    feat[0]  = temps[0];                        // temperature
    feat[1]  = hums[0];                         // humidity
    feat[2]  = ts.mean;                         // temp_mean_24h
    feat[3]  = ts.std_val;                      // temp_std_24h
    feat[4]  = ts.min_val;                      // temp_min_24h
    feat[5]  = ts.max_val;                      // temp_max_24h
    feat[6]  = ts.max_val - ts.min_val;         // temp_range_24h
    feat[7]  = hs.mean;                         // hum_mean_24h
    feat[8]  = hs.std_val;                      // hum_std_24h
    feat[9]  = hs.min_val;                      // hum_min_24h
    feat[10] = hs.max_val;                      // hum_max_24h
    feat[11] = hs.max_val - hs.min_val;         // hum_range_24h
    feat[12] = temps[0] - temps[HVM2_WIN_6H];  // temp_trend_6h
    feat[13] = hums[0]  - hums[HVM2_WIN_6H];   // hum_trend_6h
    feat[14] = temps[0] - temps[HVM2_WIN_48H]; // temp_trend_48h
    feat[15] = hums[0]  - hums[HVM2_WIN_48H];  // hum_trend_48h

    return true;
}

// ── per-hive baseline (Welford online algorithm) ──────────────────────────────
// Tracks running mean + M2 for all 16 features.
// Once count ≥ HVM2_BASELINE_N, fuses z-score stats with the baked StandardScaler
// into a single affine transform: x_norm = (x - fused_mean) / fused_scale.
struct WelfordState {
    uint32_t count        = 0;
    float    mean[HVM2_N_IN]        = {};
    float    M2[HVM2_N_IN]          = {};
    bool     done                   = false;
    float    fused_mean[HVM2_N_IN]  = {};
    float    fused_scale[HVM2_N_IN] = {};
};

static void welfordUpdate(WelfordState &w, const float feat[HVM2_N_IN]) {
    w.count++;
    for (int i = 0; i < HVM2_N_IN; i++) {
        float delta = feat[i] - w.mean[i];
        w.mean[i] += delta / w.count;
        w.M2[i]   += delta * (feat[i] - w.mean[i]);
    }
}

static void welfordFinalize(WelfordState &w) {
    for (int i = 0; i < HVM2_N_IN; i++) {
        float std = (w.count > 1) ? sqrtf(w.M2[i] / w.count) : 1.0f;
        if (std < 1e-6f) std = 1.0f;
        // Fuse per-hive z-score with StandardScaler into one affine step:
        // x_norm = ((x - hive_mean) / hive_std - scaler_mean) / scaler_scale
        //        = (x - fused_mean) / fused_scale
        w.fused_mean[i]  = w.mean[i] + HVM2_SCALER_MEAN[i]  * std;
        w.fused_scale[i] = HVM2_SCALER_SCALE[i] * std;
    }
    w.done = true;
}

static void welfordSave(Preferences &prefs, const WelfordState &w) {
    prefs.begin("hive_v2", false);
    prefs.putUInt("count",   w.count);
    prefs.putBytes("wf_mean", w.mean, sizeof(w.mean));
    prefs.putBytes("wf_M2",   w.M2,   sizeof(w.M2));
    prefs.putBool("done",    w.done);
    if (w.done) {
        prefs.putBytes("f_mean",  w.fused_mean,  sizeof(w.fused_mean));
        prefs.putBytes("f_scale", w.fused_scale, sizeof(w.fused_scale));
    }
    prefs.end();
}

static void welfordLoad(Preferences &prefs, WelfordState &w) {
    prefs.begin("hive_v2", true);
    w.count = prefs.getUInt("count", 0);
    prefs.getBytes("wf_mean", w.mean, sizeof(w.mean));
    prefs.getBytes("wf_M2",   w.M2,   sizeof(w.M2));
    w.done  = prefs.getBool("done", false);
    if (w.done) {
        prefs.getBytes("f_mean",  w.fused_mean,  sizeof(w.fused_mean));
        prefs.getBytes("f_scale", w.fused_scale, sizeof(w.fused_scale));
    }
    prefs.end();
}

// ── MLP forward pass ──────────────────────────────────────────────────────────
// Dropout is identity at inference — not applied here.
static float hiveForward(const float x[HVM2_N_IN]) {
    // Layer 1: Linear(16→32) + ReLU
    float h1[HVM2_H1];
    for (int i = 0; i < HVM2_H1; i++) {
        float s = HVM2_B1[i];
        for (int j = 0; j < HVM2_N_IN; j++) s += HVM2_W1[i][j] * x[j];
        h1[i] = s > 0.0f ? s : 0.0f;
    }
    // Layer 2: Linear(32→16) + ReLU
    float h2[HVM2_H2];
    for (int i = 0; i < HVM2_H2; i++) {
        float s = HVM2_B2[i];
        for (int j = 0; j < HVM2_H1; j++) s += HVM2_W2[i][j] * h1[j];
        h2[i] = s > 0.0f ? s : 0.0f;
    }
    // Output: Linear(16→1) + Sigmoid
    float logit = HVM2_B3;
    for (int j = 0; j < HVM2_H2; j++) logit += HVM2_W3[j] * h2[j];
    return 1.0f / (1.0f + expf(-logit));
}

// ── HiveClassifier ────────────────────────────────────────────────────────────
// Encapsulates the full pipeline: history → features → normalize → infer.
// Call begin() once in setup(), then infer() once per hourly reading.
class HiveClassifier {
public:
    HiveHistory  hist;
    WelfordState wf;
    Preferences  prefs;
    int          consecCount = 0;

    void begin() {
        welfordLoad(prefs, wf);
        Serial.printf("[hive] baseline count=%u/%u  done=%d\n",
                      wf.count, (uint32_t)HVM2_BASELINE_N, (int)wf.done);
    }

    // Returns "WAIT"        — history too short (< 49 readings)
    //         "CALIBRATING" — accumulating 21-day baseline
    //         "HEALTHY"     — below threshold or consecutive count not met
    //         "STRESSED"    — HVM2_CONSEC_NEEDED consecutive reads above threshold
    // prob is set to raw sigmoid output (0 if not yet inferring).
    const char *infer(float temp, float hum, float *prob) {
        *prob = 0.0f;
        hist.push(temp, hum);

        float feat[HVM2_N_IN];
        if (!buildFeatures(hist, feat)) return "WAIT";

        if (!wf.done) {
            welfordUpdate(wf, feat);
            if (wf.count % 24 == 0) welfordSave(prefs, wf);  // flush every 24 reads
            if (wf.count >= HVM2_BASELINE_N) {
                welfordFinalize(wf);
                welfordSave(prefs, wf);
                Serial.println("[hive] baseline complete — switching to inference");
            }
            return "CALIBRATING";
        }

        // Fused normalization (per-hive z-score + StandardScaler in one step)
        float x_norm[HVM2_N_IN];
        for (int i = 0; i < HVM2_N_IN; i++) {
            float s = wf.fused_scale[i];
            x_norm[i] = (s > 1e-8f) ? (feat[i] - wf.fused_mean[i]) / s : 0.0f;
        }

        *prob = hiveForward(x_norm);

        // Consecutive smoothing
        if (*prob >= HVM2_THRESHOLD) {
            if (++consecCount >= HVM2_CONSEC_NEEDED) return "STRESSED";
        } else {
            consecCount = 0;
        }
        return "HEALTHY";
    }

    // Wipe NVS baseline and restart calibration (e.g. after moving the hive).
    void resetBaseline() {
        wf = WelfordState{};
        welfordSave(prefs, wf);
        consecCount = 0;
        Serial.println("[hive] baseline reset");
    }

    uint32_t baselineCount() const { return wf.count; }
    bool     baselineReady() const { return wf.done; }
};
