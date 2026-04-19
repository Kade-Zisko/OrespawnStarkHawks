from arduino.app_utils import *  # type: ignore  # Arduino App Lab SDK (on-board only)
from predict_uno import (
    load_artifacts, build_features, infer,
    load_history, save_history, CONSECUTIVE_NEEDED,
)
from collections import deque
from datetime import datetime

sess, output_name = load_artifacts()
history           = load_history()
consecutive_buf   = deque(maxlen=CONSECUTIVE_NEEDED)
print(f"[predict] model loaded  output={output_name!r}  history={len(history)} readings")


def on_sensor_reading(temp: float, hum: float):
    now = datetime.now()
    history.append((temp, hum))
    save_history(history)

    feat = build_features(list(history), now)
    if feat is None:
        print(f"[predict] WAIT (n={len(history)})")
        Bridge.notify("prediction", "WAIT", 0.0)
        return

    prob, label = infer(sess, output_name, feat, consecutive_buf)
    print(f"[predict] {label}  p={prob:.4f}  T={temp:.1f}  H={hum:.0f}")
    Bridge.notify("prediction", label, float(prob))


Bridge.provide("sensor_reading", on_sensor_reading)

# App.run() must be last — launches Bridge and all registered providers
App.run()
