import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path

# ----------------------------
# CONFIGURATION
# ----------------------------
np.random.seed(7)

DURATION_S = 20.0          # test duration in seconds
FS_HZ = 100                # simulated sample rate
BASE_LATENCY_MS = 120.0    # base display latency
LATENCY_JITTER_MS = 30.0   # stddev of jitter
DROP_RATE = 0.02           # packet loss rate (2%)
ALPHA = 0.1                # smoothing filter for display
DISPLAY_VALUE_NOISE_STD = 1.0  # measurement noise
STRAIN_FULL_SCALE = 2000.0     # used for % error

OUTDIR = Path("results")   # saves all CSVs and plots
OUTDIR.mkdir(exist_ok=True)

# ----------------------------
# Generate synthetic truth data
# ----------------------------
t = np.arange(0, DURATION_S, 1/FS_HZ)
N = len(t)
seq = np.arange(N)
t_sample_ms = t * 1000 + np.random.normal(0, 1, N)

# Strain = ramp + sine
strain_truth = (1000 / DURATION_S) * t + 300 * np.sin(2*np.pi*2*t)

# Temperature = step
temp_truth = np.full(N, 25.0)
temp_truth[N//2:] += 20.0

# Acoustic = base + bursts
acoustic_truth = np.full(N, 30.0)
for start_s in [5, 12, 17]:
    start = int(start_s * FS_HZ)
    end = int((start_s + 0.8) * FS_HZ)
    acoustic_truth[start:end] += 20

truth_df = pd.DataFrame({
    "seq": seq,
    "t_sample_ms": t_sample_ms,
    "strain_uE_truth": strain_truth,
    "temp_C_truth": temp_truth,
    "acoustic_dB_truth": acoustic_truth
})
truth_df.to_csv(OUTDIR / "truth_stream.csv", index=False)

# ----------------------------
# Simulate displayed data
# ----------------------------
latency_ms = BASE_LATENCY_MS + np.random.normal(0, LATENCY_JITTER_MS, N)
t_display_ms = t_sample_ms + latency_ms
drop_mask = np.random.rand(N) < DROP_RATE

# Apply smoothing & noise
displayed_strain = np.zeros(N)
for i in range(N):
    prev = displayed_strain[i-1] if i > 0 else strain_truth[0]
    displayed_strain[i] = (1-ALPHA)*prev + ALPHA*strain_truth[i]
displayed_strain += np.random.normal(0, DISPLAY_VALUE_NOISE_STD, N)

display_df = pd.DataFrame({
    "seq": seq,
    "t_display_ms": t_display_ms,
    "strain_uE_display": displayed_strain
})
display_df = display_df[~drop_mask]
display_df.to_csv(OUTDIR / "display_log.csv", index=False)

# ----------------------------
# Join & compute metrics
# ----------------------------
joined = pd.merge(display_df, truth_df[["seq", "t_sample_ms", "strain_uE_truth"]], on="seq")
joined["latency_ms"] = joined["t_display_ms"] - joined["t_sample_ms"]
joined["error_uE"] = joined["strain_uE_display"] - joined["strain_uE_truth"]
joined["error_pct"] = (joined["error_uE"].abs()/STRAIN_FULL_SCALE)*100

rmse = sqrt(np.mean(joined["error_uE"]**2))
mae = np.mean(np.abs(joined["error_uE"]))
max_abs = np.max(np.abs(joined["error_uE"]))
mean_latency = joined["latency_ms"].mean()
p95 = joined["latency_ms"].quantile(0.95)
p99 = joined["latency_ms"].quantile(0.99)
loss_pct = 100*(1 - len(display_df)/len(truth_df))

# ----------------------------
# Save metrics summary
# ----------------------------
summary = f"""
Display Accuracy Simulation
Samples: {len(truth_df)} | Displayed: {len(display_df)}
Packet Loss: {loss_pct:.2f}%

Value Accuracy:
  RMSE: {rmse:.2f} µε
  MAE: {mae:.2f} µε
  Max(abs): {max_abs:.2f} µε

Latency:
  Mean: {mean_latency:.1f} ms | p95: {p95:.1f} | p99: {p99:.1f}
"""
print(summary)
# replace your current open(...) with this:
with open(OUTDIR / "metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)


# ----------------------------
# Plots
# ----------------------------
plt.figure(figsize=(10,4))
plt.plot(joined["seq"]/FS_HZ, joined["strain_uE_truth"], label="Truth")
plt.plot(joined["seq"]/FS_HZ, joined["strain_uE_display"], label="Displayed")
plt.xlabel("Time (s)"); plt.ylabel("Strain (µε)")
plt.title("Truth vs Display Overlay")
plt.legend(); plt.tight_layout()
plt.savefig(OUTDIR / "overlay_truth_vs_display.png")

plt.figure(figsize=(6,4))
plt.hist(joined["latency_ms"], bins=40)
plt.xlabel("Latency (ms)"); plt.ylabel("Count")
plt.title("Latency Histogram")
plt.tight_layout()
plt.savefig(OUTDIR / "latency_histogram.png")

plt.figure(figsize=(10,4))
plt.plot(joined["seq"]/FS_HZ, joined["error_uE"])
plt.xlabel("Time (s)"); plt.ylabel("Error (µε)")
plt.title("Display Error (Displayed - Truth)")
plt.tight_layout()
plt.savefig(OUTDIR / "error_timeseries.png")

plt.close("all")
print("\nSaved all results in 'results' folder.")
