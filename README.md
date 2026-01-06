# Display Accuracy Analysis (Capstone Project)

## Overview
This project evaluates the accuracy and real-time behavior of a visualization system used to display sensor data from an instrumented aircraft rudder. The analysis focuses on verifying that values shown on-screen correctly represent underlying sensor measurements and remain synchronized under realistic latency and packet loss conditions.

This study was conducted **prior to hardware integration** using a simulated sensor data stream to validate the visualization pipeline in a controlled and repeatable manner.

---

## Purpose
The purpose of this analysis is to answer the following questions:

- Does the visualization accurately reflect sensor values (strain, temperature, acoustic)?
- How much latency exists between data generation and on-screen display?
- Is packet loss present, and if so, does it impact display reliability?
- Is the visualization suitable for real-time monitoring during structural testing?

Ensuring display accuracy is critical for interpreting physical behavior during load testing and preventing misinterpretation of experimental results.

---

## Methodology
A synthetic sensor stream was generated to emulate real-time data from embedded rudder sensors:

- **Sampling rate:** 100 Hz  
- **Duration:** 20 seconds  
- **Signals generated:**
  - Strain (ramp + sinusoidal loading)
  - Temperature (step response)
  - Acoustic activity (burst events)

To replicate real-world conditions, the visualization pipeline introduces:
- Fixed base latency (~120 ms)
- Random latency jitter (±30 ms)
- Packet loss (~2%)
- Display smoothing (low-pass filtering)

The simulated “truth” data and displayed values are logged and compared using quantitative performance metrics.

---

## Metrics Evaluated
- **Value Accuracy**
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
  - Maximum absolute error
- **Timing Performance**
  - Mean latency
  - 95th percentile latency (p95)
  - 99th percentile latency (p99)
- **Data Integrity**
  - Packet loss percentage

---

## Results Summary (Example Run)
- Packet loss: **1.85%**
- Strain RMSE: **153.65 microstrain (7.7% full-scale)**
- Mean latency: **119.9 ms**
- p95 latency: **168.4 ms**
- p99 latency: **188.9 ms**

All results met predefined pre-hardware acceptance criteria for display accuracy and real-time responsiveness.

---

## Output Files
Running the analysis script generates the following outputs:

### Data Logs
- `truth_stream.csv` – simulated ground-truth sensor data  
- `display_log.csv` – values received by the visualization  
- `joined_truth_display.csv` – aligned truth vs display comparison  

### Plots
- `overlay_truth_vs_display.png` – truth vs displayed signal comparison  
- `latency_histogram.png` – distribution of display latency  
- `error_timeseries.png` – display error over time  

### Summary
- `metrics_summary.txt` – numerical performance metrics  

## How to Run
### Requirements
- Python 3.10+
- numpy
- pandas
- matplotlib

### Install dependencies
```bash
pip install numpy pandas matplotlib
