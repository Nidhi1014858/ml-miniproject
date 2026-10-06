# LinkWise Dataset Guide

The LinkWise dataset contains 1,500 simulated satellite telemetry records designed for training a Decision Tree Classifier to prioritize downlink data packets during brief ground station passes.

### Columns & Generation
- **`data_type`**: Category of telemetry packet sampled by operational frequency: Fault alert (15%), Housekeeping (35%), SSTV image (25%), and Voice/Data (25%).
- **`size_kb`**: Data volume in kilobytes, bounded realistically by type (Fault alert: 1-20 KB, Housekeeping: 10-100 KB, SSTV image: 200-800 KB, Voice/Data: 50-400 KB).
- **`battery_pct`**: Satellite state-of-charge (10-100% in Normal mode; 10-50% in Safe mode).
- **`link_quality`**: RF communication condition sampled as Poor (25%), Fair (35%), or Good (40%).
- **`pass_time_min`**: Available contact window with the ground station (0.5 to 12.0 minutes).
- **`sat_mode`**: Spacecraft operational state (Normal: 85%, Safe mode: 15%).
- **`priority`**: Ground downlink priority label: **High**, **Medium**, or **Low**.

### Scoring Rule
Transmission priority is determined by a mission-logic scoring function:
1. **Base Score**: Fault alert (75), Housekeeping (50), SSTV image (40), Voice/Data (30).
2. **Safe Mode**: +20 boost for critical health/alerts; -20 penalty for non-essential payloads.
3. **Low Battery**: +10 for Housekeeping under 30% battery to preserve vital diagnostics.
4. **Channel Penalties**: SSTV and Voice/Data lose points on low battery (<25%: -15) or Poor RF link (-10), gaining +5 for Good links.
5. **Pass Feasibility**: Packets that cannot finish transmitting during the pass window receive a -25 penalty (link speeds: 1 KB/s Poor, 3 KB/s Fair, 6 KB/s Good).
6. **Thresholds**: Score $\ge$ 60 $\rightarrow$ **High**, 35-59 $\rightarrow$ **Medium**, < 35 $\rightarrow$ **Low**.

### Why 6% Label Noise Is Added
Real telemetry scheduling involves human operators, queue overrides, and anomalies. Adding 6% random label noise prevents artificial 100% accuracy, stops the Decision Tree from memorizing deterministic thresholds, and mirrors real-world flight operations.
