"""Generate a reproducible synthetic Bluetooth signal quality dataset.
This is demonstration data, not measurements from physical Bluetooth hardware.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "bluetooth_signal_dataset.csv"

def main(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1, 30, n)
    interference = rng.choice(["low", "medium", "high"], n, p=[0.45, 0.35, 0.20])
    noise_map = {"low": 0, "medium": 1, "high": 2}
    noise = np.array([noise_map[x] for x in interference])
    obstacles = rng.integers(0, 5, n)
    rssi = np.clip(-38 - 1.25*distance - 5.5*noise - 2.3*obstacles + rng.normal(0, 4, n), -100, -25)
    packet_loss = np.clip(0.5 + 0.20*distance + 1.8*noise + 1.15*obstacles + rng.normal(0, 2.3, n), 0, 100)
    latency = np.clip(8 + 0.8*distance + 7*noise + 3.5*obstacles + rng.normal(0, 7, n), 2, 300)
    status_prob = 1/(1+np.exp(-((rssi+78)/5)))
    status = np.where(rng.random(n) < status_prob, "connected", "unstable")
    score = 0.50*((rssi+100)/75) + 0.30*(1-packet_loss/100) + 0.20*(1-np.clip(latency, 0, 200)/200)
    score -= 0.06*noise
    quality = np.where(score >= 0.62, "good", np.where(score >= 0.38, "moderate", "poor"))
    df = pd.DataFrame({
        "distance_m": np.round(distance, 2), "rssi_dbm": np.round(rssi, 2),
        "packet_loss_pct": np.round(packet_loss, 2), "latency_ms": np.round(latency, 2),
        "interference": interference, "obstacles": obstacles,
        "connection_status": status, "quality": quality
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} synthetic demonstration rows to {OUT}")

if __name__ == "__main__":
    main()
