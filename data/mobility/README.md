# Mobility data

`configs/paper.yaml` expects `sumo_trace.csv` in this directory. Generate it with:

```bash
python tools/export_sumo_trace.py --sumocfg /path/to/scenario.sumocfg \
  --output data/mobility/sumo_trace.csv --epoch 1
```

Required columns are:

```text
time_s,vehicle_id,x_m,y_m,speed_m_s,heading_rad
```

The repository does not relabel synthetic trajectories as SUMO data. The synthetic mode in `smoke.yaml` is only for tests and installation checks.

