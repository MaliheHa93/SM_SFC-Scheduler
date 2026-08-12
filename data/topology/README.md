# Topology data

The default topology is generated reproducibly as 24 RSUs and 43 bidirectional physical links. Export one realization for inspection with:

```bash
python tools/make_topology_csv.py --config configs/default.yaml
```

To use measured/custom RSU data, set both `network.topology_nodes_csv` and `network.topology_links_csv`. See `stg_ddqn/topology.py` for the required columns.

