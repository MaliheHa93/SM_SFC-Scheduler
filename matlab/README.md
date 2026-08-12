# MATLAB figure generation

`plot_paper_results.m` reads the long summary CSV produced by Python and
exports the same combined and separate figures as the Python plotting module.
It does not invent, smooth, or recompute simulation values.

From MATLAB:

```matlab
cd matlab
plot_paper_results('../results/behavior/evaluation_summary.csv', ...
                   '../plots/matlab_curve_only_behavior')
```

For the final paper, replace the input with
`../results/paper/evaluation_summary.csv` after the ten-seed SUMO experiment.

The output contains only mean curves for STG-DDQN, IM, and DLAPM. Confidence
intervals remain in the CSV for tables and statistical reporting; the script
does not draw bands, patches, or shadows.
