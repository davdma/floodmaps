# Benchmarking Models

The arguments for the benchmark config group:
```yaml
trials: 10 # total number of trials per benchmark
max_evals: 5 # max number of evaluations per script run (<trials if walltime is limiting)
save_file: ${paths.benchmarks_dir}/runs_s2.csv # path to save file
save_chkpt_path: ${paths.benchmarks_dir}/chkpt_s2.pkl # path to save chkpt file
seed: 263932 # random seed
```

For each model, the benchmarking script runs an experiment with a unique seed `trials` times,
then calculates the mean and std of the metrics across those runs. The benchmarking results are
saved to a csv file defined by `save_file` path. If experiments are time intensive
and the full `trials` runs might exceed the walltime on the cluster, there is an option to run
only `max_evals` times and saving those results in a checkpoint file at `save_chkpt_path`.
This allows for chunked results.