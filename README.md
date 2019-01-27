# ICLR2019
Reproduction of [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km) paper from ICLR 2019 as a part of [ICLR 2019 Reproducibility Challenge](https://reproducibility-challenge.github.io/iclr_2019)

## Reproduction report
`report/report.pdf` contains the report from the reproduction attempt.

## Training the model
Run the following command to start training for all configurations specified in `config.json`. Specify path to the directory the results and checkpoints will be written to. The code automatically continues training the model for a configuration from the last checkpoint (if present) so you can safely interrupt training and then continue it. This is very useful when training on preemptible VM instances (Cheaper than regular ones) that can be terminated at any time.

```python iclr2019/run_trainings.py --config_path iclr2019/config.json --results_path <path to directory your results will be written to> --data_path <path to directory datasets will be downloaded to>```

## Display results
Run the following command to display the results for all configurations specified in `config.json`.

```python iclr2019/display_results.py --config_path iclr2019/config.json --results_path <path to directory your results will be written to>```

