from Evaluation.benchmark_protokoll import run_bench

# lr=0.0408738, weight-lr=0.00139971, delta=2.06634e-07, mci=8.66358e-06, (4, 'global_grace'), electricity, NLLLoss: 84.815%

if __name__ == "__main__":
    run_bench()
    # todo: effects compare normalization, one hot_encoding