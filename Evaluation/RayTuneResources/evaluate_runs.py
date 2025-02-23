import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from capymoa.drift.detectors import ADWIN
from capymoa.evaluation import prequential_evaluation

from Evaluation._config import ADWIN_DELTA_STANDIN
from Evaluation.ComparisionDNNClassifier.SimpleDNN.SimpleDNNClassifier import SimpleDNNClassifier
from Evaluation.EvaluationFunctions import __evaluate_on_stream, __write_summary
from Evaluation.PlottingFunctions import __plot_and_save_result
from Evaluation.config_handling import load_config, config_to_stream, config_to_learner, config_to_loss_fn


def evaluate_adl_run(run_id):
    config = load_config(run_id)
    return evaluate_adl_run_config(config, run_id)


def evaluate_simple_run(run_id):
    config = load_config(run_id)
    return evaluate_simple_dnn_config(config, run_id)


def evaluate_simple_dnn_config(config, run_id):
    stream = config_to_stream(config['stream'])
    learner = SimpleDNNClassifier(
        schema=stream.schema,
        lr=config['lr'],
        model_structure=config['model_structure'],
    )
    print("--------------------------------------------------------------------------")
    print(f"---------------Start time: {datetime.now()}---------------------")
    total_time_start = time.time_ns()
    results_ht = prequential_evaluation(stream=stream, learner=learner, window_size=100, optimise=True, store_predictions=False, store_y=False, max_instances=MAX_INSTANCES_TEST)
    total_time_end = time.time_ns()
    print(f"---------------End time: {datetime.now()}-----------------------")
    print(f"total time spend training the network: {(total_time_end - total_time_start):.2E}ns, that equals {(total_time_end - total_time_start) / 10 ** 9:.2E}s or {(total_time_end - total_time_start) / 10 ** 9 /60:.2f}min")
    print(f"instances={results_ht.cumulative.metrics_dict()['instances']}, accuracy={results_ht.cumulative.metrics_dict()['accuracy']}")
    print("--------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------")

    results_path = Path("results/runs") / f"runID={run_id}" / f"lr={config['lr']:.4e}_modelStructure={config['model_structure']}" / config['stream']
    results_path.mkdir(parents=True, exist_ok=True)
    results_at_end = pd.DataFrame([results_ht.cumulative.metrics()], columns=results_ht.cumulative.metrics_header())
    results_at_end['lr'] = config['lr']
    results_at_end.insert(loc=0, column='model_structure', value=str(config['model_structure']))
    results_at_end.to_pickle(results_path / "metrics.pickle")
    results_ht.windowed.metrics_per_window().to_pickle(results_path / "metrics_per_window.pickle")
    results_at_end.to_csv(results_path / "summary.csv")


def evaluate_adl_run_config(config, run_id):
    classifier = config_to_learner(*config['learner'], grace_period=config['grace_period'])
    added_params = {
        "mci_threshold_for_layer_pruning": config['mci'],
        'drift_detector': ADWIN(config['adwin-delta']),
        'lr': config['lr'],
        'loss_fn': config_to_loss_fn(config['loss_fn']),
    }
    renames = {
        "MCICutOff": f"{config['mci']:4e}",
        ADWIN_DELTA_STANDIN: f"{config['adwin-delta']:.4e}",
        'classifier': config_to_learner(*config['learner'], grace_period=None).name(),
        'lr': f"{config['lr']:.4e}",
        'loss_fn': config['loss_fn'],
    }
    added_names = {'MCICutOff', 'classifier', 'stream', ADWIN_DELTA_STANDIN, 'lr', 'loss_fn'}

    if config['grace_period'] is not None and config['grace_period'][1] == 'global_grace':
        renames['globalGracePeriod'] = config['grace_period'][0]
        added_names.add('globalGracePeriod')
    elif config['grace_period'] is not None and config['grace_period'][1] == 'layer_grace':
        renames['gracePeriodPerLayer'] = config['grace_period'][0]
        added_names.add('gracePeriodPerLayer')
    if 'WithUserChosenWeightLR' in classifier.name():
        added_params['layer_weight_learning_rate'] = config['layer_weight_learning_rate']
        renames['layerWeightLR'] = config['layer_weight_learning_rate']
        added_names.add('layerWeightLR')

    __evaluate_on_stream(
        stream_data=config_to_stream(config['stream']),
        run_id=run_id,
        classifier=classifier,
        adl_parameters=added_params,
        rename_values=renames,
        stream_name=config['stream'],
    )
    __write_summary(run_id, added_names)
    __plot_and_save_result(run_id, show=False)
