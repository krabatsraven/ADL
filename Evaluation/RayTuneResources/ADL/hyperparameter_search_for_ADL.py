from capymoa.drift.detectors import ADWIN
from ray import tune

from ADLClassifier import grace_period_per_layer, global_grace_period
from Evaluation import __evaluate_on_stream, __get_run_id, __plot_and_save_result, __compare_all_of_one_run
from Evaluation.EvaluationFunctions import ADWIN_DELTA_STANDIN, __write_summary
from Evaluation.RayTuneResources.ADL.ADLScheduler import ADLScheduler
from Evaluation.RayTuneResources.ADL.ADLSearchSpace import ADLSearchSpace
from Evaluation.RayTuneResources.ADL.ADLTrainable import ADLTrainable, config_to_learner, config_to_stream


def hyperparameter_search_for_ADL(nr_of_trials: int = 100):
    tuner = tune.Tuner(
        trainable=ADLTrainable,
        tune_config=tune.TuneConfig(
            num_samples=nr_of_trials,
            scheduler=ADLScheduler,
        ),
        param_space=ADLSearchSpace
    )

    results = tuner.fit()
    best_result_config = results.get_best_result(metric="score", mode="max").config
    print(best_result_config)

    classifier = config_to_learner(*best_result_config['learner'], grace_period=best_result_config['grace_period'])
    added_params = {
        "mci_threshold_for_layer_pruning": best_result_config['mci'],
        'drift_detector': ADWIN(best_result_config['adwin-delta']),
        'lr': best_result_config['lr'],

    }
    renames = {
        "MCICutOff": f"{best_result_config['mci']:4e}",
        ADWIN_DELTA_STANDIN: f"{best_result_config['adwin-delta']:.4e}",
        'classifier': config_to_learner(*best_result_config['learner'], grace_period=None).name(),
        'lr': f"{best_result_config['lr']:.4e}",
    }
    added_names = {'MCICutOff', 'classifier', 'stream', ADWIN_DELTA_STANDIN, 'lr'}

    if best_result_config['grace_period'] is not None and best_result_config['grace_period'][1] == 'global_grace':
        renames['globalGracePeriod'] = best_result_config['grace_period'][0]
        added_names.add('globalGracePeriod')

    elif best_result_config['grace_period'] is not None and best_result_config['grace_period'][1] == 'layer_grace':
        renames['gracePeriodPerLayer'] = best_result_config['grace_period'][0]
        added_names.add('gracePeriodPerLayer')

    run_id = __get_run_id()
    __evaluate_on_stream(
        stream_data=config_to_stream(best_result_config['stream']),
        run_id=run_id,
        classifier=classifier,
        adl_parameters=added_params,
        rename_values=renames,
        stream_name=best_result_config['stream'],
    )
    __write_summary(run_id, added_names)
    __plot_and_save_result(run_id, show=False)