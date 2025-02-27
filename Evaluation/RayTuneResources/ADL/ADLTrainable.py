import os
import tempfile

import torch
from capymoa.drift.detectors import ADWIN
from capymoa.misc import save_model, load_model
from ray import train
from ray.train import Checkpoint

from Evaluation.config_handling import config_to_stream, config_to_learner, config_to_loss_fn
from Evaluation._config import MAX_INSTANCES, MIN_INSTANCES


def ADLTrainable(config):
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=config['grace_period'])

    if 'WithUserChosenWeightLR' in learner.name():
        learner = learner(
            schema=stream.get_schema(),
            lr=config['lr'],
            drift_detector=ADWIN(delta=config['adwin-delta']),
            mci_threshold_for_layer_pruning=config['mci'],
            loss_fn=config_to_loss_fn(config['loss_fn']),
            layer_weight_learning_rate=config['layer_weight_learning_rate']
        )
    else:
        learner = learner(
            schema=stream.get_schema(),
            lr=config['lr'],
            drift_detector=ADWIN(delta=config['adwin-delta']),
            mci_threshold_for_layer_pruning=config['mci'],
            loss_fn=config_to_loss_fn(config['loss_fn'])
        )

    checkpoint = train.get_checkpoint()
    print(checkpoint)
    start = 0
    print("before checkpoint")
    if checkpoint is not None:
        print("restoring from checkpoint")
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
            print(f"checkpoint: {checkpoint_dict['instances_seen']}")
            start = checkpoint_dict['instances_seen'] + 1
            learner.state_dict = checkpoint_dict['learner']
            print(learner.evaluator.accuracy())
            assert learner.learning_rate == config['lr']
            assert learner.device == config['device']
            assert learner.drift_detector == ADWIN(delta=config['adwin-delta'])
            assert learner.mci_threshold_for_layer_pruning == config['mci']
            assert learner.loss_fn == config_to_loss_fn(config['loss_fn'])
            assert learner.model.nr_of_active_layers == checkpoint_dict['nr_of_active_layers']

    nr_of_instances_seen = 0

    print('before stream restart')
    stream.restart()
    print('before fast forwarding')
    while stream.has_more_instances and nr_of_instances_seen < start and nr_of_instances_seen < MAX_INSTANCES:
        # fast-forward to start
        _ = stream.get_next_instance()
        nr_of_instances_seen += 1

    print('before training')
    metrics = {"score": 0, 'instances_seen': nr_of_instances_seen}
    while stream.has_more_instances() and nr_of_instances_seen < MAX_INSTANCES:
        instance = stream.next_instance()
        learner.train(instance)
        nr_of_instances_seen += 1
        if nr_of_instances_seen % max(MIN_INSTANCES // 100, 1) == 0:
            metrics = {"score": learner.evaluator.accuracy(), 'instances_seen': nr_of_instances_seen}
            if nr_of_instances_seen % max(MIN_INSTANCES, 1) == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    torch.save(
                        {'instances_seen': nr_of_instances_seen, 'learner': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
                        os.path.join(tmpdir, 'checkpoint.pt')
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))
                    raise Exception(f"planned abort: instances seen: {learner.evaluator.instances_seen}, accuracy: {learner.evaluator.accuracy()}")
            else:
                train.report(metrics=metrics)

    print('after training')
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {'instances_seen': nr_of_instances_seen, 'learner_state': learner.state_dict},
            os.path.join(tmpdir, 'checkpoint.pt')
        )
        train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))