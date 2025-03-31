import os
import tempfile

import numpy as np
import torch
from capymoa.drift.detectors import ADWIN
from capymoa.misc import save_model, load_model
from ray import train
from ray.train import Checkpoint

from Evaluation.config_handling import config_to_stream, config_to_learner, config_to_loss_fn
from Evaluation._config import MAX_INSTANCES, MIN_INSTANCES


def ADLTrainable(config):
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=(config['grace_period'], config['grace_type']))

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
    start = 0
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
            start = checkpoint_dict['instances_seen'] + 1
            learner.state_dict = checkpoint_dict['learner']
            assert learner.learning_rate == config['lr']
            assert learner.drift_detector.delta == config['adwin-delta']
            assert learner.mci_threshold_for_layer_pruning == config['mci']
            assert learner.loss_function == config_to_loss_fn(config['loss_fn'])
            assert learner.model.nr_of_active_layers == checkpoint_dict['nr_of_active_layers']

    nr_of_instances_seen = 0

    stream.restart()
    while stream.has_more_instances and nr_of_instances_seen < start and nr_of_instances_seen < MAX_INSTANCES:
        # fast-forward to start
        _ = stream.next_instance()
        nr_of_instances_seen += 1

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
            else:
                train.report(metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {'instances_seen': nr_of_instances_seen, 'learner_state': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
            os.path.join(tmpdir, 'checkpoint.pt')
        )
        train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))


def ADLTrainableUnstable(config):
    '''to find a unstable version of adl we add to the accurcay score by the normalized (by the current mean) values of active layers'''
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=(config['grace_period'], config['grace_type']))

    learner = learner(
        schema=stream.get_schema(),
        lr=config['lr'],
        drift_detector=ADWIN(delta=config['adwin-delta']),
        mci_threshold_for_layer_pruning=config['mci'],
        loss_fn=config_to_loss_fn(config['loss_fn']),
        layer_weight_learning_rate=config['layer_weight_learning_rate']
    )
    checkpoint = train.get_checkpoint()
    start = 0
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
            start = checkpoint_dict['instances_seen'] + 1
            learner.state_dict = checkpoint_dict['learner']
            assert learner.learning_rate == config['lr']
            assert learner.drift_detector.delta == config['adwin-delta']
            assert learner.mci_threshold_for_layer_pruning == config['mci']
            assert learner.loss_function == config_to_loss_fn(config['loss_fn'])
            assert learner.model.nr_of_active_layers == checkpoint_dict['nr_of_active_layers']

    nr_of_instances_seen = 0


    stream.restart()
    while stream.has_more_instances and nr_of_instances_seen < start and nr_of_instances_seen < MAX_INSTANCES:
        # fast-forward to start
        _ = stream.next_instance()
        nr_of_instances_seen += 1

    nr_of_active_layers_mean = 1
    nr_of_active_layers_variance = 0
    max_of_nr_of_active_layers_variance = 0
    metrics = {"score": 0, 'instances_seen': nr_of_instances_seen}
    while stream.has_more_instances() and nr_of_instances_seen < MAX_INSTANCES:
        instance = stream.next_instance()
        learner.train(instance)

        nr_of_instances_seen += 1
        nr_of_active_layers_current = learner.model.nr_of_active_layers
        nr_of_active_layers_new_mean = nr_of_active_layers_mean + (nr_of_active_layers_current - nr_of_active_layers_mean) /nr_of_instances_seen
        nr_of_active_layers_variance = nr_of_active_layers_variance + (nr_of_active_layers_current - nr_of_active_layers_new_mean) * (nr_of_active_layers_current - nr_of_active_layers_mean)/nr_of_instances_seen
        nr_of_active_layers_mean = nr_of_active_layers_new_mean
        max_of_nr_of_active_layers_variance = max(max_of_nr_of_active_layers_variance, nr_of_active_layers_variance)

        if nr_of_instances_seen % max(MIN_INSTANCES // 100, 1) == 0:
            metrics = {"score": (learner.evaluator.accuracy() / 100.0) + (nr_of_active_layers_variance)/(max_of_nr_of_active_layers_variance), 'instances_seen': nr_of_instances_seen}
            if nr_of_instances_seen % max(MIN_INSTANCES, 1) == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    torch.save(
                        {'instances_seen': nr_of_instances_seen, 'learner': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
                        os.path.join(tmpdir, 'checkpoint.pt')
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))
            else:
                train.report(metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {'instances_seen': nr_of_instances_seen, 'learner_state': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
            os.path.join(tmpdir, 'checkpoint.pt')
        )
        train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))


def ADLTrainableStable(config):
    '''to find a stable version of adl we punish the accurcay score by the normalized (by the current mean) values of active layers'''
    stream = config_to_stream(config['stream'])
    learner = config_to_learner(*config['learner'], grace_period=(config['grace_period'], config['grace_type']))

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
    start = 0
    if checkpoint is not None:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'), weights_only=False)
            start = checkpoint_dict['instances_seen'] + 1
            learner.state_dict = checkpoint_dict['learner']
            assert learner.learning_rate == config['lr']
            assert learner.drift_detector.delta == config['adwin-delta']
            assert learner.mci_threshold_for_layer_pruning == config['mci']
            assert learner.loss_function == config_to_loss_fn(config['loss_fn'])
            assert learner.model.nr_of_active_layers == checkpoint_dict['nr_of_active_layers']

    nr_of_instances_seen = 0

    stream.restart()
    while stream.has_more_instances and nr_of_instances_seen < start and nr_of_instances_seen < MAX_INSTANCES:
        # fast-forward to start
        _ = stream.next_instance()
        nr_of_instances_seen += 1
    nr_of_active_layers_mean = 1
    nr_of_active_layers_variance = 0
    metrics = {"score": 0, 'instances_seen': nr_of_instances_seen}
    while stream.has_more_instances() and nr_of_instances_seen < MAX_INSTANCES:
        instance = stream.next_instance()
        learner.train(instance)

        nr_of_instances_seen += 1
        nr_of_active_layers_current = learner.model.nr_of_active_layers
        nr_of_active_layers_new_mean = nr_of_active_layers_mean + (nr_of_active_layers_current - nr_of_active_layers_mean) /nr_of_instances_seen
        nr_of_active_layers_variance = nr_of_active_layers_variance + (nr_of_active_layers_current - nr_of_active_layers_new_mean) * (nr_of_active_layers_current - nr_of_active_layers_mean)/nr_of_instances_seen
        nr_of_active_layers_mean = nr_of_active_layers_new_mean

        if nr_of_instances_seen % max(MIN_INSTANCES // 100, 1) == 0:
            metrics = {"score": (learner.evaluator.accuracy() / 100.0) - (nr_of_active_layers_variance / nr_of_active_layers_mean ** 2), 'instances_seen': nr_of_instances_seen}
            if nr_of_instances_seen % max(MIN_INSTANCES, 1) == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    torch.save(
                        {'instances_seen': nr_of_instances_seen, 'learner': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
                        os.path.join(tmpdir, 'checkpoint.pt')
                    )
                    train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))
            else:
                train.report(metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {'instances_seen': nr_of_instances_seen, 'learner_state': learner.state_dict, 'nr_of_active_layers': learner.model.nr_of_active_layers},
            os.path.join(tmpdir, 'checkpoint.pt')
        )
        train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir))