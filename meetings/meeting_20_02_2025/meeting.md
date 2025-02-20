# Wir haben die falsche Loss function verwendet?
## Aus der Pytorch Dokumentation:  
**"The input is expected to contain the unnormalized logits for each class"**
### der unreduzierte Cross Entropy Loss:
![unreduced_loss_CrossEntropy.png](images%2Funreduced_loss_CrossEntropy.png)  
Default ist reduction = 'mean':  
![reduction_CrossEntropyLoss.png](images%2Freduction_CrossEntropyLoss.png)   
**"Note that this case is equivalent to applying LogSoftmax on an input, followed by NLLLoss"**  
  
ADL wendet einen Softmax bereits im Forward Pass an, daher ist CrossEntropy vielleicht nicht die geeigneteste Loss Funktion. 
## Vorschlag einer neuen Loss Funktion:
```python
import torch
from torch import nn
nr_of_classes, idx_of_true_class = 4, 1

y_pred: torch.Tensor = torch.rand(nr_of_classes)
y_true: torch.Tensor = torch.tensor(idx_of_true_class, dtype=torch.int)
nn.NLLLoss()(torch.log(y_pred), y_true)
```
## Vergleich der alten gegen die neue Loss Funktion für ein Set an Hyperparametern:  
  
![compare_diff_loss_fn.png](plots/compare_diff_loss_fn.png)
  
# Decoupeling von Learning Rate und Weight Correction Factor um LR zu senken?
## Vergleich best LR Coupled vs Best LR Decoupled

[//]: #todo: (run best LR Coupled)

[//]: #todo: (compare best coupled vs best decoupled by plot)

# Syntetic Streams Build:
## Type of Streams:
|                 Type | Agrawal  | SEA | details                                                                                                                     |  
|---------------------:|:--------:|:---------:|:----------------------------------------------------------------------------------------------------------------------------|
|             no drift | &#x2611; | &#x2611; | Function 1                                                                                                                  |  
|            one drift | &#x2611; | &#x2611; | Function 1 -> abrupt drift @ 5000 -> Function 3                                                                             |  
|         three drifts | &#x2611; | &#x2611; | Function 1 -> abrupt drift @ 5000 -> Function 3 -> abrupt drift @ 10000 -> Function 4 -> abrupt drift @ 15000 -> Function 1 |
| drift back and forth | &#x2611; | &#x2611; | Function 1 -> abrupt drift @ 5000 -> Function 3 -> abrupt drift @ 10000 -> Function 1                                       |

## Results for ADL on Types of Streams  

|                 Type | Agrawal |  SEA   |
|---------------------:|:--------|:------:|
|             no drift | 53.3%   | 54.06% |  
|            one drift | 53.54%  | 82.00% | 
|         three drifts | 56.68%  | 80.95% |
| drift back and forth | 65.71%  | 81.1%  |

# Suchraum einschränken:
## 1. Versuch: einfach jeweils die drei Parameter nehmen die am besten auf EletricityTiny performed haben und sie auf Electricity testen:
### Lr:
- LinearLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001)
- ExponentialLearningRateProgression(initial_learning_rate=1, decay_alpha=0.001)
- 5e-1
- 5e-2
- 1e-3
> 4 Werte weil mir eigentlich alle werte außer den letzen beiden zu hoch sind.
### MCI:
- 1e-6
- 1e-7
- 1e-8
> in tiny haben wir damit immer so um die 6-10 activen layer am Ende gehabt
### adwin-delta:
- 1e-3
- 1e-5
- 1e-7
### grace period per layer
- 4
- 8
- 16?
- None
> finding aus der isolation: je höher die grace period, desto schlechter das ergebnis
> und per layer out-performed global nach dem ich global auch anwende #-.-  
> es kann sein, dass auf mehr instanzen größere grace periods sinn machen bzw positive sind,
> weil auf den kleinen datastreams haben wir ja anscheinend nur das problem nicht schnell genug lernen zu können
> Grace Period None: Sieht sehr lange Trainingszeiten, habe ich dann erstmal zu grace period=1 gemacht.

# Ray Tunes:  
kein Grid Search mehr, sondern Probing:
## 1. Suchraum:
- maximal 50000
- frühester abbruch nach = 500
- anzahl stichproben = 500
- 'learner': ('vectorized', 'winning_layer', 'decoupled_lrs')
- stream: only one stream at a time
- 'lr': tune.loguniform(1e-4, 5e-1)
- 'layer_weight_learning_rate': tune.loguniform(1e-4, 5e-1),
- 'adwin-delta': tune.loguniform(1e-7, 1e-3),
- 'mci': tune.loguniform(1e-7, 1e-5),
- 'loss_fn': 'NLLLoss'
- 'grace_period': choice aus: global/layer/none in 4,8,16,32

### Ergebniss des ersten Suchraums:

- "lr": 0.17037433308206834,
- "layer_weight_learning_rate": 0.0051048969488651065,
- "adwin-delta": 2.2019797256079463e-05,
- "mci": 2.3105218391180886e-07,
- "grace_period": global, 32  
  
=> 82.15% acc bei 6 hidden, 6 active, und 1502 nodes in hidden layern bei 45000 instancen

<img alt="result_first_ray_tune_search_space.png" src="plots/result_first_ray_tune_search_space.png" width="500"/>

### These: min_runs=500 zu niedrig, bestraft anfänglich langsame lerner

> Nachteil von hohem min_run: suchen dauern sehr lange  
> exemplarisch für Electricity

[//]: #todo: (run mit min_run=4000 für electricity again und höherer LR im Suchraum -.-)

## 2. Suchraum:
- maximal 50000
- frühester abbruch nach = 500
- anzahl stichproben = 500
- 'learner': ('vectorized', 'winning_layer', 'decoupled_lrs')
- stream: only one stream at a time
- 'lr': tune.loguniform(1e-4, 5e-2), (habe die obere grenze extra niedriger gesetzt um lr zu bekommen die "gut" sind)
- 'layer_weight_learning_rate': tune.loguniform(1e-4, 5e-2),
- 'adwin-delta': tune.loguniform(1e-7, 1e-3),
- 'mci': tune.loguniform(1e-7, 1e-5),
- 'loss_fn': 'NLLLoss'
- 'grace_period': choice aus: global/layer/1 in 4,8,16,32

### Ergebnisse aus 2. Suchraum:
> vgl tabelle bei streams
> beste hyperparameter:

[//]: #todo (add hyperparameter as config)
|                 Type | Agrawal | SEA  |
|---------------------:|:--------|:----:|
|             no drift | xx%     | xx%  |  
|            one drift | xx%     | xx%  |
|         three drifts | xx%     | xx%  |
| drift back and forth | xx%     | xx%  |


# Comparision Network
## Strukture
![Skizze_simple_dnn.png](images%2FSkizze_simple_dnn.png)  

## Results on Electricity

Suche durch den Suchraum:
**alle kombinationen an 2er potenzen an nodes mit genau so vielen layern wie das adl netzwerk (solange die anzahl an layern kleiner als 9 ist, sonst ist space complexität zu groß)**
```python
from itertools import product
from ray import tune
import numpy as np

def SimpleDNNSearchSpace(stream_name: str, nr_of_hidden_layers: int = 5, nr_of_neurons: int = 2**12):
   """
   creates a search space for the SimpleDNN model
   that has no more than nr_of_hidden_layers many linear layers
   and in total not more than 2*nr_of_neurons many nodes
   """
   if nr_of_neurons > 256:
      list_of_possible_neuron_configs = [
         list(perm)
         for h in range(1, nr_of_hidden_layers + 1)
         for perm in product(list(map(int, 2 ** np.arange(8, int(np.ceil(np.log2(nr_of_neurons))) + 1))), repeat=h)
         if np.sum(perm) <= 2**np.ceil(np.log2(nr_of_neurons))
      ]
   else:
      list_of_possible_neuron_configs = [
         list(perm)
         for h in range(1, nr_of_hidden_layers + 1)
         for perm in product(list(map(int, 2 ** np.arange(int(np.ceil(np.log2(nr_of_neurons))) + 1))), repeat=h)
         if np.sum(perm) <= 2**np.ceil(np.log2(nr_of_neurons))
      ]
   return {
      "lr": tune.loguniform(1e-4, 5e-1),
      "model_structure": tune.choice(list_of_possible_neuron_configs),
      'stream': tune.grid_search([stream_name])
   }
```
**also nicht jedes mal die gleiche model struktur**
lr = 0,005
model 1 layer mit 4096 Nodes
Acc: 85.23%

## Results for ADL on Types of Streams
Zur Erinnerung:  

|                 Type | Agrawal |  SEA   |
|---------------------:|:--------|:------:|
|             no drift | 53.3%   | 54.06% |  
|            one drift | 53.54%  | 82.00% | 
|         three drifts | 56.68%  | 80.95% |
| drift back and forth | 65.71%  | 81.1%  |

## Result for Comparision Network
|                 Type | Agrawal |   SEA  |
|---------------------:|:--------|:------:|
|             no drift | 58,19%  | 57.57% |  
|            one drift | 54.53%  | 84.25% | 
|         three drifts | 64.31%  | 83.94% |
| drift back and forth | 65.584% | 83.61% |


# Hidden layers Disablen:
## Einfacher Weg:
```python
from torch import nn
nr_of_inputs, nr_of_nodes = 3, 4

nn.Linear(nr_of_inputs, nr_of_nodes).requires_grad_(False)
```
Für alle Hidden layer die gelöscht werden.
ohne Gradient keine Berechnung von Backward
aber im Forward immer noch anwendung der Matrixmultiplikation und der Sigmoidfunction

implementiert in :
```python
from ADLClassifier import ADLClassifier, disabeling_deleted_layers

adl_classifier = ADLClassifier()
disabeling_deleted_layers(adl_classifier)
```
## Proposal:
![How_to_Delete_hidden_layer_skizze.jpg](images%2FHow_to_Delete_hidden_layer_skizze.jpg)

implementiert in:
```python
from ADLClassifier import ADLClassifier, delete_deleted_layers

adl_classifier = ADLClassifier()
delete_deleted_layers(adl_classifier)
```
# Ergebnisse des disablen von hidden layern:
keine zeit mehr für runs gehabt
## Accuracy Changes

[//]: #todo: (run best electricity mit neuem classifier run=1 and enable emission tracking)  
[//]: #todo: (add comparision plot here)
## Emission Changes
[//]: #todo: (rerun best electricity mit emsission tracking enabled run=1)

[//]: #todo: (add comparision of co2 here)

# Notizen:
1. Wenn Concept Change passiert macht eine Learning Rate Progression nur Sinn wenn sie dann wieder von vorne beginnt  
   -> Future Work (nach dem 20.03.)
2. Future Work: Write Capymoa classifier that runs the Matlab Implementation (for benchmarking reasons)

