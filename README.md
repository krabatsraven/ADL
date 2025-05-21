from prompt_toolkit.key_binding.bindings.emacs import load_emacs_shift_selection_bindingsfrom tests.resources import optimizer_choicesfrom tests.resources import optimizer_choices

# Autonomous Deep Learner
Implementation of the Autonomous Deep Learning: Continual Learning Approach for Dynamic Environments Paper by Andri Ashfahani and Mahardhika Pratama
using torch to define the model and capymoa to define the classifier.

[Paper by Andri Ashfahani and Mahardhika Pratama](https://arxiv.org/pdf/1810.07348)

The authors of that paper have designed their own implementation using matlab which can be found [here](https://github.com/andriash001/ADL/). 
To this date we were unable to compare this implementation against ours due to a lacking matlab license.


## Quick Start
As the classifier extends the capymoa one we can you use the functions available to train and evaluate it.
To create a classifier we recommend the fabricator functions. 
```python
from capymoa.evaluation import prequential_evaluation

from data import Electricity

from ADLClassifier import grace_period_per_layer, winning_layer_training, vectorized_for_loop, \
add_weight_correction_parameter_to_user_choices, input_preprocessing
from ADLClassifier import extended_classifier, extend_classifier_for_evaluation

stream_data = Electricity
adl_classifier = extended_classifier(
    input_preprocessing,
    grace_period_per_layer(300), 
    vectorized_for_loop,
    winning_layer_training,
    add_weight_correction_parameter_to_user_choices
)

results_ht = prequential_evaluation(stream=stream_data, learner=adl_classifier)
```

## Structure
The architecture of the adl is kept track in the torch model $\texttt{class AutoDeepLearner(nn.Module)}$.
Meaning that it encapsulates all the expected functionalities of a torch model, and can be trained as such:
```python
import torch

from Model import AutoDeepLearner
from data import Electricity

n = Electricity.n_features
m = Electricity.n_classes

model = AutoDeepLearner(nr_of_features=n, nr_of_classes=m)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_function = lambda pred, label: torch.nn.NLLLoss()(torch.log(pred), label)

while Electricity.has_more_instances():
    data = Electricity.next_instance()
    prediction = model(data)
    loss = loss_function(prediction, data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
Note that this example does not change the architecture of the model.

The base classifier $\texttt{ADLClassifier(Classifier)}$ implemented in capymoa provides $\texttt{train(x: instance) -> None}$ $\texttt{predict(x: instance) -> index of class}$ $\texttt{predict_proba(x: instance) -> class probabilities}$
which are the same as the [capymoa functionalities](https://capymoa.org/api/modules/capymoa.base.Classifier.html#capymoa.base.Classifier), and allows the user utilization of capymoas training functionalities to implement the adl training loop.
```python
from data import Electricity
from ADLClassifier.BaseClassifier import ADLClassifier

stream_data = Electricity
adl_classifier = ADLClassifier(schema=stream_data.schema)

while stream_data.has_more_instances():
    adl_classifier.train(stream_data.next_instance())
```

The base classifiers functions is attempted to be as close as possible on the design of the [paper](https://arxiv.org/pdf/1810.07348) and the [mathlab implementation](https://github.com/andriash001/ADL/), 
the only exception is the winning layer training of the weight matrix and bias vector of the hidden linear layer with the highest voting weight, which is implemented as a feature.
$\texttt{extended_classifier(winning_layer_training, add_weight_correction_parameter_to_user_choices)}$ provides a (close to) canonical classifier.

## Features Available
We differentiated between features only for [evaluation](#evaluation) and features extending the functionality of the [classifier](#classifier-features).

### Classifier Features
A fabricator function for a classifier with the described functionalities is provided in $\texttt{extended_classifier(*decorators: Callable[[type(ADLClassifier)], type(ADLClassifier)]) -> type(ADLClassifier)}$.
The arguments are simply the wrapper functionalities mentioned below, which add the described alterations to the base classifier.

#### Grace Period
$\texttt{global_grace_period(duration: int = 1000)}$ and $\texttt{grace_period_per_layer(duration: int = 1000)}$ both implement a grace period.
Both forbid any changes either to the model in case of the global version, or the specific layer in case of the per layer version 
for the duration of the grace period measured in instances seen.
A new grace period is started when either a node or a layer is added or deleted.
To add the grace period to the classifier either add as a decorator or provide to the factory function as argument with the intented grace period:
$\texttt{extended_classifier(global_grace_period(256))}$ to add a global grace period of 256 instances.
The behaviour of both functionalities together is untested.

#### Layer Deletion Handling
In addition to the default of only deleting the output layer  
$\texttt{disabeling_deleted_layers}$ sets $\texttt{requires_grad}$ of the corresponding hidden layer to False,  
and $\texttt{delete_deleted_layers}$ omits the hidden layers activation function and deletes the hidden layer by merging the weightmatrix with the one of the layer behind it.  
The behaviour of both functionalities together is untested.

#### Adding a Voting Weight Learning Rate
Adds $\texttt{layer_weight_learning_rate}$ as an additional hyperparameter to the kwargs of the adl classifier.
If not used the learning rate doubles as step size to increase/decrease the voting weight correction factor inside the voting weight training.
To use this simply add $\texttt{add_weight_correction_parameter_to_user_choices()}$ to the args of the fabricator function.

#### Input Preprocessing
Adds input normalization as well as one-hot-encoding of the input instance before processing it.
To use this simply add $\texttt{input_preprocessing}$ to the args of the fabricator function.

#### Vectorizing the MCI Calculation
The calculation of the mci calculation for each class probability for each pair of active layers was redone using torch tensors to speed up the calculation.
To use the vectorized version simply add $\texttt{vectorized_for_loop}$ to the args of the fabricator function.

#### Winning Layer
Implements the Winning Layer Training described by the [paper](https://arxiv.org/pdf/1810.07348) to be able to research its effects on the classifier.
To employ winning layer training simply add $\texttt{winning_layer_training}$ to the args of the fabricator function 
otherwise all active output layers and all hidden layers are trained using the optimizer.

### Evaluation
A fabricator function for a classifier with the described evaluation functionalities is provided in $\texttt{extend_classifier_for_evaluation}$.
The argument $\texttt{with_emissions: bool = False}$ chooses whether the codecarbon framework is to be employed. ([Energy Recorder](#recorder-of-emitted-carbon-dioxide-and-used-energy))  
The rest of the arguments are the [classifier features](#classifier-features), where the order of adding should not matter. 

#### Recorder of the current architecture
$\texttt{record_network_graph}$ keeps track of the architecture adl possesses before it processes the instance.
Tracked are the number of hidden layers, the shape of these layers, what layers are active, which layer was the winning layer, 
and what the current learning rate was.
#### Recorder of emitted Carbon-dioxide and used Energy
Using Code Carbon $\texttt{record_emissions}$ tracks the resources extended by the training in $\texttt{_train}$.
This extends the run time of the classifier considerably.


## Future Work
### Other Drift Detector
We used Adwin to detect drifts and trigger layer growth, the paper used different method, 
that to our knowledge was not implemented by one of the bigger libraries.

### Relu Activation Function
The hidden layers use sigmoid activation function, Relu has been shown to converge quicker, 
which could allow to use lower learning rates for the parameter optimization.

### Adam or other Optimizer
The used Stochastic Gradient Decent is a optimizer that has been iterated upon. 
Using a more modern technique could improve the convergence and improve adl performance.

### Learning Rate Progressions
Hyperparameter Tuning has shown that the best performing sets all use a very high learning rate.
A Learning Rate Progression that resets as soon as a new Drift is detected 
could help improve learning for long stretches of the same concept.

### Window Size for Parameters
Adl utilizes a lot of location and distribution measures like the mean of the data or covariance between layers.
In stream learning it is common to use a window of the last n data points to calculate these measures.
Currently, all seen data is kept inside these measures

### Weighted sum/ Voting weight replacement
Instead of the weighted sum to calculate the output a linear layer allows the same functionality,
but with the added benefit that a parameter optimization algorithm can be used to learn the best possible voting weight distribution.
The role of the voting weights could then be replaced by the weight matrix of this new voting layer.

## State of the Work
A lot of this work was done in the context of my undergrad study and is by no mean complete. 
At the moment I lack the time to further research possible improvements for this stream learner, which there might be.
We have left our ideas as [notes](#future-work), as well as issues in the github if onother person is interested to pick some of that up.

## Evaluation Folder
The tests we ran to evaluate the adl-classifier, tune its hyperparameter, 
and a simple static with a variable architecture (but fixed) deep neuronal network to compare adl against.
A more thorough test bench with more varied and more volatile drifts could be useful.
$\texttt{run_bench}$ runs the tests needed for the thesis plots created by $\texttt{ba_plots}$ in sequence, 
while $asyncio.run(bench_async())$ runs them using async with a higher utilization of processor resources. 
The interaction between codecarbon and async io are untested as the final data was created on a slurm cluster 
using $\texttt{mogon_run.sh}$.
