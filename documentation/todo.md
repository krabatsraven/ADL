- [x] Kirstenml zum repo hinzufügen
- question [ ]: sanity checks?
- [ ] adl class:
  - [x] structure:
    - defines the layers of the adl
    - [x] one linear layer is one row of nodes?
    - [x] how to handle first pass/ how to initialise?
      -  one layer with in and out
  - [ ] forward:
    - defines the forward pass through the network
    - compare eq 4.1 (1) in the paper
    - [x] just the forward pass
    - [ ] question: voting: according to paper max, but shouldn't it be argmax?
  - [ ] _add_layer() 
    - add an untrained layer 
    - [x] how to add layers?
      - new layer in list
      - [x] where?
        - prob at the end
    - [ ] question: new layer initialised with one node $\hat =$ ```nn.linear(in=x, out=1)```?
  - [ ] _add_node()
      - adds nodes to a layer at the bottom of the matrix
      - gets the index of the layer to add in
      - [x] how to add nodes?
      - new ```layer[:-1] = weights```
      - [ ] ```new_layer[-1]```: xavier initialization
        - [ ] q: xavier normal ($\mathcal N(0,std^2)$) or uniform ($\mathcal U(−a,a)$)?
        - [ ] q: gain = 1?
        - $std = gain \times \sqrt{\frac{2} {fan_in + fan_out}}$
        - $a = gain \times \sqrt{\frac{6} {fan_in + fan_out}}$
      - the layer after needs a new column:
      - layer l changes from: $(in_l, out_l) \to (in_l, out_l + 1)$ and layer $l + 1$ from $(in_{l+1}, out) \to (in_{l+1} + 1, out)$
      - new node is initialised with the xavier initialization
  - [ ] _prune_layer()
    - [x] implement at first: just remove voting rights, keep nodes (_prune_layer_by_vote_removal)
      - removes one layer from voting
      - [ ] q: after removal of the layer should the voting weight be re-normalized?
        - atm it is implemented like this but should be researched
      - [ ] q: they mention that this will speed up learning: ("This strategy also accelerates the model update because the pruned hidden layer is ignored in the
        learning procedure" (p7) do they just mean that without voting the optimizer will only mess with it when it layers after it are optimized or is there something to be done still?)
    - [ ] check in code does: merging might just delete the voting rights of the hidden layer or the hidden layer completely?
  - [x] _delete_node()
    - creates a copy of the layer with the specific node (and its weights and biases) removed
    - gets: the index of the node to be pruned, as well as the index of the layer to prune in
    - ![delete_sketch](images/_delete_node_sketch.jpg)

- [x] test for adl class:
  - [x] forward()
  - [x] _add_node()
  - [x] _add_layer()
  - [x] _prune_layers_by_vote_removal()
  - [x] _merge_nodes()  

- [ ] optimizer:
  - [ ] backward:
    - algo:
      1. normal backward (```loss.backward()``` to get gradient, ```optimizer().step()``` to improve and ```optimizer().zero_grad()``` to reset gradients)
      2. adjusting of the weights $\beta$
      2. high lvl
      3. low lvl
    - [ ] research: maybe backward hook? they hook just before, and just after?
  - [ ] dynamical_voting_weight_adjusting
  - [ ] high_level_learning:
    - algo:
      1. hidden layer pruning
      3. drift detection
    - [ ] hidden_layer_pruning:
      - find correlated layers
      - merge them
    - [ ] drift_detection:
      - [ ] drift detection with capimoa
      - [ ] add new layer:
        1. ```model.__add_layer()```
        2. train new layer:
           1. set weight of new layer temporarily to 1, all else to 0
           2. train with buffered/current data
           3. adjust weights?
      - q: old voting weights are used?
      - q: voting weight of new layer?
  - [ ] low_lvl_learning
    1. hidden node growing
    2. hidden node pruning

- loss function: probably SGD in single pass fashion

- Training loop

- [ ] tests for optimizer
  - [ ] high_level_learning
  - [ ] low_lvl_learning 

- [ ] tests for loss?

- probleme: optimizer kennt die gewichte der neuen layer noch nicht
  - kirstens lösung: optimizer neu initialisieren
  - idee: optimizer muss möglicherweise sowieso teilweise neu geschrieben werden? vllt kann man dann dem optimizer auch einen iter geben der sich ändert. Internetrecherche hat auch keine Löusng außer neue Instanz ergeben.
- cross-entropy-loss funktion in pytorch führt automatisch soft max aus

- Questions out of Curiosity:
  - linear layer between hidden layer and voting to transform vector?