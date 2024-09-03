- [x] Kirstenml zum repo hinzufügen
- [ ] adl class:
  - [x] structure:
    - defines the layers of the adl
    - [x] one linear layer is one row of nodes?
    - [x] how to handle first pass/ how to initialise?
      -  one layer with in and out
  - [ ] forward:
    - defines the forward pass through the network
    - [x] just the forward pass
    - [ ] question: voting: according to paper max, but shouldn't it be argmax?
  - [ ] _add_layer() 
    - add an untrained layer 
    - [x] how to add layers?
      - new layer in list
      - [x] where?
        - prob at the end
    - [ ] question: new layer initialised with one node $\hat =$ nn.linear(in=x, out=1)?
  - [ ] _add_node()
      - adds nodes to a layer
      - [ ] how to add nodes?
      - new layer[:-1] = weights
      - new layer[-1] = random
      - the layer after needs a new column:
      - layer l changes from: (in_l, out_l) -> (in_l, out_l + 1) and layer l + 1 from (in_{l+1}, out) -> (in_{l+1} + 1, out)
      - new node is initialised with the xavier initialization
        - question: here or in the optimizer?
  - [ ] _merge_layers()
    - merges two layers
    - [ ] implement at first: just remove voting rights, keep nodes
    - [ ] check in code does: merging might just delete the voting rights of the hidden layer or the hidden layer total?
  - [ ] _merge_nodes()
    - prunes nodes?

- [ ] test for adl class:
  - [x] forward()
  - [ ] high_level_learning
  - [ ] low_lvl_learning
  - [ ] _add_node()
  - [x] _add_layer()
  - [ ] _merge_layers()
  - [ ] _merge_nodes()  

- [ ] optimizer:
  - [ ] backward:
    - algo:
      1. normal backward (__super\__())
      2. adjusting of the weights $\beta$
      2. high lvl
      3. low lvl
    - [ ] research: maybe backward hook? they hook just before, and just after?
  - [ ] high_level_learning:
    - algo:
      1. hidden layer pruning
      3. drift detection
    - [ ] hidden_layer_pruning:
      - find correlated layers
      - merge them
    - model._merge_layer()
    - ?
    - [ ] drift_detection:
      - [ ] drift detection with capimoa
      - [ ] add new layer:
    1. model.__add_layer()
    2. train new layer:
    1. set weight of new layer temporarily to 1, all else to 0
    2. train with buffered/current data
    3. adjust weights?
    - q: old weights are used?
    - q: weight of new layer?
  - [ ] low_lvl_learning
    1. hidden node growing
    2. hidden node pruning

- loss function

- Training loop

- [ ] tests for optimizer

- [ ] tests for loss?

- probleme: optimizer kennt die gewichte der neuen layer noch nicht
  - kirstens lösung: optimizer neu initialisieren
  - idee: optimizer muss möglicherweise sowieso teilweise neu geschrieben werden? vllt kann man dann dem optimizer auch einen iter geben der sich ändert. Internetrecherche hat auch keine Löusng außer neue Instanz ergeben.
- cross-entropy-loss funktion in pytorch führt automatisch soft max aus

