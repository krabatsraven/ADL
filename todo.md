- [x] Kirstenml zum repo hinzufügen
- [ ] adl class:
  - [x] structure:
    - [x] one linear layer is one row of nodes?
    - [x] how to handle first pass/ how to initialise?
      -  one layer with in and out
  - [ ] forward:
    - [x] just the forward pass
    - [ ] question: voting: according to paper max, but shouldn't it be argmax?
  - [ ] backward:
  - 
    - order:
      1. normal backward (__super\__())
      2. high lvl
      3. low lvl
    - [ ] research: maybe backward hook? they hook just before, and just after?
  - [ ] high_level_learning:
    - [ ] _add_layer()
      - [x] how to add layers?
        - new layer in list
        - [x] where?
            - prob at the end
      - [ ] question: new layer initialised with one node $\hat =$ nn.linear(in=x, out=1)?
    - [ ] _merge_layers()
      - [ ] implement at first: just remove voting rights, keep nodes 
      - [ ] check in code does: merging might just delete the voting rights of the hidden layer or the hidden layer total?
  - [ ] low_lvl_learning:
      - [ ] _add_node()
          - [ ] how to add nodes?
            - new layer[:-1] = weights
            - new layer[-1] = random
            - the layer after needs a new column:
            - layer l changes from: (in_l, out_l) -> (in_l, out_l + 1) and layer l + 1 from (in_{l+1}, out) -> (in_{l+1} + 1, out)
          - new node is initialised with the xavier initialization
            - [ ] _merge_nodes()
- [ ] test for adl class:
  - [x] forward()
  - [ ] high_level_learning
  - [ ] low_lvl_learning
  - [ ] _add_node()
  - [ ] _add_layer()
  - [ ] _merge_layers()
  - [ ] _merge_nodes()

- probleme: optimizer kennt die gewichte der neuen layer noch nicht
  - kirstens lösung: optimizer neu initialisieren
- cross-entropy-loss funktion in pytorch führt automatisch soft max aus

