- [x] Kirstenml zum repo hinzufügen
- [ ] adl class:
  - [ ] structure:
    - [ ] how to add nodes?
      - new layer[:-1] = weights
      - new layer[-1] = random
      - the layer after needs a new column:
        - layer l changes from: (in_l, out_l) -> (in_l, out_l + 1) and layer l + 1 from (in_{l+1}, out) -> (in_{l+1} + 1, out)
      - new node is initialised with the xavier initialization
    - [x] how to add layers?
      - new layer in list
      - [x] where?
        - prob at the end
    - [x] one linear layer is one row of nodes?
    - [ ] how to handle first pass/ how to initialise?
      -  one layer with in and out
  - [ ] forward:
    - just the forward pass
    - [ ] voting: according to paper max, but shouldn't it be argmax?
  - [ ] backward:
    - order:
      1. normal backward (__super\__())
      2. high lvl
      3. low lvl
    - [ ] research: maybe backward hook? they hook just before, and just after?
  - [ ] high_level_learning:
    - [ ] _add_layer()
    - [ ] _merge_layers()
      - implement at first: just remove voting rights, keep nodes 
      - [ ] check in code does: merging might just delete the voting rights of the hidden layer or the hidden layer total?
  - [ ] low_lvl_learning:
      - [ ] _add_node()
      - [ ] _merge_nodes()
- [ ] test for adl class:
  - [ ] forward()
  - [ ] high_level_learning
  - [ ] low_lvl_learning
  - [ ] _add_node()
  - [ ] _add_layer()
  - [ ] _merge_layers()
  - [ ] _merge_nodes()

- probleme: optimizer kennt die gewichte der neuen layer noch nicht
  - kirstens lösung: optimizer neu initialisieren
- cross-entropy-loss funktion in pytorch führt automatisch soft max aus

