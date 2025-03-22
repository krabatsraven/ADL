# idee 1:
warum werden die gewichte in adl so kompliziert gelernt? die sind doch direkt an der prediction beteiligt? könnte man die nicht direkt mit sgd oder welchem optimizer auch immer optimieren?

also einfach als weitere parameter?

in fact ist die weighted sum die die leute benutzen nicht einfach ein  linear layer mit in="nr of active layer" * "amount of classes" und out="amount of classes"?

# idee 2:
die verwenden im paper einen anderen drift detector