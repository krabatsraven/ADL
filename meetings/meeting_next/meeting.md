# idee 1:
warum werden die gewichte in adl so kompliziert gelernt? die sind doch direkt an der prediction beteiligt? könnte man die nicht direkt mit sgd oder welchem optimizer auch immer optimieren?

also einfach als weitere parameter?

in fact ist die weighted sum die die leute benutzen nicht einfach ein  linear layer mit in="nr of active layer" * "amount of classes" und out="amount of classes"?

# idee 2:
die verwenden im paper einen anderen drift detector

\item adl requires a high learning rate to quickly reach a local minimum in loss, is then hindered in that high learning rate to converge on said minimum
\item any concept change restarts that chain, which means that high learning rates perform better on streams with regular concept drifts,
\item if that holds when more subtle drifts are present is not certain.