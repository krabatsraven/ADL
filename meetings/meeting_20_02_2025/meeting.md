# Notizen:
1. Wenn Concept Change passiert macht eine Learning Rate Progression nur Sinn wenn sie dann reseted
2. Wir haben die falsche Loss function verwendet?
> laut herrn gpt macht cross entropy bereits elber ein softmax und erwartet reine logits, nlll loss wäre dementsprechend für unsere funktion die bessere loss function wenn wir auf die wsk jeweils den log anwenden
3. Future Work: Write Capymoa classifier that runs the Matlab Implementation (for benchmarking reasons)

# Comparision Network
## Strukture
## Results on Electricity

# Syntetic Streams Build:
## Type of Streams
## Results for ADL on Types of Streams
## Result for Comparision Network

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
> finding aus der isolation: je höher die grace period desto schlechter das ergebnis
> und per layer outpoerformed global nach dem ich global auch anwende #-.-  
> kann sein, dass auf mehr instancen größere grace periods sinn machen bzw positive sind,
> weil auf den kleinen datastreams haben wir ja anscheinend nur das problem nicht schnell genug lernen zu können

# Ray Tunes:
## Suchraum:  

## Ergebniss mit X Nr of Runs:  

# Ergebnisse des disablen von hidden layern:
## Accuracy Changes
## Emission Changes
  
