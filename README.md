# Spin Glasses & Neural Networks

We study how training changes the energy landscape and critical temperature of a multilayer Sherrington–Kirkpatrick spin system associated with a neural network. Neurons map to Ising spins; weights map to couplings.

- **Model:** MNIST classifier with binary activations and continuous weights.
- **Training:** momentum gradient descent with a straight-through estimator.
- **Analysis:** energy evolution and critical-temperature estimation using linearized TAP equations.

[Presentation](docs/PRESENTATION.pdf) · [Experiments](code/notebooks) · [Implementation](code/spin_nn)

Rodion Akinzhala & Ksenia Ibragimova
