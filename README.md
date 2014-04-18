## EnerJ

This library implements some relatively-popular Deep Learning algorithms for propositional problems (e.g. written digits recognitions), such as Stacked Denoising Autoencoders [Vin10] (SDAE) and Marginalized Stacked Denoising Autoencoders [Che12] (MSDAE).

The library also implements online learning of energy-based models for relational learning, such as the Linear and Bi-Linear models in [Bor14], or the RESCAL model in [Nic11].

[Vin10] Vincent, P. et al. - Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion - JMLR'10

[Che12] Chen, M. et al. - Marginalized Stacked Denoising Autoencoders for Domain Adaptation - ICML'12

[Bor14] Bordes, A. et al. - A semantic matching energy function for learning with multi-relational data - ML'14

[Nic11] Nickel, M. et al. - A Three-Way Model for Collective Learning on Multi-Relational Data - ICML'11

## Example

Training a SDAE on the MNIST digits dataset:

```java
// Loss function
CrossEntropyLoss loss = CrossEntropyLoss.create();

// Regularizers
List<Pair<AbstractRegularizer, Double>> regularizers = Lists.newLinkedList();
{
	AbstractRegularizer l2 = L2Regularizer.create();
	Pair<AbstractRegularizer, Double> reg = Pair.of(l2, 1e-3);
	regularizers.add(reg);
}

// SDAE (specifying input size and the size of the three hidden layers).
//	Sigmoid is the activation function of choice.

SDAE mlae = new SDAE(INPUT_SIZE,
	ImmutableList.of(LAYER_1, LAYER_2, LAYER_3),
	Sigmoid.create(), loss, regularizers, RandomUtils.getPRNG());

// Number of iterations over the dataset
final int ITS = 10;

// SGD step size and Input corruption rate
final double step = 0.1, corr = 0.75;

// Pretrain each layer generatively using backpropagation
mlae.pretrain(xs, corr, step, 0.01, MINITS, MAXITS);

// Train the whole model generatively using backpropagation
for (int t = 0; t < ITS; ++t) {
	double loss = 0.0;

	for (DoubleMatrix1D x : xs) {
		mlae.train(x, corr, step);
		loss += mlae.loss(x);
	}

	log.info("[" + t + "] (T) loss: " + loss);
}
```
