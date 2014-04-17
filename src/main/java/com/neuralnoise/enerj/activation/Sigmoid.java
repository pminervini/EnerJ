package com.neuralnoise.enerj.activation;

public class Sigmoid extends AbstractActivationFunction {

	public static Sigmoid create() {
		return new Sigmoid();
	}

	@Override
	public double f(double x) {
		final double s = sigmoid(x);
		return s;
	}

	@Override
	public double df(double x) {
		final double s = sigmoid(x);
		return s * (1.0 - s);
	}

	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

}
