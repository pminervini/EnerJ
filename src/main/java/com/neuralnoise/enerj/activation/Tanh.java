package com.neuralnoise.enerj.activation;

public class Tanh extends AbstractActivationFunction {

	public static Tanh create() {
		return new Tanh();
	}

	@Override
	public double f(double x) {
		return Math.tanh(x);
	}

	@Override
	public double df(double x) {
		final double tanh = Math.tanh(x);
		return 1.0 - (tanh * tanh);
	}

}
