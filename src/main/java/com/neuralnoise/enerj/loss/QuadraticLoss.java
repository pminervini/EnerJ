package com.neuralnoise.enerj.loss;

public class QuadraticLoss extends AbstractLossFunction {

	public static QuadraticLoss create() {
		return new QuadraticLoss();
	}

	@Override
	public double f(double x, double z) {
		// 1/2 (x - z)^2 = 1/2 (x^2 + z^2 - 2 x z)
		return 0.5 * Math.pow(x - z, 2.0);
	}

	// derivative w.r.t. z
	@Override
	public double df(double x, double z) {
		// (z - x)
		return (z - x);
	}

}
