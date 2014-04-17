package com.neuralnoise.enerj.loss;

public class LinearLoss extends AbstractLossFunction {

	public static LinearLoss create() {
		return new LinearLoss();
	}

	@Override
	public double f(double x, double z) {
		// | x - z |
		return Math.abs(x - z);
	}

	// derivative w.r.t. z
	@Override
	public double df(double x, double z) {
		// sqrt((sy - y)^2) / (sy - y)
		return (z == x ? 0.0 : Math.sqrt(Math.pow(z - x, 2.0)) / (z - x));
	}

}
