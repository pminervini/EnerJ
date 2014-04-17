package com.neuralnoise.enerj.regularizer;

public class L2Regularizer extends AbstractRegularizer {

	public static L2Regularizer create() {
		return new L2Regularizer();
	}

	@Override
	public double f(double x) {
		// 1/2 x^2
		return 0.5 * x * x;
	}

	@Override
	public double df(double x) {
		// x
		return x;
	}

}
