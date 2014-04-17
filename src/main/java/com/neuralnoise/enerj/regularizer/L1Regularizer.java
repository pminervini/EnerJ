package com.neuralnoise.enerj.regularizer;

public class L1Regularizer extends AbstractRegularizer {

	public static L1Regularizer create() {
		return new L1Regularizer();
	}

	@Override
	public double f(double x) {
		// |x|
		return Math.abs(x);
	}

	@Override
	public double df(double x) {
		// sign(x)
		return Math.signum(x);
	}

}
