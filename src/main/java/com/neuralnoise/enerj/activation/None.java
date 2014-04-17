package com.neuralnoise.enerj.activation;

public class None extends AbstractActivationFunction {

	public static None create() {
		return new None();
	}
	
	@Override
	public double f(double x) {
		return x;
	}

	@Override
	public double df(double x) {
		return 1.0;
	}

}
