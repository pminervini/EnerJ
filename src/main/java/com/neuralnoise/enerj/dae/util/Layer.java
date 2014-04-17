package com.neuralnoise.enerj.dae.util;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

public class Layer {

	private final DoubleMatrix2D W;
	private final DoubleMatrix1D B1, B2;

	public Layer(DoubleMatrix2D W, DoubleMatrix1D B1, DoubleMatrix1D B2) {
		this.W = W;
		this.B1 = B1;
		this.B2 = B2;
	}

	public DoubleMatrix2D getW() {
		return W;
	}

	public DoubleMatrix1D getB1() {
		return B1;
	}

	public DoubleMatrix1D getB2() {
		return B2;
	}

}
