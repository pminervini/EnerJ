package com.neuralnoise.enerj.activation;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import com.google.common.primitives.Ints;

public abstract class AbstractActivationFunction {

	public abstract double f(double x);

	public abstract double df(double x);

	public DoubleMatrix1D f(DoubleMatrix1D x) {
		final int N = Ints.checkedCast(x.size());
		DoubleMatrix1D y = new DenseDoubleMatrix1D(N);
		for (int i = 0; i < N; ++i) {
			y.set(i, f(x.get(i)));
		}
		return y;
	}

	public DoubleMatrix2D f(DoubleMatrix2D X) {
		final int R = X.rows(), C = X.columns();
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(R, C);
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				Y.set(r, c, f(X.get(r, c)));
			}
		}
		return Y;
	}

	public DoubleMatrix1D df(DoubleMatrix1D x) {
		final int N = Ints.checkedCast(x.size());
		DoubleMatrix1D y = new DenseDoubleMatrix1D(N);
		for (int i = 0; i < N; ++i) {
			y.set(i, df(x.get(i)));
		}
		return y;
	}

	public DoubleMatrix2D df(DoubleMatrix2D X) {
		final int R = X.rows(), C = X.columns();
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(R, C);
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				Y.set(r, c, df(X.get(r, c)));
			}
		}
		return Y;
	}

}
