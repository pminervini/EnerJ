package com.neuralnoise.enerj.loss;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import com.google.common.primitives.Ints;

public abstract class AbstractLossFunction {

	public abstract double f(double x, double z);

	public abstract double df(double x, double z);

	public DoubleMatrix1D f(DoubleMatrix1D x, DoubleMatrix1D z) {
		final int N = Ints.checkedCast(Math.min(x.size(), z.size()));
		DoubleMatrix1D y = new DenseDoubleMatrix1D(N);
		for (int i = 0; i < N; ++i) {
			y.set(i, f(x.get(i), z.get(i)));
		}
		return y;
	}

	public DoubleMatrix2D f(DoubleMatrix2D X, DoubleMatrix2D Z) {
		final int R = X.rows(), C = X.columns();
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(R, C);
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				Y.set(r, c, f(X.get(r, c), Z.get(r, c)));
			}
		}
		return Y;
	}

	public DoubleMatrix1D df(DoubleMatrix1D x, DoubleMatrix1D z) {
		final int N = Ints.checkedCast(Math.min(x.size(), z.size()));
		DoubleMatrix1D y = new DenseDoubleMatrix1D(N);
		for (int i = 0; i < N; ++i) {
			y.set(i, df(x.get(i), z.get(i)));
		}
		return y;
	}

	public DoubleMatrix2D df(DoubleMatrix2D X, DoubleMatrix2D Z) {
		final int R = X.rows(), C = X.columns();
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(R, C);
		for (int r = 0; r < R; ++r) {
			for (int c = 0; c < C; ++c) {
				Y.set(r, c, df(X.get(r, c), Z.get(r, c)));
			}
		}
		return Y;
	}

}
