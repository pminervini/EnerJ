package com.neuralnoise.enerj.regularizer;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;

import com.google.common.primitives.Ints;

public abstract class AbstractRegularizer {

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
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(X.rows(), X.columns());
		for (int r = 0; r < X.rows(); ++r) {
			for (int c = 0; c < X.columns(); ++c) {
				Y.set(r, c, f(X.get(r, c)));
			}
		}
		return Y;
	}

	public DoubleMatrix3D f(DoubleMatrix3D X) {
		DoubleMatrix3D Y = new DenseDoubleMatrix3D(X.slices(), X.rows(), X.columns());
		for (int s = 0; s < X.slices(); ++s) {
			for (int r = 0; r < X.rows(); ++r) {
				for (int c = 0; c < X.columns(); ++c) {
					Y.set(s, r, c, f(X.get(s, r, c)));
				}
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
		DoubleMatrix2D Y = new DenseDoubleMatrix2D(X.rows(), X.columns());
		for (int r = 0; r < X.rows(); ++r) {
			for (int c = 0; c < X.columns(); ++c) {
				Y.set(r, c, df(X.get(r, c)));
			}
		}
		return Y;
	}
	
	public DoubleMatrix3D df(DoubleMatrix3D X) {
		DoubleMatrix3D Y = new DenseDoubleMatrix3D(X.slices(), X.rows(), X.columns());
		for (int s = 0; s < X.slices(); ++s) {
			for (int r = 0; r < X.rows(); ++r) {
				for (int c = 0; c < X.columns(); ++c) {
					Y.set(s, r, c, df(X.get(s, r, c)));
				}
			}
		}
		return Y;
	}
	
}
