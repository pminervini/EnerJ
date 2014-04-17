package com.neuralnoise.enerj.dae;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.math.tdouble.DoubleFunctions;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class DAE extends AbstractAE {

	private DoubleMatrix2D W;
	private DoubleMatrix1D B1, B2;

	public DAE(final int n, final int m, AbstractActivationFunction activation, AbstractLossFunction loss, List<Pair<AbstractRegularizer, Double>> regularizers, DoubleRandomEngine prng) {
		super(activation, loss, regularizers, prng);

		final double min = -4.0 * (6.0 / Math.sqrt(n + m));
		final double max = +4.0 * (6.0 / Math.sqrt(n + m));

		this.W = RandomUtils.randomMatrix(prng, m, n, min, max);
		this.B1 = RandomUtils.randomVector(prng, m, min, max);
		this.B2 = RandomUtils.randomVector(prng, n, min, max);
	}

	public DoubleMatrix1D w1(DoubleMatrix1D x) {
		DoubleMatrix1D y = MatrixUtils.sum(MatrixUtils.product(this.W, x), this.B1);
		return y;
	}

	// W : m x n, X : n x t
	public DoubleMatrix2D W1(DoubleMatrix2D X) {
		DoubleMatrix2D Y = MatrixUtils.product(this.W, X);
		for (int c = 0; c < Y.columns(); ++c) {
			Y.viewColumn(c).assign(this.B1, DoubleFunctions.plus);
		}
		return Y;
	}

	@Override
	public DoubleMatrix2D f(DoubleMatrix2D X) {
		return this.activation.f(W1(X));
	}

	// W : m x n, Y : m x t
	public DoubleMatrix2D W2(DoubleMatrix2D Y) {
		DoubleMatrix2D Z = MatrixUtils.product(MatrixUtils.transpose(this.W), Y);
		for (int c = 0; c < Z.columns(); ++c) {
			Z.viewColumn(c).assign(this.B2, DoubleFunctions.plus);
		}
		return Z;
	}

	@Override
	public DoubleMatrix2D g(DoubleMatrix2D Y) {
		return this.activation.f(W2(Y));
	}

	// X : n x t
	@Override
	public double loss(DoubleMatrix2D X) {
		DoubleMatrix2D Z = g(f(X));
		DoubleMatrix2D L = this.loss.f(X, Z);

		double t = X.columns();
		
		// XXX
		double dt = t;
		L = MatrixUtils.product(L, 1.0 / dt);

		double r = 0.0;
		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.B1), weight));// * t));
			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.B2), weight));// * t));
			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.W), weight));// * t));
		}

		double ret = MatrixUtils.sum(L) + r;
		return ret;
	}

	// X : n x t
	@Override
	public void train(DoubleMatrix2D X, double p, double step) {
		DoubleMatrix2D tX = corrupt(X, p);

		final int m = this.W.rows(), n = this.W.columns();
		double t = X.columns();

		// A1 : m x t, Y : m x t
		DoubleMatrix2D A1 = W1(tX), Y = this.activation.f(A1);

		// A2 : n x t, Z : n x t
		DoubleMatrix2D A2 = W2(Y), Z = this.activation.f(A2);

		// dl : n x t, ds : n x t
		DoubleMatrix2D dl = this.loss.df(X, Z), ds = this.activation.df(A2);

		// XXX
		double dt = t;
		dl = MatrixUtils.product(dl, 1.0 / dt);

		// d2 : n x t
		DoubleMatrix2D d2 = MatrixUtils.hadamardProduct(dl, ds);

		// d1 : m x t
		DoubleMatrix2D d1 = MatrixUtils.hadamardProduct(MatrixUtils.product(this.W, d2), this.activation.df(A1));

		// B1u: m
		DoubleMatrix1D B1u = new DenseDoubleMatrix1D(m);
		for (int i = 0; i < t; ++i) {
			B1u = MatrixUtils.sum(B1u, d1.viewColumn(i));
		}

		// B2u: n
		DoubleMatrix1D B2u = new DenseDoubleMatrix1D(n);
		for (int i = 0; i < t; ++i) {
			B2u = MatrixUtils.sum(B2u, d2.viewColumn(i));
		}

		// m x t, n x t
		DoubleMatrix2D t1 = MatrixUtils.product(d1, MatrixUtils.transpose(tX));

		// m x t, n x t
		DoubleMatrix2D t2 = MatrixUtils.product(Y, MatrixUtils.transpose(d2));

		// m x n
		DoubleMatrix2D T = MatrixUtils.sum(t1, t2);

		DoubleMatrix2D Wu = T;

		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			B1u = MatrixUtils.sum(B1u, MatrixUtils.product(reg.df(this.B1), weight));// * t));
			B2u = MatrixUtils.sum(B2u, MatrixUtils.product(reg.df(this.B2), weight));// * t));
			Wu = MatrixUtils.sum(Wu, MatrixUtils.product(reg.df(this.W), weight));// * t));
		}

		this.B1 = MatrixUtils.subtraction(this.B1, MatrixUtils.product(B1u, step));
		this.B2 = MatrixUtils.subtraction(this.B2, MatrixUtils.product(B2u, step));
		this.W = MatrixUtils.subtraction(this.W, MatrixUtils.product(Wu, step));
	}

	public DoubleMatrix2D getW() {
		return W;
	}

	public DoubleMatrix1D getB1() {
		return this.B1;
	}

	public DoubleMatrix1D getB2() {
		return this.B2;
	}

}
