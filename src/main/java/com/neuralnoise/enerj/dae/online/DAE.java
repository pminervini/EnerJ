package com.neuralnoise.enerj.dae.online;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class DAE extends AbstractAE {

	private DoubleMatrix2D W;
	private DoubleMatrix1D b1, b2;

	public DAE(final int N, final int M, AbstractActivationFunction activation, AbstractLossFunction loss,
			List<Pair<AbstractRegularizer, Double>> regularizers, DoubleRandomEngine prng) {

		super(activation, loss, regularizers, prng);

		final int n = N, m = M;

		final double min = -4.0 * (6.0 / Math.sqrt(n + m));
		final double max = +4.0 * (6.0 / Math.sqrt(n + m));

		this.W = RandomUtils.randomMatrix(prng, m, n, min, max);
		this.b1 = RandomUtils.randomVector(prng, m, min, max);
		this.b2 = RandomUtils.randomVector(prng, n, min, max);
	}

	public DoubleMatrix1D w1(DoubleMatrix1D x) {
		DoubleMatrix1D y = MatrixUtils.sum(MatrixUtils.product(this.W, x), this.b1);
		return y;
	}

	@Override
	public DoubleMatrix1D f(DoubleMatrix1D x) {
		return activation.f(w1(x));
	}

	public DoubleMatrix1D w2(DoubleMatrix1D y) {
		DoubleMatrix1D z = MatrixUtils.sum(MatrixUtils.product(MatrixUtils.transpose(W), y), b2);
		return z;
	}

	@Override
	public DoubleMatrix1D g(DoubleMatrix1D y) {
		return activation.f(w2(y));
	}

	@Override
	public double loss(DoubleMatrix1D x) {
		DoubleMatrix1D z = g(f(x));
		DoubleMatrix1D L = loss.f(x, z);

		double r = 0.0;
		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.b1), weight));
			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.b2), weight));
			r += MatrixUtils.sum(MatrixUtils.product(reg.f(this.W), weight));
		}

		double ret = MatrixUtils.sum(L) + r;
		return ret;
	}

	@Override
	public void train(DoubleMatrix1D x, double p, double step) {
		DoubleMatrix1D tx = corrupt(x, p);

		DoubleMatrix1D A1 = w1(tx), y = activation.f(A1);
		DoubleMatrix1D A2 = w2(y), z = activation.f(A2);

		DoubleMatrix1D dl = this.loss.df(x, z), ds = this.activation.df(A2);

		DoubleMatrix1D d2 = MatrixUtils.hadamardProduct(dl, ds);
		DoubleMatrix1D d1 = MatrixUtils.hadamardProduct(MatrixUtils.product(this.W, d2), this.activation.df(A1));

		DoubleMatrix1D b1u = d1;
		DoubleMatrix1D b2u = d2;

		DoubleMatrix2D t1 = MatrixUtils.outerProduct(d1, tx);
		DoubleMatrix2D t2 = MatrixUtils.outerProduct(y, d2);
		DoubleMatrix2D T = MatrixUtils.sum(t1, t2);

		DoubleMatrix2D Wu = T;

		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			b1u = MatrixUtils.sum(b1u, MatrixUtils.product(reg.df(this.b1), weight));
			b2u = MatrixUtils.sum(b2u, MatrixUtils.product(reg.df(this.b2), weight));
			Wu = MatrixUtils.sum(Wu, MatrixUtils.product(reg.df(this.W), weight));
		}

		this.b1 = MatrixUtils.subtraction(this.b1, MatrixUtils.product(b1u, step));
		this.b2 = MatrixUtils.subtraction(this.b2, MatrixUtils.product(b2u, step));
		this.W = MatrixUtils.subtraction(this.W, MatrixUtils.product(Wu, step));
	}

	public DoubleMatrix2D getW() {
		return W;
	}

	public DoubleMatrix1D getB1() {
		return this.b1;
	}

	public DoubleMatrix1D getB2() {
		return this.b2;
	}

}
