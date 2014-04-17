package com.neuralnoise.enerj.dae.online;

import java.util.List;
import java.util.Vector;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.dae.util.Layer;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class MLAE extends AbstractAE {

	protected final Layer[] layers;

	public MLAE(final int N, List<Integer> hidden, AbstractActivationFunction activation, AbstractLossFunction loss, List<Pair<AbstractRegularizer, Double>> regularizers, DoubleRandomEngine prng) {

		super(activation, loss, regularizers, prng);

		this.layers = new Layer[hidden.size()];

		int n = N, c = 0;
		for (int m : hidden) {
			final double dn = n, dm = m;
			final double min = -4.0 * (6.0 / Math.sqrt(dn + dm));
			final double max = +4.0 * (6.0 / Math.sqrt(dn + dm));

			DoubleMatrix2D W = RandomUtils.randomMatrix(prng, m, n, min, max);

			DoubleMatrix1D b1 = RandomUtils.randomVector(prng, m, min, max);
			DoubleMatrix1D b2 = RandomUtils.randomVector(prng, n, min, max);

			Layer layer = new Layer(W, b1, b2);

			this.layers[c++] = layer;
			n = m;
		}
	}

	public DoubleMatrix1D W1(DoubleMatrix1D x, int l) {
		Layer layer = this.layers[l];
		DoubleMatrix1D y = MatrixUtils.sum(MatrixUtils.product(layer.getW(), x), layer.getB1());
		return y;
	}

	public DoubleMatrix1D f(DoubleMatrix1D x, int l) {
		return this.activation.f(W1(x, l));
	}

	@Override
	public DoubleMatrix1D f(DoubleMatrix1D x) {
		DoubleMatrix1D t = x;
		for (int l = 0; l < this.layers.length; ++l) {
			t = f(t, l);
		}
		return t;
	}

	public DoubleMatrix1D W2(DoubleMatrix1D y, int l) {
		Layer layer = this.layers[l];
		DoubleMatrix1D z = MatrixUtils.sum(MatrixUtils.product(MatrixUtils.transpose(layer.getW()), y), layer.getB2());
		return z;
	}

	public DoubleMatrix1D g(DoubleMatrix1D y, int l) {
		return this.activation.f(W2(y, l));
	}

	@Override
	public DoubleMatrix1D g(DoubleMatrix1D y) {
		DoubleMatrix1D t = y;
		for (int l = this.layers.length - 1; l >= 0; --l) {
			t = g(t, l);
		}
		return t;
	}

	@Override
	public double loss(DoubleMatrix1D x) {
		DoubleMatrix1D t = x;
		for (int l = 0; l < this.layers.length; ++l) {
			t = f(t, l);
		}
		for (int l = this.layers.length - 1; l >= 0; --l) {
			t = g(t, l);
		}
		DoubleMatrix1D z = t;
		DoubleMatrix1D L = this.loss.f(x, z);

		double r = 0.0;
		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			for (Layer layer : this.layers) {
				r += MatrixUtils.sum(MatrixUtils.product(reg.f(layer.getB1()), weight));
				r += MatrixUtils.sum(MatrixUtils.product(reg.f(layer.getB2()), weight));
				r += MatrixUtils.sum(MatrixUtils.product(reg.f(layer.getW()), weight));
			}
		}

		double ret = MatrixUtils.sum(L) + r;
		return ret;
	}

	@Override
	public void train(DoubleMatrix1D x, double p, double step) {
		DoubleMatrix1D tx = corrupt(x, p);

		final int L = this.layers.length, N = L * 2;

		Vector<DoubleMatrix1D> As = new Vector<DoubleMatrix1D>(N);
		As.setSize(N);

		Vector<DoubleMatrix1D> Zs = new Vector<DoubleMatrix1D>(N);
		Zs.setSize(N);

		DoubleMatrix1D tz = tx, ta = null;

		for (int i = 0; i < N; ++i) {
			int l = (i < L ? i : N - (i + 1));

			DoubleMatrix1D A = (i < L ? W1(tz, l) : W2(tz, l));
			DoubleMatrix1D z = this.activation.f(A);

			As.set(i, A);
			Zs.set(i, z);

			tz = z;
			ta = A;
		}

		DoubleMatrix1D dl = this.loss.df(x, tz), ds = this.activation.df(ta);

		Vector<DoubleMatrix1D> Ds = new Vector<DoubleMatrix1D>(N);
		Ds.setSize(N);

		for (int i = (N - 1); i >= 0; --i) {

			if (i == N - 1) {
				DoubleMatrix1D D = MatrixUtils.hadamardProduct(dl, ds);

				Ds.set(i, D);
			} else if (i < (L - 1)) {
				final int l = i;

				Layer layer = this.layers[l + 1];
				DoubleMatrix2D W = MatrixUtils.transpose(layer.getW());
				DoubleMatrix1D tD = Ds.get(i + 1), A = As.get(i);

				DoubleMatrix1D D = MatrixUtils.hadamardProduct(MatrixUtils.product(W, tD), this.activation.df(A));
				Ds.set(i, D);
			} else {
				final int l = N - (i + 1);

				Layer layer = this.layers[l - 1];
				DoubleMatrix2D W = layer.getW();
				DoubleMatrix1D tD = Ds.get(i + 1), A = As.get(i);

				DoubleMatrix1D D = MatrixUtils.hadamardProduct(MatrixUtils.product(W, tD), this.activation.df(A));
				Ds.set(i, D);
			}
		}

		Vector<Layer> ups = new Vector<Layer>(L);
		ups.setSize(L);

		for (int l = 0; l < L; ++l) {
			DoubleMatrix1D B1u = Ds.get(l), B2u = Ds.get(N - (l + 1));

			DoubleMatrix1D pZ = (l == 0 ? tx : Zs.get(l - 1));
			DoubleMatrix1D aZ = Zs.get(N - (l + 2));

			DoubleMatrix2D t1 = MatrixUtils.outerProduct(Ds.get(l), pZ);
			DoubleMatrix2D t2 = MatrixUtils.outerProduct(aZ, Ds.get(N - (l + 1)));
			DoubleMatrix2D Wu = MatrixUtils.sum(t1, t2);

			Layer u = new Layer(Wu, B1u, B2u);
			ups.set(l, u);
		}

		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			for (int i = 0; i < ups.size(); ++i) {
				Layer layer = this.layers[i], up = ups.get(i);

				DoubleMatrix1D B1u = MatrixUtils.sum(up.getB1(), MatrixUtils.product(reg.df(layer.getB1()), weight));
				DoubleMatrix1D B2u = MatrixUtils.sum(up.getB2(), MatrixUtils.product(reg.df(layer.getB2()), weight));

				DoubleMatrix2D Wu = MatrixUtils.sum(up.getW(), MatrixUtils.product(reg.df(layer.getW()), weight));

				Layer nup = new Layer(Wu, B1u, B2u);
				ups.set(i, nup);
			}
		}

		for (int l = 0; l < L; ++l) {
			Layer layer = this.layers[l], up = ups.get(l);

			DoubleMatrix1D B1 = MatrixUtils.subtraction(layer.getB1(), MatrixUtils.product(up.getB1(), step));
			DoubleMatrix1D B2 = MatrixUtils.subtraction(layer.getB2(), MatrixUtils.product(up.getB2(), step));
			DoubleMatrix2D W = MatrixUtils.subtraction(layer.getW(), MatrixUtils.product(up.getW(), step));

			Layer nlayer = new Layer(W, B1, B2);
			this.layers[l] = nlayer;
		}
	}

}
