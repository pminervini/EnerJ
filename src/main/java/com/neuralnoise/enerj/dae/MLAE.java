package com.neuralnoise.enerj.dae;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.math.tdouble.DoubleFunctions;
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

	public DoubleMatrix2D W1(DoubleMatrix2D X, int l) {
		Layer layer = this.layers[l];
		// m x t
		DoubleMatrix2D Y = MatrixUtils.product(layer.getW(), X);
		for (int c = 0; c < Y.columns(); ++c) {
			Y.viewColumn(c).assign(layer.getB1(), DoubleFunctions.plus);
		}
		return Y;
	}

	public DoubleMatrix2D f(DoubleMatrix2D X, int l) {
		return this.activation.f(W1(X, l));
	}

	@Override
	public DoubleMatrix2D f(DoubleMatrix2D X) {
		DoubleMatrix2D t = X;
		for (int l = 0; l < this.layers.length; ++l) {
			t = f(t, l);
		}
		return t;
	}

	public DoubleMatrix2D W2(DoubleMatrix2D Y, int l) {
		Layer layer = this.layers[l];
		DoubleMatrix2D Z = MatrixUtils.product(MatrixUtils.transpose(layer.getW()), Y);
		for (int c = 0; c < Z.columns(); ++c) {
			Z.viewColumn(c).assign(layer.getB2(), DoubleFunctions.plus);
		}
		return Z;
	}

	public DoubleMatrix2D g(DoubleMatrix2D Y, int l) {
		return this.activation.f(W2(Y, l));
	}

	@Override
	public DoubleMatrix2D g(DoubleMatrix2D Y) {
		DoubleMatrix2D t = Y;
		for (int l = this.layers.length - 1; l >= 0; --l) {
			t = g(t, l);
		}
		return t;
	}

	@Override
	public void train(DoubleMatrix2D X, double p, double step) {
		DoubleMatrix2D tX = corrupt(X, p);

		final int L = this.layers.length, N = L * 2;

		DoubleMatrix2D As[] = new DoubleMatrix2D[N];
		DoubleMatrix2D Zs[] = new DoubleMatrix2D[N];

		DoubleMatrix2D tZ = tX, tA = null;

		for (int i = 0; i < N; ++i) {
			int l = (i < L ? i : N - (i + 1));

			DoubleMatrix2D A = (i < L ? W1(tZ, l) : W2(tZ, l));
			DoubleMatrix2D Z = this.activation.f(A);

			As[i] = A;
			Zs[i] = Z;

			tZ = Z;
			tA = A;
		}

		DoubleMatrix2D dl = this.loss.df(X, tZ), ds = this.activation.df(tA);

		// XXX
		double dt = X.columns();
		dl = MatrixUtils.product(dl, 1.0 / dt);
		
		DoubleMatrix2D Ds[] = new DoubleMatrix2D[N];

		for (int i = (N - 1); i >= 0; --i) {

			if (i == N - 1) {
				DoubleMatrix2D D = MatrixUtils.hadamardProduct(dl, ds);
				Ds[i] = D;
			} else if (i < (L - 1)) {
				final int l = i;

				Layer layer = this.layers[l + 1];
				DoubleMatrix2D Wt = MatrixUtils.transpose(layer.getW());
				DoubleMatrix2D tD = Ds[i + 1];
				DoubleMatrix2D A = As[i];

				DoubleMatrix2D D = MatrixUtils.hadamardProduct(MatrixUtils.product(Wt, tD), this.activation.df(A));
				Ds[i] = D;
			} else {
				final int l = N - (i + 1);

				Layer layer = this.layers[l - 1];
				DoubleMatrix2D W = layer.getW();
				DoubleMatrix2D tD = Ds[i + 1];
				DoubleMatrix2D A = As[i];

				DoubleMatrix2D D = MatrixUtils.hadamardProduct(MatrixUtils.product(W, tD), this.activation.df(A));
				Ds[i] = D;
			}
		}

		Layer[] ups = new Layer[L];

		for (int l = 0; l < L; ++l) {
			DoubleMatrix2D _B1u = Ds[l];
			DoubleMatrix2D _B2u = Ds[N - (l + 1)];

			// B1u: m
			DoubleMatrix1D B1u = new DenseDoubleMatrix1D(_B1u.rows());
			for (int i = 0; i < _B1u.columns(); ++i) {
				B1u = MatrixUtils.sum(B1u, _B1u.viewColumn(i));
			}

			// B2u: n
			DoubleMatrix1D B2u = new DenseDoubleMatrix1D(_B2u.rows());
			for (int i = 0; i < _B2u.columns(); ++i) {
				B2u = MatrixUtils.sum(B2u, _B2u.viewColumn(i));
			}

			DoubleMatrix2D pZ = (l == 0 ? tX : Zs[l - 1]);
			DoubleMatrix2D aZ = Zs[N - (l + 2)];

			DoubleMatrix2D t1 = MatrixUtils.product(Ds[l], MatrixUtils.transpose(pZ));

			DoubleMatrix2D t2 = MatrixUtils.product(aZ, MatrixUtils.transpose(Ds[N - (l + 1)]));

			DoubleMatrix2D Wu = MatrixUtils.sum(t1, t2);

			Layer u = new Layer(Wu, B1u, B2u);
			// ups.set(l, u);
			ups[l] = u;
		}

		for (Pair<AbstractRegularizer, Double> pair : this.regularizers) {
			AbstractRegularizer reg = pair.getKey();
			Double weight = pair.getValue();

			for (int i = 0; i < ups.length /* ups.size() */; ++i) {
				Layer layer = this.layers[i], up = ups[i]; // ups.get(i);

				DoubleMatrix1D B1u = MatrixUtils.sum(up.getB1(), MatrixUtils.product(reg.df(layer.getB1()), weight));
				DoubleMatrix1D B2u = MatrixUtils.sum(up.getB2(), MatrixUtils.product(reg.df(layer.getB2()), weight));

				DoubleMatrix2D Wu = MatrixUtils.sum(up.getW(), MatrixUtils.product(reg.df(layer.getW()), weight));

				Layer nup = new Layer(Wu, B1u, B2u);
				ups[i] = nup; // ups.set(i, nup);
			}
		}

		for (int l = 0; l < L; ++l) {
			Layer layer = this.layers[l], up = ups[l]; // ups.get(l);

			DoubleMatrix1D B1 = MatrixUtils.subtraction(layer.getB1(), MatrixUtils.product(up.getB1(), step));
			DoubleMatrix1D B2 = MatrixUtils.subtraction(layer.getB2(), MatrixUtils.product(up.getB2(), step));
			DoubleMatrix2D W = MatrixUtils.subtraction(layer.getW(), MatrixUtils.product(up.getW(), step));

			Layer nlayer = new Layer(W, B1, B2);
			this.layers[l] = nlayer;
		}
	}

	@Override
	public double loss(DoubleMatrix2D X) {
		DoubleMatrix2D t = X;
		for (int l = 0; l < this.layers.length; ++l) {
			t = f(t, l);
		}
		for (int l = this.layers.length - 1; l >= 0; --l) {
			t = g(t, l);
		}
		DoubleMatrix2D Z = t;
		DoubleMatrix2D L = this.loss.f(X, Z);

		// XXX
		double dt = X.columns();
		L = MatrixUtils.product(L, 1.0 / dt);
		
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
	
	public Layer[] getLayers() {
		return layers;
	}

}
