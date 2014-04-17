package com.neuralnoise.enerj.energy.tensor.play;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class BilinearEnergyModel {

	private static final Logger log = LoggerFactory.getLogger(BilinearEnergyModel.class);
	
	public DoubleMatrix1D bL, bR, E[];
	public DoubleMatrix2D[] wL, wR;

	public BilinearEnergyModel(final int n, final int p, final int d, DoubleRandomEngine prng) {
		super();
		
		final double min = - 4.0 * (6.0 / Math.sqrt(n));
		final double max = + 4.0 * (6.0 / Math.sqrt(n));
		
		this.bL = RandomUtils.randomVector(prng, p, min, max);
		this.bR = RandomUtils.randomVector(prng, p, min, max);
		this.E = new DoubleMatrix1D[n];
		this.wL = new DoubleMatrix2D[d];
		this.wR = new DoubleMatrix2D[d];
		for (int i = 0; i < n; ++i) {
			this.E[i] = RandomUtils.randomVector(prng, d, min, max);
		}
		for (int i = 0; i < d; ++i) {
			this.wL[i] = RandomUtils.randomMatrix(prng, p, d, min, max);
			this.wR[i] = RandomUtils.randomMatrix(prng, p, d, min, max);
		}
	}
	
	public double evaluate(int s, int p, int o) {
		DoubleMatrix1D Es = this.E[s], Ep = this.E[p], Eo = this.E[o];
		DoubleMatrix1D w1 = MatrixUtils.sum(MatrixUtils.product(x3(this.wL, Ep), Es), this.bL);		
		DoubleMatrix1D w2 = MatrixUtils.sum(MatrixUtils.product(x3(this.wR, Ep), Eo), this.bR);
		return MatrixUtils.innerProduct(w1, w2);
	}

	public DoubleMatrix1D gbL(int s, int p, int o) {
		DoubleMatrix1D Ep = this.E[p], Eo = this.E[o];
		return MatrixUtils.sum(MatrixUtils.product(x3(this.wR, Ep), Eo), this.bR);
	}
	
	public DoubleMatrix1D gbR(int s, int p, int o) {
		DoubleMatrix1D Ep = this.E[p], Es = this.E[s];
		return MatrixUtils.sum(MatrixUtils.product(x3(this.wL, Ep), Es), this.bL);
	}
	
	public DoubleMatrix1D gEs(int s, int p, int o) {
		DoubleMatrix1D Eo = this.E[o], Ep = this.E[p];
		DoubleMatrix2D W1 = x3(this.wL, Ep), W2 = x3(this.wR, Ep);
		DoubleMatrix1D gEs = MatrixUtils.product(MatrixUtils.product(MatrixUtils.transpose(W1), W2), Eo);
		return MatrixUtils.sum(gEs, MatrixUtils.product(MatrixUtils.transpose(W1), this.bR));
	}
	
	public DoubleMatrix1D gEo(int s, int p, int o) {
		DoubleMatrix1D Es = this.E[s], Ep = this.E[p];
		DoubleMatrix2D W1 = x3(this.wL, Ep), W2 = x3(this.wR, Ep);
		DoubleMatrix1D gEo = MatrixUtils.product(MatrixUtils.product(MatrixUtils.transpose(W2), W1), Es);
		return MatrixUtils.sum(gEo, MatrixUtils.product(MatrixUtils.transpose(W2), this.bL));
	}
	
	public DoubleMatrix1D gEp(int s, int p, int o) {
		DoubleMatrix1D Es = this.E[s], Ep = this.E[p], Eo = this.E[o];
		DoubleMatrix1D gEp = new DenseDoubleMatrix1D(this.wL.length);
		for (int i = 0; i < gEp.size(); ++i) {
			DoubleMatrix1D wl = MatrixUtils.product(this.wL[i], Es), wr = MatrixUtils.product(this.wR[i], Eo);
			double gEpi = MatrixUtils.innerProduct(this.bL, wr)
					+ MatrixUtils.innerProduct(wl, this.bR)
					+ (2.0 * Ep.get(i) * MatrixUtils.innerProduct(wl, wr));
			for (int j = 0; j < gEp.size(); ++j) {
				if (j != i) {
					DoubleMatrix1D wlt = MatrixUtils.product(this.wL[j], Es), wrt = MatrixUtils.product(this.wR[j], Eo);
					gEpi += Ep.get(j) * (MatrixUtils.innerProduct(wlt, wr) + MatrixUtils.innerProduct(wl, wrt));
				}
			}
			gEp.set(i, gEpi);
		}
		return gEp;
	}
	
	public DoubleMatrix2D gWl(int s, int p, int o, int i) {
		DoubleMatrix1D Es = this.E[s], Ep = this.E[p], Eo = this.E[o];
		DoubleMatrix2D gWl = MatrixUtils.outerProduct(this.bR, Es);
		for (int j = 0; j < this.wR.length; ++j) {
			DoubleMatrix1D wr = MatrixUtils.product(this.wR[j], Eo);
			gWl = MatrixUtils.sum(gWl, MatrixUtils.product(MatrixUtils.outerProduct(wr, Es), Ep.get(j)));
		}
		return MatrixUtils.product(gWl, Ep.get(i));
	}
	
	public DoubleMatrix2D gWr(int s, int p, int o, int i) {
		DoubleMatrix1D Es = this.E[s], Ep = this.E[p], Eo = this.E[o];
		DoubleMatrix2D gWr = MatrixUtils.outerProduct(this.bL, Eo);
		for (int j = 0; j < this.wL.length; ++j) {
			DoubleMatrix1D wl = MatrixUtils.product(this.wL[j], Es);
			gWr = MatrixUtils.sum(gWr, MatrixUtils.product(MatrixUtils.outerProduct(wl, Eo), Ep.get(j)));
		}
		return MatrixUtils.product(gWr, Ep.get(i));
	}
	
	public static DoubleMatrix2D x3(DoubleMatrix2D[] T, DoubleMatrix1D v) {
		DoubleMatrix2D ret = null;
		for (int i = 0; i < T.length; ++i) {
			DoubleMatrix2D X = MatrixUtils.product(T[i], v.get(i));
			ret = (ret != null ? MatrixUtils.sum(ret, X) : X);
		}
		return ret;
	}
	
}
