package com.neuralnoise.enerj.energy.tensor.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class BilinearParameters extends AbstractParameters {

	private static final Logger log = LoggerFactory.getLogger(BilinearParameters.class);
	
	private final DoubleMatrix1D bL, bR, E[];
	private final DoubleMatrix2D[] wL, wR;

	@Override
	public AbstractParameters sum(AbstractParameters _pa, AbstractParameters _pb, double alpha) {
		BilinearParameters pa = (BilinearParameters) _pa, pb = (BilinearParameters) _pb;
		final int d = pa.getwL().length, n = pa.getE().length;
		DoubleMatrix1D[] E = new DoubleMatrix1D[n];
		for (int i = 0; i < n; ++i) {
			//E[i] = (pb.getE(i) != null ? MatrixUtils.sum(pa.getE(i), MatrixUtils.product(pb.getE(i), alpha)) : pa.getE(i));
			DoubleMatrix1D Ea = pa.getE(i), Eb = pb.getE(i);
			E[i] = (Ea != null && Eb != null ? MatrixUtils.sum(Ea, MatrixUtils.product(Eb, alpha)) : (Ea != null ? Ea : Eb));
		}
		DoubleMatrix1D bL = MatrixUtils.sum(pa.getbL(), MatrixUtils.product(pb.getbL(), alpha));
		DoubleMatrix1D bR = MatrixUtils.sum(pa.getbR(), MatrixUtils.product(pb.getbR(), alpha));
		DoubleMatrix2D[] wL = new DoubleMatrix2D[d], wR = new DoubleMatrix2D[d];
		for (int i = 0; i < d; ++i) {
			wL[i] = MatrixUtils.sum(pa.getwL(i), MatrixUtils.product(pb.getwL(i), alpha));
			wR[i] = MatrixUtils.sum(pa.getwR(i), MatrixUtils.product(pb.getwR(i), alpha));
		}
		return new BilinearParameters(E, wL, wR, bL, bR);
	}
	
	public BilinearParameters(DoubleMatrix1D[] E,
			DoubleMatrix2D[] wL, DoubleMatrix2D[] wR,
			DoubleMatrix1D bL, DoubleMatrix1D bR) {
		this.E = E;
		this.wL = wL;
		this.wR = wR;
		this.bL = bL;
		this.bR = bR;
	}
	
	public BilinearParameters(final int n, final int p, final int d, DoubleRandomEngine prng) {
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

	public DoubleMatrix1D getbL() {
		return bL;
	}

	public DoubleMatrix1D getbR() {
		return bR;
	}

	public DoubleMatrix1D[] getE() {
		return E;
	}
	
	public DoubleMatrix1D getE(int i) {
		return E[i];
	}

	public DoubleMatrix2D[] getwL() {
		return wL;
	}

	public DoubleMatrix2D getwL(int i) {
		return wL[i];
	}
	
	public DoubleMatrix2D[] getwR() {
		return wR;
	}
	
	public DoubleMatrix2D getwR(int i) {
		return wR[i];
	}
	
}
