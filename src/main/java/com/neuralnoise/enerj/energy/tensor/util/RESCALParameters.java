package com.neuralnoise.enerj.energy.tensor.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class RESCALParameters extends AbstractParameters {

	private static final Logger log = LoggerFactory.getLogger(RESCALParameters.class);
	
	private final DoubleMatrix1D bL, bR, E[];
	private final DoubleMatrix2D W[];
	
	public RESCALParameters(DoubleMatrix1D E[], DoubleMatrix1D bL, DoubleMatrix1D bR,
			DoubleMatrix2D W[]) {
		this.E = E;
		this.bL = bL;
		this.bR = bR;
		this.W = W;
	}

	public RESCALParameters(final int N, final int NR, final int p, final int d, DoubleRandomEngine prng) {
		super();

		final double min = -4.0 * (6.0 / Math.sqrt(N));
		final double max = +4.0 * (6.0 / Math.sqrt(N));

		this.bL = RandomUtils.randomVector(prng, p, min, max);
		this.bR = RandomUtils.randomVector(prng, p, min, max);
		
		this.E = new DoubleMatrix1D[N];
		for (int i = 0; i < N; ++i) {
			this.E[i] = RandomUtils.randomVector(prng, d, min, max);
		}
		
		this.W = new DoubleMatrix2D[NR];
		for (int i = 0; i < NR; ++i) {
			this.W[i] = RandomUtils.randomMatrix(prng, p, d, min, max);
		}
	}
	
	@Override
	public AbstractParameters sum(AbstractParameters _pa, AbstractParameters _pb, double alpha) {
		RESCALParameters pa = (RESCALParameters) _pa, pb = (RESCALParameters) _pb;
		final int R = pa.getW().length, n = pa.getE().length;

		DoubleMatrix1D[] E = new DoubleMatrix1D[n];
		
		for (int i = 0; i < n; ++i) {
			DoubleMatrix1D Ea = pa.getE(i), Eb = pb.getE(i);
			E[i] = (Ea != null && Eb != null ? MatrixUtils.sum(Ea, MatrixUtils.product(Eb, alpha)) : (Ea != null ? Ea : Eb));
		}

		DoubleMatrix1D bL = MatrixUtils.sum(pa.getbL(), MatrixUtils.product(pb.getbL(), alpha));
		DoubleMatrix1D bR = MatrixUtils.sum(pa.getbR(), MatrixUtils.product(pb.getbR(), alpha));
		
		DoubleMatrix2D[] W = new DoubleMatrix2D[R];
		
		for (int i = 0; i < R; ++i) {
			DoubleMatrix2D Wa = pa.getW(i), Wb = pb.getW(i);
			W[i] = (Wa != null && Wb != null ? MatrixUtils.sum(Wa, MatrixUtils.product(Wb, alpha)) : (Wa != null ? Wa : Wb));
		}
		
		return new RESCALParameters(E, bL, bR, W);
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

	public DoubleMatrix2D[] getW() {
		return W;
	}

	public DoubleMatrix2D getW(int i) {
		return W[i];
	}
	
}
