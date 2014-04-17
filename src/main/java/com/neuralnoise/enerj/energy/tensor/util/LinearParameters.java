package com.neuralnoise.enerj.energy.tensor.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class LinearParameters extends AbstractParameters {

	private static final Logger log = LoggerFactory.getLogger(LinearParameters.class);
	
	private final DoubleMatrix1D bL, bR, E[];
	private final DoubleMatrix2D wL1, wL2, wR1, wR2;
	
	@Override
	public AbstractParameters sum(AbstractParameters _pa, AbstractParameters _pb, double alpha) {
		LinearParameters pa = (LinearParameters) _pa, pb = (LinearParameters) _pb;
		final int d = pa.getwL1().columns(), n = pa.getE().length;
		DoubleMatrix1D[] E = new DoubleMatrix1D[n];
		
		for (int i = 0; i < n; ++i) {
			DoubleMatrix1D Ea = pa.getE(i), Eb = pb.getE(i);
			E[i] = (Ea != null && Eb != null ? MatrixUtils.sum(Ea, MatrixUtils.product(Eb, alpha)) : (Ea != null ? Ea : Eb));
			//E[i] = pa.getE(i);
		}
		DoubleMatrix1D bL = MatrixUtils.sum(pa.getbL(), MatrixUtils.product(pb.getbL(), alpha));
		DoubleMatrix1D bR = MatrixUtils.sum(pa.getbR(), MatrixUtils.product(pb.getbR(), alpha));
		DoubleMatrix2D WL1 = MatrixUtils.sum(pa.getwL1(), MatrixUtils.product(pb.getwL1(), alpha));
		DoubleMatrix2D WL2 = MatrixUtils.sum(pa.getwL2(), MatrixUtils.product(pb.getwL2(), alpha));
		DoubleMatrix2D WR1 = MatrixUtils.sum(pa.getwR1(), MatrixUtils.product(pb.getwR1(), alpha));
		DoubleMatrix2D WR2 = MatrixUtils.sum(pa.getwR2(), MatrixUtils.product(pb.getwR2(), alpha));
		
		//bL = pa.bL;
		//bR = pa.bR;
		//WL1 = pa.wL1;
		//WL2 = pa.wL2;
		//WR1 = pa.wR1;
		//WR2 = pa.wR2;
		
		return new LinearParameters(E, bL, bR, WL1, WL2, WR1, WR2);
	}
	
	public LinearParameters(DoubleMatrix1D E[], DoubleMatrix1D bL, DoubleMatrix1D bR,
			DoubleMatrix2D wL1, DoubleMatrix2D wL2, DoubleMatrix2D wR1, DoubleMatrix2D wR2) {
		this.E = E;
		this.bL = bL;
		this.bR = bR;
		this.wL1 = wL1;
		this.wL2 = wL2;
		this.wR1 = wR1;
		this.wR2 = wR2;
	}

	public LinearParameters(final int n, final int p, final int d, DoubleRandomEngine prng) {
		super();
		
		final double min = - 4.0 * (6.0 / Math.sqrt(n));
		final double max = + 4.0 * (6.0 / Math.sqrt(n));
		
		this.bL = RandomUtils.randomVector(prng, p, min, max);
		this.bR = RandomUtils.randomVector(prng, p, min, max);
		this.E = new DoubleMatrix1D[n];
		for (int i = 0; i < n; ++i) {
			this.E[i] = RandomUtils.randomVector(prng, d, min, max);
		}
		this.wL1 = RandomUtils.randomMatrix(prng, p, d, min, max);
		this.wL2 = RandomUtils.randomMatrix(prng, p, d, min, max);
		this.wR1 = RandomUtils.randomMatrix(prng, p, d, min, max);
		this.wR2 = RandomUtils.randomMatrix(prng, p, d, min, max);
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
	
	public DoubleMatrix2D getwL1() {
		return wL1;
	}

	public DoubleMatrix2D getwL2() {
		return wL2;
	}

	public DoubleMatrix2D getwR1() {
		return wR1;
	}

	public DoubleMatrix2D getwR2() {
		return wR2;
	}
	
}
