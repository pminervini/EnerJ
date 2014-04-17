package com.neuralnoise.enerj.energy.tensor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.neuralnoise.enerj.energy.tensor.util.RESCALParameters;
import com.neuralnoise.enerj.util.MatrixUtils;

public class RESCALEnergyModel extends AbstractEnergyTensorModel<RESCALParameters> {

	private static final Logger log = LoggerFactory.getLogger(RESCALEnergyModel.class);

	private final RESCALParameters params;

	public RESCALEnergyModel(RESCALParameters params) {
		this.params = params;
	}

	@Override
	public double evaluate(int s, int r, int o) {
		DoubleMatrix1D Es = params.getE(s), Eo = params.getE(o), bL = params.getbL(), bR = params.getbR();
		DoubleMatrix2D Wp = params.getW(r);
		return i(s(m(Wp, Es), bL), s(m(Wp, Eo), bR));
	}

	@Override
	public RESCALParameters grad(int s, int r, int o) {
		DoubleMatrix1D Es = params.getE(s), Eo = params.getE(o), bL = params.getbL(), bR = params.getbR();
		DoubleMatrix2D Wp = params.getW(r);
		
		final int N = params.getE().length, NR = params.getW().length;
		
		DoubleMatrix1D gEs = s(m(m(t(Wp), Wp), Eo), m(t(Wp), bR));
		DoubleMatrix1D gEo = s(m(m(t(Wp), Wp), Es), m(t(Wp), bL));
		
		DoubleMatrix1D gbL = s(m(Wp, Eo), bR);
		DoubleMatrix1D gbR = s(m(Wp, Es), bL);
		
		DoubleMatrix2D gWp = s(s(m(Wp, s(o(Es, Eo), o(Eo, Es))), o(bR, Es)), o(bL, Eo));
		
		DoubleMatrix2D[] gW = new DoubleMatrix2D[NR];
		gW[r] = gWp;
		
		DoubleMatrix1D[] gE = new DoubleMatrix1D[N];
		gE[s] = (gE[s] != null ? s(gE[s], gEs) : gEs);
		gE[o] = (gE[o] != null ? s(gE[o], gEo) : gEo);
		
		return new RESCALParameters(gE, gbL, gbR, gW);
	}
	
	public RESCALParameters getParams() {
		return params;
	}
	
	public static DoubleMatrix2D t(DoubleMatrix2D A) {
		return MatrixUtils.transpose(A);
	}
	
	public static DoubleMatrix2D m(DoubleMatrix2D A, DoubleMatrix2D B) {
		return MatrixUtils.product(A, B);
	}

	public static DoubleMatrix1D m(DoubleMatrix2D A, DoubleMatrix1D b) {
		return MatrixUtils.product(A, b);
	}
	
	public static DoubleMatrix1D s(DoubleMatrix1D a, DoubleMatrix1D b) {
		return MatrixUtils.sum(a, b);
	}
	
	public static DoubleMatrix2D s(DoubleMatrix2D A, DoubleMatrix2D B) {
		return MatrixUtils.sum(A, B);
	}

	public static double i(DoubleMatrix1D a, DoubleMatrix1D b) {
		return MatrixUtils.innerProduct(a, b);
	}
	
	public static DoubleMatrix2D o(DoubleMatrix1D a, DoubleMatrix1D b) {
		return MatrixUtils.outerProduct(a, b);
	}
	
}
