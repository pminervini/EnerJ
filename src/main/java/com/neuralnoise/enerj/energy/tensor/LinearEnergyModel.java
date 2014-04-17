package com.neuralnoise.enerj.energy.tensor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.neuralnoise.enerj.energy.tensor.util.LinearParameters;
import com.neuralnoise.enerj.util.MatrixUtils;

public class LinearEnergyModel extends AbstractEnergyTensorModel<LinearParameters> {

	private static final Logger log = LoggerFactory.getLogger(LinearEnergyModel.class);

	private final LinearParameters params;

	public LinearEnergyModel(LinearParameters params) {
		this.params = params;
	}
	
	@Override
	public LinearParameters grad(int s, int r, int o) {
		DoubleMatrix1D Es = params.getE(s), Ep = params.getE(r), Eo = params.getE(o), bL = params.getbL(), bR = params.getbR();
		
		DoubleMatrix2D WL1 = params.getwL1(), WL2 = params.getwL2(), WR1 = params.getwR1(), WR2 = params.getwR2();
		
		final int N = params.getE().length;
		
		DoubleMatrix1D gEs = s(s(m(m(t(WL1), WR1), Eo), m(m(t(WL1), WR2), Ep)), m(t(WL1), bR));
		DoubleMatrix1D gEo = s(s(m(m(t(WR1), WL1), Es), m(m(t(WR1), WL2), Ep)), m(t(WR1), bL));
		DoubleMatrix1D gEp = s(s(s(s(s(m(m(t(WR2), WL1), Es), m(m(t(WL2), WR1), Eo)), m(m(t(WL2), WR2), Ep)), m(m(t(WR2), WL2), Ep)), m(t(WL2), bR)), m(t(WR2), bL));
		DoubleMatrix1D gBL = s(s(m(WR1, Eo), m(WR2, Ep)), bR);
		DoubleMatrix1D gBR = s(s(m(WL1, Es), m(WL2, Ep)), bL);
		
		// W is p x d, E is d, B is p
		DoubleMatrix2D gWL1 = s(s(o(m(WR1, Eo), Es), o(m(WR2, Ep), Es)), o(bR, Es)); // p x d
		DoubleMatrix2D gWR1 = s(s(o(m(WL1, Es), Eo), o(m(WL2, Ep), Eo)), o(bL, Eo));
		DoubleMatrix2D gWL2 = s(s(o(m(WR1, Eo), Ep), o(m(WR2, Ep), Ep)), o(bR, Ep));
		DoubleMatrix2D gWR2 = s(s(o(m(WL1, Es), Ep), o(m(WL2, Ep), Ep)), o(bL, Ep));

		DoubleMatrix1D[] gE = new DoubleMatrix1D[N];
		gE[s] = (gE[s] != null ? s(gE[s], gEs) : gEs);
		gE[r] = (gE[r] != null ? s(gE[r], gEp) : gEp);
		gE[o] = (gE[o] != null ? s(gE[o], gEo) : gEo);
		
		return new LinearParameters(gE, gBL, gBR, gWL1, gWL2, gWR1, gWR2);
	}
	
	@Override
	public double evaluate(int s, int r, int o) {
		DoubleMatrix1D Es = params.getE(s), Ep = params.getE(r), Eo = params.getE(o);
		DoubleMatrix1D L = s(s(m(params.getwL1(), Es), m(params.getwL2(), Ep)), params.getbL());
		DoubleMatrix1D R = s(s(m(params.getwR1(), Eo), m(params.getwR2(), Ep)), params.getbR());
		return i(L, R);
	}

	public LinearParameters getParams() {
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
	
	public static DoubleMatrix2D o(DoubleMatrix1D a, DoubleMatrix1D b) {
		return MatrixUtils.outerProduct(a, b);
	}
	
	public static double i(DoubleMatrix1D a, DoubleMatrix1D b) {
		return MatrixUtils.innerProduct(a, b);
	}
	
}
