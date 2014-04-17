package com.neuralnoise.enerj.energy.tensor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;

import com.google.common.primitives.Ints;
import com.neuralnoise.enerj.energy.tensor.util.BilinearParameters;
import com.neuralnoise.enerj.util.MatrixUtils;

public class BilinearEnergyModel extends AbstractEnergyTensorModel<BilinearParameters> {

	private static final Logger log = LoggerFactory.getLogger(BilinearEnergyModel.class);

	private final BilinearParameters params;

	public BilinearEnergyModel(BilinearParameters params) {
		this.params = params;
	}

	@Override
	public BilinearParameters grad(int s, int r, int o) {
		DoubleMatrix1D Es = params.getE(s), Ep = params.getE(r), Eo = params.getE(o);
		
		DoubleMatrix2D W1 = x3(params.getwL(), Ep), W2 = x3(params.getwR(), Ep), W1T = t(W1), W2T = t(W2);
		
		DoubleMatrix1D gbL = s(m(W2, Eo), params.getbR());
		DoubleMatrix1D gbR = s(m(W1, Es), params.getbL());
	
		DoubleMatrix1D gEs = s(m(m(W1T, W2), Eo), m(W1T, params.getbR()));
		DoubleMatrix1D gEo = s(m(m(W2T, W1), Es), m(W2T, params.getbL()));
		
		final int d = Ints.checkedCast(gEs.size()), n = params.getE().length;
		
		DoubleMatrix2D[] gWr = new DoubleMatrix2D[d], gWl = new DoubleMatrix2D[d];
		
		DoubleMatrix1D gEp = new DenseDoubleMatrix1D(params.getwL().length);
		for (int i = 0; i < d; ++i) {
			DoubleMatrix1D wl = m(params.getwL(i), Es), wr = m(params.getwR(i), Eo);

			double gEpi = i(params.getbL(), wr) + i(wl, params.getbR()) + (2.0 * Ep.get(i) * i(wl, wr));
			
			for (int j = 0; j < d; ++j) {
				if (j != i) {
					DoubleMatrix1D wlt = m(params.getwL(j), Es), wrt = m(params.getwR(j), Eo);
					gEpi += Ep.get(j) * (i(wlt, wr) + i(wl, wrt));
				}
			}
			
			gEp.set(i, gEpi);
			
			DoubleMatrix2D gWli = o(params.getbR(), Es);
			DoubleMatrix2D gWri = o(params.getbL(), Eo);
			
			for (int k = 0; k < d; ++k) {
				DoubleMatrix1D wrk = m(params.getwR(k), Eo);
				gWli = s(gWli, m(o(wrk, Es), Ep.get(k)));
				
				DoubleMatrix1D wlk = m(params.getwL(k), Es);
				gWri = s(gWri, m(o(wlk, Eo), Ep.get(k)));
			}
			
			gWli = m(gWli, Ep.get(i));
			gWri = m(gWri, Ep.get(i));
			
			gWr[i] = gWri;
			gWl[i] = gWli;
		}
		
		DoubleMatrix1D[] gE = new DoubleMatrix1D[n];
		gE[s] = (gE[s] != null ? s(gE[s], gEs) : gEs);
		gE[r] = (gE[r] != null ? s(gE[r], gEp) : gEp);
		gE[o] = (gE[o] != null ? s(gE[o], gEo) : gEo);
		return new BilinearParameters(gE, gWl, gWr, gbL, gbR);
	}
	
	@Override
	public double evaluate(int s, int p, int o) {
		DoubleMatrix1D Es = params.getE(s), Ep = params.getE(p), Eo = params.getE(o);
		DoubleMatrix1D w1 = s(m(x3(params.getwL(), Ep), Es), params.getbL());		
		DoubleMatrix1D w2 = s(m(x3(params.getwR(), Ep), Eo), params.getbR());
		return i(w1, w2);
	}
	
	public BilinearParameters getParams() {
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
	
	public static DoubleMatrix2D m(DoubleMatrix2D A, double b) {
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
