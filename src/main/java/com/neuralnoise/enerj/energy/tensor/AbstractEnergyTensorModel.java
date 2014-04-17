package com.neuralnoise.enerj.energy.tensor;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.neuralnoise.enerj.energy.AbstractEnergyModel;
import com.neuralnoise.enerj.energy.tensor.util.AbstractParameters;
import com.neuralnoise.enerj.util.MatrixUtils;

public abstract class AbstractEnergyTensorModel<T extends AbstractParameters> extends AbstractEnergyModel {

	public AbstractEnergyTensorModel() {
		super();
	}

	public abstract double evaluate(int s, int p, int o);
	
	public abstract T grad(int s, int r, int o);
	
	public static DoubleMatrix2D x3(DoubleMatrix2D[] T, DoubleMatrix1D v) {
		DoubleMatrix2D ret = null;
		for (int i = 0; i < T.length; ++i) {
			DoubleMatrix2D X = MatrixUtils.product(T[i], v.get(i));
			ret = (ret != null ? MatrixUtils.sum(ret, X) : X);
		}
		return ret;
	}
	
}
