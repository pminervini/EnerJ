package com.neuralnoise.enerj.energy.tensor.loss;

import com.neuralnoise.enerj.energy.tensor.AbstractEnergyTensorModel;
import com.neuralnoise.enerj.energy.tensor.util.AbstractParameters;

public abstract class AbstractLossFunction<T extends AbstractEnergyTensorModel<?>> {

	public abstract double f(T m, int ps, int pp, int po, int ns, int np, int no);
	
	public abstract AbstractParameters df(T m, int ps, int pp, int po, int ns, int np, int no);
	
}
