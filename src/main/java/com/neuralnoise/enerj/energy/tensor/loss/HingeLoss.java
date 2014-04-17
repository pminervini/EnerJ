package com.neuralnoise.enerj.energy.tensor.loss;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.neuralnoise.enerj.energy.tensor.AbstractEnergyTensorModel;
import com.neuralnoise.enerj.energy.tensor.util.AbstractParameters;

public class HingeLoss<T extends AbstractEnergyTensorModel<?>> extends AbstractLossFunction<T> {

	private static final Logger log = LoggerFactory.getLogger(HingeLoss.class);
	
	@Override
	public double f(T m, int ps, int pp, int po, int ns, int np, int no) {
		final double pe = m.evaluate(ps, pp, po), ne = m.evaluate(ns, np, no); 
		return (pe > ne - 1.0 ? pe - ne + 1 : 0.0);
	}

	@Override
	public AbstractParameters df(T m, int ps, int pp, int po, int ns, int np, int no) {
		final double pe = m.evaluate(ps, pp, po), ne = m.evaluate(ns, np, no);
		AbstractParameters df = null;
		if (pe > ne - 1.0) {
			AbstractParameters dpe = m.grad(ps, pp, po), dne = m.grad(ns, np, no);
			//log.info("dpe: " + dpe + ", dne: " + dne);
			df = dpe.sum(dpe, dne, - 1.0);
		}
		return df;
	}

	
	
}
