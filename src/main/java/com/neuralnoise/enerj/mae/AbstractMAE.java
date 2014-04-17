package com.neuralnoise.enerj.mae;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;

public abstract class AbstractMAE {

	private static final Logger log = LoggerFactory.getLogger(AbstractMAE.class);

	protected final AbstractActivationFunction activation;
	
	public AbstractMAE(AbstractActivationFunction activation) {
		this.activation = activation;
	}
	
	public abstract void train(DoubleMatrix2D X) throws Exception;
	
	public abstract DoubleMatrix2D f(DoubleMatrix2D X);
}
