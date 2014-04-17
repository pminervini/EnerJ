package com.neuralnoise.enerj.mae;

import java.util.Iterator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;

public class MSDAE extends AbstractMAE {

	private static final Logger log = LoggerFactory.getLogger(MSDAE.class);
	
	private MDAE[] msdae;
	
	private final List<Double> ps;
	private final List<Double> lambdas;
	
	public MSDAE(AbstractActivationFunction activation, final List<Double> ps, final List<Double> lambdas) {
		super(activation);
		
		this.ps = ps;
		this.lambdas = lambdas;
		
		final int layers = ps.size();
		this.msdae = new MDAE[layers];
	}
	
	public void train(DoubleMatrix2D X) throws Exception {
		final int L = this.msdae.length;

		Iterator<Double> pit = ps.iterator(), lit = lambdas.iterator();
		
		for (int l = 0; l < L; ++l) {
			this.msdae[l] = new MDAE(activation, pit.next(), lit.next());
			this.msdae[l].train(X);
			X = this.msdae[l].f(X);
		}
	}

	@Override
	public DoubleMatrix2D f(DoubleMatrix2D X) {
		DoubleMatrix2D R = X;
		for (int l = 0; l < this.msdae.length; ++l) {
			R = this.msdae[l].f(R);
		}
		return R;
	}

}
