package com.neuralnoise.enerj.dae.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.function.tdouble.IntIntDoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.util.RandomUtils;

public class CorruptionFunction implements IntIntDoubleFunction, DoubleFunction {

	private static final Logger log = LoggerFactory.getLogger(CorruptionFunction.class);

	private final DoubleRandomEngine prng;
	private final double p;

	public CorruptionFunction(DoubleRandomEngine prng, double p) {
		this.prng = prng;
		this.p = p;
	}

	//@Override
	public double apply(final int a, final int b, final double c) {
		return apply(c);
	}

	//@Override
	public double apply(final double c) {
		double ret = c;
		if (c != 0.0) {
			if (RandomUtils.binomial(this.prng, 1, p) == 0) {
				ret = 0.0;
			}
		}
		return ret;
	}

	public static void main(String[] args) {
		DoubleRandomEngine prng = RandomUtils.getPRNG();

		DoubleFunction corrupt = new CorruptionFunction(prng, 0.9);

		for (int i = 0; i < 10; ++i) {
			log.info("c: " + corrupt.apply(3.4));
		}
		
		System.exit(0);
		
		final int R = 500, C = 1000;

		for (int t = 0; t < 100000; ++t) {
			log.debug("sparse ..");
			DoubleMatrix2D srv = RandomUtils.randomSparseMatrix(prng, R, C, 0.1);
			// rv.forEachNonZero(corrupt);
			srv.assign(corrupt);

			log.debug("dense ..");
			// DoubleMatrix2D rv = RandomUtils.randomMatrix(prng, R, C);
			// rv.assign(corrupt);
		}
	}

}
