package com.neuralnoise.enerj.loss;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CrossEntropyLoss extends AbstractLossFunction {

	private static final Logger log = LoggerFactory.getLogger(CrossEntropyLoss.class);

	public static CrossEntropyLoss create() {
		return new CrossEntropyLoss();
	}

	@Override
	public double f(double x, double z) {
		// - [ x log z + (1 - x) log(1 - z) ]
		final double ret = -(x * Math.log(z) + (1.0 - x) * Math.log(1.0 - z));
		// System.out.println("f(" + x + ", " + z + ") = " + ret);
		return ret;
	}

	// derivative w.r.t. z
	@Override
	public double df(double x, double z) {
		// (x - z)/(z*(z - 1))
		double ret = ((x - z) / (z * (z - 1.0)));

		// XXX: quick hack for avoiding NANs, assuming x is either 0 or 1
		if (Double.isNaN(ret)) {
			ret = (z > (1.0 - 1e-12) ? -1.0 : 1.0);
		}

		/*
		 * if (Double.isNaN(ret)) { log.info("NAN from: ce(" + x + ", " + z +
		 * ")"); }
		 * 
		 * if (Double.isInfinite(ret)) { log.info("Inf from: ce(" + x + ", " + z
		 * + ")"); }
		 */

		// System.out.println("df(" + x + ", " + z + ") = " + ret);
		return ret;
	}

}
