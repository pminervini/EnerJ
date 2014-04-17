package com.neuralnoise.enerj.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

public class RandomUtils {

	private static final Logger log = LoggerFactory.getLogger(RandomUtils.class);

	private RandomUtils() {
	}

	public static DoubleRandomEngine getPRNG() {
		DoubleRandomEngine prng = new DoubleMersenneTwister(0);
		return prng;
	}

	public static DoubleMatrix2D randomMatrix(DoubleRandomEngine prng, int rows, int cols, double min, double max) {
		DoubleMatrix2D M = new DenseDoubleMatrix2D(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				double val = uniform(prng, min, max);
				M.set(r, c, val);
			}
		}
		return M;
	}

	public static DoubleMatrix2D randomSparseMatrix(DoubleRandomEngine prng, int rows, int cols, double p, double min, double max) {
		DoubleMatrix2D M = new SparseDoubleMatrix2D(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (uniform(prng, 0.0, 1.0) < p) {
					double val = uniform(prng, min, max);
					M.set(r, c, val);
				}
			}
		}
		return M;
	}

	public static DoubleMatrix2D randomMatrix(DoubleRandomEngine prng, int rows, int cols) {
		return randomMatrix(prng, rows, cols, 0.0, 1.0);
	}

	public static DoubleMatrix2D randomSparseMatrix(DoubleRandomEngine prng, int rows, int cols, double p) {
		return randomSparseMatrix(prng, rows, cols, p, 0.0, 1.0);
	}

	public static DoubleMatrix1D randomVector(DoubleRandomEngine prng, int rows, double min, double max) {
		DoubleMatrix1D v = new DenseDoubleMatrix1D(rows);
		for (int r = 0; r < rows; ++r) {
			double val = uniform(prng, min, max);
			v.set(r, val);
		}
		return v;
	}

	public static float[] randomArray(DoubleRandomEngine prng, int N, float min, float max) {
		float ret[] = new float[N];
		for (int i = 0; i < N; ++i) {
			ret[i] = funiform(prng, min, max);
		}
		return ret;
	}
	
	public static double uniform(DoubleRandomEngine prng, double min, double max) {
		return prng.nextDouble() * (max - min) + min;
	}

	public static float funiform(DoubleRandomEngine prng, float min, float max) {
		return prng.nextFloat() * (max - min) + min;
	}

	public static int binomial(DoubleRandomEngine prng, int n, double p) {
		int c = 0;
		for (int i = 0; i < n; i++) {
			final double r = prng.nextDouble();
			if (r < p) {
				c++;
			}
		}
		return c;
	}

	public static DoubleMatrix1D randomVector(DoubleRandomEngine prng, int rows) {
		return randomVector(prng, rows, 0.0, 1.0);
	}

}
