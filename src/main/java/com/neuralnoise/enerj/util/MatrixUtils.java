package com.neuralnoise.enerj.util;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DiagonalDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;

public class MatrixUtils {

	private static final Logger log = LoggerFactory.getLogger(MatrixUtils.class);

	private MatrixUtils() { }

	public static double innerProduct(DoubleMatrix1D v, DoubleMatrix1D w) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return alg.mult(v, w);
	}
	
	public static DoubleMatrix2D product(DoubleMatrix2D A, DoubleMatrix2D B) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return alg.mult(A, B);
	}

	public static DoubleMatrix1D product(DoubleMatrix2D A, DoubleMatrix1D B) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return alg.mult(A, B);
	}

	public static DoubleMatrix2D sum(DoubleMatrix2D A, DoubleMatrix2D B) {
		DoubleMatrix2D S = A.copy();
		S.assign(B, DoubleFunctions.plus);
		return S;
	}

	public static DoubleMatrix1D sum(DoubleMatrix1D A, DoubleMatrix1D B) {
		DoubleMatrix1D S = A.copy();
		S.assign(B, DoubleFunctions.plus);
		return S;
	}

	public static DoubleMatrix1D subtraction(DoubleMatrix1D A, DoubleMatrix1D B) {
		DoubleMatrix1D S = A.copy();
		S.assign(B, DoubleFunctions.minus);
		return S;
	}

	public static DoubleMatrix2D subtraction(DoubleMatrix2D A, DoubleMatrix2D B) {
		DoubleMatrix2D S = A.copy();
		S.assign(B, DoubleFunctions.minus);
		return S;
	}

	public static DoubleMatrix2D transpose(DoubleMatrix2D A) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return alg.transpose(A);
	}

	public static DoubleMatrix1D product(DoubleMatrix1D A, double b) {
		DoubleMatrix1D bA = A.copy();
		bA.assign(DoubleFunctions.mult(b));
		return bA;
	}

	public static DoubleMatrix2D product(DoubleMatrix2D A, double b) {
		if (A == null) {
			return null;
		}
		DoubleMatrix2D bA = A.copy();
		bA.assign(DoubleFunctions.mult(b));
		return bA;
	}

	public static DoubleMatrix2D abs(DoubleMatrix2D A) {
		if (A == null) {
			return null;
		}
		DoubleMatrix2D bA = A.copy();
		bA.assign(DoubleFunctions.abs);
		return bA;
	}
	
	public static double sum(DoubleMatrix1D v) {
		double sum = v.zSum();
		return sum;
	}

	public static double sum(DoubleMatrix2D A) {
		double sum = A.zSum();
		return sum;
	}

	public static DoubleMatrix1D hadamardProduct(DoubleMatrix1D A, DoubleMatrix1D B) {
		DoubleMatrix1D S = A.copy();
		S.assign(B, DoubleFunctions.mult);
		return S;
	}

	public static DoubleMatrix2D hadamardProduct(DoubleMatrix2D A, DoubleMatrix2D B) {
		DoubleMatrix2D S = A.copy();
		S.assign(B, DoubleFunctions.mult);
		return S;
	}

	public static DoubleMatrix2D outerProduct(DoubleMatrix1D A, DoubleMatrix1D B) {
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return alg.xmultOuter(A, B);
	}

	public static boolean hasNAN(DoubleMatrix1D A) {
		for (int i = 0; i < A.size(); ++i) {
			double v = A.get(i);
			if (Double.isNaN(v) || Double.isInfinite(v)) {
				return true;
			}
		}
		return false;
	}

	public static boolean hasNAN(DoubleMatrix2D A) {
		for (int i = 0; i < A.rows(); ++i) {
			for (int j = 0; j < A.columns(); ++j) {
				double v = A.get(i, j);
				if (Double.isNaN(v) || Double.isInfinite(v)) {
					log.info("NAN: " + v);
					return true;
				}
			}
		}
		return false;
	}

	// list<n x t>
	public static List<DoubleMatrix2D> batches(List<DoubleMatrix1D> xs, int batchSize) {
		List<DoubleMatrix2D> Xs = Lists.newLinkedList();
		int c = 0;
		DoubleMatrix2D X = null;
		for (DoubleMatrix1D x : xs) {
			final int remaining = xs.size() - c;
			if (c % batchSize == 0) {
				X = new DenseDoubleMatrix2D(Ints.checkedCast(x.size()), Math.min(batchSize, remaining));
				Xs.add(X);
			}
			for (int r = 0; r < x.size(); ++r) {
				X.set(r, c % batchSize, x.get(r));
			}
			c += 1;
		}
		return Xs;
	}

	public static DiagonalDoubleMatrix2D fractionalPow(DiagonalDoubleMatrix2D D, double power) {
		DiagonalDoubleMatrix2D _D = new DiagonalDoubleMatrix2D(D.rows(), D.columns(), 0);
		for (int i = 0; i < Math.min(_D.rows(), _D.columns()); ++i) {
			double dii = D.get(i, i);
			_D.set(i, i, (dii != 0.0 ? Math.pow(dii, power) : 0.0));
		}
		return _D;
	}

	public static DiagonalDoubleMatrix2D diagonal(double[] diagonal) {
		int n = diagonal.length;
		DiagonalDoubleMatrix2D D = new DiagonalDoubleMatrix2D(n, n, 0);
		for (int i = 0; i < n; ++i) {
			D.set(i, i, diagonal[i]);
		}
		return D;
	}
	
}
