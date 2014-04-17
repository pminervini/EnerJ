package com.neuralnoise.enerj.mae;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.util.MatrixUtils;

public class MDAE extends AbstractMAE {

	private static final Logger log = LoggerFactory.getLogger(MDAE.class);
	
	final double p, lambda;
	private DoubleMatrix2D W;
	
	public MDAE(AbstractActivationFunction activation, final double p, final double lambda) {
		super(activation);
		//this.p = p;

		// XXX: this is needed to match the interpretation of 'p' used in other denoising methods;
		// i.e. 'p' now represent the probability that the original data is preserved
		this.p = 1.0 - p;
		
		this.lambda = lambda;
	}

	// X \in R^(d x n), d: features, n: instances
	@Override
	public void train(DoubleMatrix2D _X) throws Exception {
		final int d = _X.rows() + 1, n = _X.columns();
		
		log.info("train(" + _X.rows() + " x " + _X.columns() + ", " + p + ", " + lambda + ") ..");
		
		// X = [ X; ones(1, size(X, 2)) ];
		final DoubleMatrix2D X = new DenseDoubleMatrix2D(d, n);
		for (int r = 0; r < d; ++r) {
			for (int c = 0; c < n; ++c) {
				X.set(r, c, (r == (d - 1) ? 1.0 : _X.get(r, c)));
			}
		}
		
		// S = X * X'
		DoubleMatrix2D S = MatrixUtils.product(X, MatrixUtils.transpose(X));

		// q = [ ones(d - 1, 1) .* (1 - p) ; 1 ];
		DoubleMatrix1D q = new DenseDoubleMatrix1D(d);
		for (int r = 0; r < d; ++r) {
			q.set(r, (r == d - 1 ? 1.0 : (1.0 - p)));
		}
		
		DoubleMatrix2D qqt = MatrixUtils.outerProduct(q, q);
		
		// Q = S .* (q * q');
		DoubleMatrix2D Q = MatrixUtils.hadamardProduct(S, qqt);
		
		// Q(1 : d + 1 : end) = q . diag(S);
		for (int i = 0; i < Math.min(Q.rows(), Q.columns()); ++i) {
			Q.set(i, i, q.get(i) * S.get(i, i));
		}
		
		// P = S .* repmat(Q', d, 1);
		DoubleMatrix2D P = S.copy();
		for (int r = 0; r < P.rows(); ++r) {
			P.viewRow(r).assign(q, DoubleFunctions.mult);
		}
		
		// Q = Q + 1e-5 * eye(d)
		for (int i = 0; i < Math.min(Q.rows(), Q.columns()); ++i) {
			Q.set(i, i, Q.get(i, i) + lambda);
		}
		
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		DoubleMatrix2D _P = P.viewPart(0, 0, d - 1, d); 

		this.W = alg.transpose(alg.solve(Q, alg.transpose(_P)));
		//this.W = alg.mult(_P, alg.inverse(Q));
	}

	@Override
	public DoubleMatrix2D f(DoubleMatrix2D _X) {
		final int d = _X.rows() + 1, n = _X.columns();
		
		// X = [ X; ones(1, size(X, 2)) ];
		final DoubleMatrix2D X = new DenseDoubleMatrix2D(d, n);
		for (int r = 0; r < d; ++r) {
			for (int c = 0; c < n; ++c) {
				X.set(r, c, (r == (d - 1) ? 1.0 : _X.get(r, c)));
			}
		}
		
		DoubleMatrix2D R = MatrixUtils.product(W, X).viewPart(0, 0, d - 1, n);
		return this.activation.f(R);
	}
	
}
