package com.neuralnoise.dae.test;

import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.activation.Sigmoid;
import com.neuralnoise.enerj.dae.MLAE;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.loss.QuadraticLoss;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.regularizer.L2Regularizer;
import com.neuralnoise.enerj.util.MNISTUtils;
import com.neuralnoise.enerj.util.MatrixUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class BMNISTAETest {

	private static final Logger log = LoggerFactory.getLogger(BMNISTAETest.class);

	public static void main(String[] args) throws IOException {
		
		final int _N = 1000;
		
		List<Pair<DoubleMatrix2D, Integer>> list = MNISTUtils.getInstances("res/mnist/train-images-idx3-ubyte", "res/mnist/train-labels-idx1-ubyte", _N);

		final int N = 28 * 28, M = 100;
		//final int N = 28, M = 10;

		List<DoubleMatrix1D> vs = Lists.newLinkedList();
		for (Pair<DoubleMatrix2D, Integer> p : list) {
			vs.add(p.getKey().vectorize().viewPart(0, N));
		}

		final int batchSize = _N; //500;//500;
		// list<n x t>
		List<DoubleMatrix2D> Xs = MatrixUtils.batches(vs, batchSize);

		AbstractActivationFunction af = Sigmoid.create();
		AbstractLossFunction lf = QuadraticLoss.create();

		List<Pair<AbstractRegularizer, Double>> regularizers = Lists.newLinkedList();

		{
			AbstractRegularizer l2 = L2Regularizer.create();
			regularizers.add(Pair.of(l2, 1e-8));
		}

		DoubleRandomEngine prng = RandomUtils.getPRNG();

		final int minits = 25, maxits = 10000, window = 25;
		final double thr = 0.001, corr = 1.0; // 0.75;
		final float step = 0.1f;
		
		//DAE ae = new DAE(N, M, af, lf, regularizers, prng);
		MLAE ae = new MLAE(N, ImmutableList.of(M), af, lf, regularizers, prng);

		log.info("Training the MLAE ..");
		ae.train(Xs, corr, step, thr, window, minits, maxits);
	}

}
