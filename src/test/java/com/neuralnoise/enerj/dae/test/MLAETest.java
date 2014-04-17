package com.neuralnoise.enerj.dae.test;

import java.io.IOException;
import java.util.List;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

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

public class MLAETest extends TestCase {

	private static final Logger log = LoggerFactory.getLogger(MLAETest.class);

	public MLAETest(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(MLAETest.class);
	}

	public void testMLAE() throws IOException {
		
		log.info("Testing a Multi-Layer Auto Encoder ..");
		
		final int BATCH_SIZE = 100;
		
		final String imgPath = MLAETest.class.getResource("/mnist/train-images-idx3-ubyte").getFile(),
				lblPath = MLAETest.class.getResource("/mnist/train-labels-idx1-ubyte").getFile();
		
		List<Pair<DoubleMatrix2D, Integer>> list = MNISTUtils.getInstances(imgPath, lblPath, BATCH_SIZE);

		final int N = 28 * 28, M = 100;

		List<DoubleMatrix1D> vs = Lists.newLinkedList();
		for (Pair<DoubleMatrix2D, Integer> p : list) {
			vs.add(p.getKey().vectorize().viewPart(0, N));
		}

		// list<n x t>
		List<DoubleMatrix2D> Xs = MatrixUtils.batches(vs, BATCH_SIZE);

		AbstractActivationFunction af = Sigmoid.create();
		AbstractLossFunction lf = QuadraticLoss.create();

		List<Pair<AbstractRegularizer, Double>> regularizers = Lists.newLinkedList();

		{
			AbstractRegularizer l2 = L2Regularizer.create();
			regularizers.add(Pair.of(l2, 1e-8));
		}

		DoubleRandomEngine prng = RandomUtils.getPRNG();

		final int minits = 25, maxits = 100, window = 25;
		final double thr = 0.001, corr = 1.0; // 0.75;
		final float step = 0.1f;
		
		MLAE ae = new MLAE(N, ImmutableList.of(M), af, lf, regularizers, prng);

		log.info("Training the MLAE ..");
		
		ae.train(Xs, corr, step, thr, window, minits, maxits);
		assertTrue(true);
	}

}
