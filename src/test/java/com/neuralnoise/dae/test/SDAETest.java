package com.neuralnoise.dae.test;

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
import com.neuralnoise.enerj.activation.Sigmoid;
import com.neuralnoise.enerj.dae.online.SDAE;
import com.neuralnoise.enerj.loss.CrossEntropyLoss;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.regularizer.L2Regularizer;
import com.neuralnoise.enerj.util.MNISTUtils;
import com.neuralnoise.enerj.util.RandomUtils;

public class SDAETest extends TestCase {

	private static final Logger log = LoggerFactory.getLogger(SDAETest.class);

	public SDAETest(String testName) {
		super(testName);
	}

	public static Test suite() {
		return new TestSuite(SDAETest.class);
	}
	
	public static void main(String[] args) throws IOException {

		final int BATCH_SIZE = 100;
		
		final String imgPath = SDAETest.class.getResource("/mnist/train-images-idx3-ubyte").getFile(),
				lblPath = SDAETest.class.getResource("/mnist/train-labels-idx1-ubyte").getFile();
		
		List<Pair<DoubleMatrix2D, Integer>> list = MNISTUtils.getInstances(imgPath, lblPath, BATCH_SIZE);
		
		List<DoubleMatrix1D> xs = Lists.newLinkedList();
		for (Pair<DoubleMatrix2D, Integer> p : list) {
			xs.add(p.getKey().vectorize());
		}

		CrossEntropyLoss ce = CrossEntropyLoss.create();

		List<Pair<AbstractRegularizer, Double>> regularizers = Lists.newLinkedList();

		{
			AbstractRegularizer l2 = L2Regularizer.create();
			Pair<AbstractRegularizer, Double> reg = Pair.of(l2, 1e-3);
			regularizers.add(reg);
		}

		DoubleRandomEngine prng = RandomUtils.getPRNG();

		SDAE mlae = new SDAE(28 * 28, ImmutableList.of(1000, 1000, 1000), Sigmoid.create(), ce, regularizers, prng);

		final int minits = 500, maxits = 1000;
		final double step = 0.1, thr = 0.01, corr = 0.75;

		mlae.pretrain(xs, corr, step, thr, minits, maxits);

		double prevAvgLoss = Double.POSITIVE_INFINITY, gain = Double.POSITIVE_INFINITY;
		for (int t = 0; (t < maxits && gain >= thr) || t < minits; ++t) {
			double loss = 0.0, dn = xs.size();

			for (DoubleMatrix1D x : xs) {
				mlae.train(x, corr, step);
				loss += mlae.loss(x);
			}

			final double avgLoss = (loss / dn);
			log.info("[" + t + "] (T) avg loss: " + avgLoss);

			gain = prevAvgLoss - avgLoss;
			prevAvgLoss = avgLoss;
		}

	}

}
