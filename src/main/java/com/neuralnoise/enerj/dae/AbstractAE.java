package com.neuralnoise.enerj.dae;

import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.dae.util.CorruptionFunction;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;

public abstract class AbstractAE {

	private static final Logger log = LoggerFactory.getLogger(AbstractAE.class);

	protected AbstractActivationFunction activation;
	protected List<Pair<AbstractRegularizer, Double>> regularizers;
	protected AbstractLossFunction loss;
	protected DoubleRandomEngine prng;

	public AbstractAE(AbstractActivationFunction activation, AbstractLossFunction loss, List<Pair<AbstractRegularizer, Double>> regularizers, DoubleRandomEngine prng) {
		this.activation = activation;
		this.loss = loss;
		this.regularizers = regularizers;
		this.prng = prng;
	}

	public abstract DoubleMatrix2D f(DoubleMatrix2D X);

	public abstract DoubleMatrix2D g(DoubleMatrix2D Y);

	public abstract double loss(DoubleMatrix2D X);

	public abstract void train(DoubleMatrix2D X, double p, double step);

	public DoubleMatrix2D corrupt(DoubleMatrix2D X, double p) {
		DoubleMatrix2D tX = X.copy();
		DoubleFunction corrupt = new CorruptionFunction(this.prng, p);
		if (tX instanceof DenseDoubleMatrix2D) {
			for (int r = 0; r < tX.rows(); ++r) {
				for (int c = 0; c < tX.columns(); ++c) {
					tX.set(r, c, corrupt.apply(tX.get(r, c)));
				}
			}
		} else {
			tX.assign(corrupt);
		}
		return tX;
	}
	
	public void train(List<DoubleMatrix2D> Xs, final double p, final double step, final double thr, final int window, final int minits, int maxits) {

		PriorityQueue<Double> lowestLosses = new PriorityQueue<Double>(window, Collections.reverseOrder());

		double maxGain = Double.POSITIVE_INFINITY;
		for (int t = 0; (t < maxits && maxGain >= thr) || t < minits; ++t) {
			double loss = 0.0;

			for (DoubleMatrix2D X : Xs) {
				train(X, p, step);
				double l = loss(X);

				loss += l;
			}

			final double avgLoss = loss;
			log.info("[" + t + "] (T) avg loss: " + avgLoss);

			maxGain = Double.NEGATIVE_INFINITY;
			for (double prevAvgLoss : lowestLosses) {
				if (maxGain < (prevAvgLoss - avgLoss)) {
					maxGain = prevAvgLoss - avgLoss;
				}
			}

			lowestLosses.add(avgLoss);
			if (lowestLosses.size() > window) {
				lowestLosses.poll();
			}
		}
	}

	public static void main(String[] args) {
		PriorityQueue<Double> pq = new PriorityQueue<Double>(5, Collections.reverseOrder());

		pq.add(0.1);
		pq.add(0.2);
		pq.add(0.3);
		pq.add(0.4);
		pq.add(0.5);
		pq.add(0.6);
		pq.add(0.7);

		log.info("pq: " + pq);

		pq.poll();

		log.info("pq: " + pq);
	}

}
