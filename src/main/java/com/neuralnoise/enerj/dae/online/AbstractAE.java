package com.neuralnoise.enerj.dae.online;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;
import com.neuralnoise.enerj.util.RandomUtils;

public abstract class AbstractAE {

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

	public abstract DoubleMatrix1D f(DoubleMatrix1D x);

	public abstract DoubleMatrix1D g(DoubleMatrix1D y);

	public abstract double loss(DoubleMatrix1D x);

	public abstract void train(DoubleMatrix1D x, double p, double step);

	public DoubleMatrix1D corrupt(DoubleMatrix1D x, double p) {
		DoubleMatrix1D tx = x.copy();
		for (int r = 0; r < tx.size(); ++r) {
			if (tx.get(r) != 0) {
				if (RandomUtils.binomial(this.prng, 1, p) == 0) {
					tx.set(r, 0.0);
				}
			}
		}
		return tx;
	}

}
