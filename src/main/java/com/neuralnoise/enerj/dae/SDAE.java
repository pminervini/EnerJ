package com.neuralnoise.enerj.dae;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.random.tdouble.engine.DoubleRandomEngine;

import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;
import com.neuralnoise.enerj.activation.AbstractActivationFunction;
import com.neuralnoise.enerj.dae.util.Layer;
import com.neuralnoise.enerj.loss.AbstractLossFunction;
import com.neuralnoise.enerj.regularizer.AbstractRegularizer;

public class SDAE extends MLAE {

	private static final Logger log = LoggerFactory.getLogger(SDAE.class);

	public SDAE(final int N, List<Integer> hidden, AbstractActivationFunction activation, AbstractLossFunction loss, List<Pair<AbstractRegularizer, Double>> regularizers, DoubleRandomEngine prng) {
		super(N, hidden, activation, loss, regularizers, prng);
	}

	public void pretrain(List<DoubleMatrix2D> Xs, final double p, final double step, final double thr, final int window, final int minits, int maxits) {

		List<DoubleMatrix2D> nXs = Xs;
		for (int lid = 0; lid < this.layers.length; ++lid) {
			Layer layer = this.layers[lid];

			log.info("Pretraining layer " + lid + " ..");

			final int m = Ints.checkedCast(layer.getB1().size()), n = Ints.checkedCast(layer.getB2().size());

			DAE dae = new DAE(n, m, this.activation, this.loss, this.regularizers, this.prng);
			dae.train(nXs, p, step, thr, window, minits, maxits);

			DoubleMatrix2D nW = dae.getW();
			DoubleMatrix1D nB1 = dae.getB1(), nB2 = dae.getB2();

			Layer nl = new Layer(nW, nB1, nB2);

			this.layers[lid] = nl;

			List<DoubleMatrix2D> _nXs = Lists.newLinkedList();
			for (DoubleMatrix2D X : nXs) {
				_nXs.add(dae.f(X));
			}
			nXs = _nXs;
		}

	}

}
