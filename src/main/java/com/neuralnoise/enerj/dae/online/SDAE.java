package com.neuralnoise.enerj.dae.online;

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

	public void pretrain(List<DoubleMatrix1D> xs, final double p, final double step, final double thr, final int minits, int maxits) {

		List<DoubleMatrix1D> nxs = xs;
		for (int l = 0; l < this.layers.length; ++l) {
			Layer layer = this.layers[l];

			log.info("Pretraining layer " + l + " ..");

			final int m = Ints.checkedCast(layer.getB1().size()), n = Ints.checkedCast(layer.getB2().size());
			DAE dae = new DAE(n, m, this.activation, this.loss, this.regularizers, this.prng);

			double prevAvgLoss = Double.POSITIVE_INFINITY, gain = Double.POSITIVE_INFINITY;
			for (int t = 0; (t < maxits && gain >= thr) || t < minits; ++t) {
				double loss = 0.0, dn = nxs.size();

				for (DoubleMatrix1D x : nxs) {
					dae.train(x, (l == 0 ? p : 1.0), step);

					loss += dae.loss(x);
				}

				final double avgLoss = (loss / dn);
				log.info("[" + t + "] (PT, layer " + l + ") avg loss: " + avgLoss);

				gain = prevAvgLoss - avgLoss;
				prevAvgLoss = avgLoss;
			}

			DoubleMatrix2D nW = dae.getW();
			DoubleMatrix1D nB1 = dae.getB1(), nB2 = dae.getB2();

			Layer nl = new Layer(nW, nB1, nB2);

			this.layers[l] = nl;

			List<DoubleMatrix1D> _nxs = Lists.newLinkedList();
			for (DoubleMatrix1D x : nxs) {
				_nxs.add(dae.f(x));
			}

			nxs = _nxs;
		}

	}

}
