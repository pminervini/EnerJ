package com.neuralnoise.enerj.util;

import java.io.IOException;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import com.google.common.collect.Lists;
import com.neuralnoise.util.mnist.MnistManager;

public class MNISTUtils {

	private MNISTUtils() {
	}

	public static List<Pair<DoubleMatrix2D, Integer>> getInstances(final String imgPath, final String lblPath, final int n) throws IOException {
		MnistManager m = new MnistManager(imgPath, lblPath);

		int imgCount = m.getImages().getCount();
		int lblCount = m.getLabels().getCount();

		//assertTrue(imgCount == lblCount);

		List<Pair<DoubleMatrix2D, Integer>> list = Lists.newLinkedList();

		for (int idx = 1; idx <= (n > 0 ? Math.min(n, imgCount) : imgCount); ++idx) {
			m.setCurrent(idx);

			int[][] image = m.readImage();
			Integer label = m.readLabel();

			DoubleMatrix2D M = new DenseDoubleMatrix2D(image.length, image[0].length);
			for (int r = 0; r < M.rows(); ++r) {
				for (int c = 0; c < M.columns(); ++c) {
					double v = image[r][c];
					M.set(r, c, v / 255.0);
					// if (v != 0.0) {
					// M.set(r, c, 1.0);
					// }
				}
			}

			list.add(Pair.of(M, label));
		}

		return list;
	}

}
