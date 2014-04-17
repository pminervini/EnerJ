package com.neuralnoise.enerj.energy.tensor.util;

public abstract class AbstractParameters {

	public AbstractParameters() { }
	
	public abstract AbstractParameters sum(AbstractParameters a, AbstractParameters b, double alpha);
	
}
