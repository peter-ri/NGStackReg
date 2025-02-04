/*
    NGStackReg a fast image stack registration software
    Copyright (C) 2025  Peter D. Ringel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

package ch.unibas.biozentrum.imagejplugins.implementations;
import ch.unibas.biozentrum.imagejplugins.NGStackReg;
import ch.unibas.biozentrum.imagejplugins.abstracts.CPUAligner;
import ch.unibas.biozentrum.imagejplugins.abstracts.ImageConverter;
import ch.unibas.biozentrum.imagejplugins.abstracts.RegistrationAndTransformation;
import ch.unibas.biozentrum.imagejplugins.util.ImageConverterFactory;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */

public class MTNGStackReg extends RegistrationAndTransformation {
    public static final double pole = -0.26794919243112270647255365849413;
    public static final double[] polesDeg7 = {-0.5352804307964381655424037816816460718339231523426924148812, -0.122554615192326690515272264359357343605486549427295558490763, -0.0091486948096082769285930216516478534156925639545994482648003 };
    public static final double h0D3 = 0.66666666666666666666666666666666666666666666666666666666666666667;
    public static final double h1D3 = 0.16666666666666666666666666666666666666666666666666666666666666667;
    public static final double h0D7 = 0.4793650793650793650793650793650793650793650793650793650793650793651;
    public static final double h1D7 = 0.23630952380952380952380952380952380952380952380952380952380952380952;
    public static final double h2D7 = 0.023809523809523809523809523809523809523809523809523809523809523810;
    public static final double h3D7 = 0.00019841269841269841269841269841269841269841269841269841269841269841;
    public static final double lambda7 = 5040.0;
    public static final double rh0 = 0.375;
    public static final double rh1 = 0.25;
    public static final double rh2 = 0.0625;
    
    private int pyramidDepth = 1;
    private final int nrOfProcessors;
    private final ImageConverter converter;
    
    private CPUAligner[] workers;
    
    
    public MTNGStackReg(final AbstractSharedContext sharedContext) {
        super(sharedContext);
        converter = ImageConverterFactory.getImageConverter(sharedContext.img);
        calculatePyramidDepth();
        nrOfProcessors = Runtime.getRuntime().availableProcessors() - sharedContext.nrOfGPUDevices; // This will always be >0 due to the instantiation pretest in NGStackReg
        sharedContext.addParties(nrOfProcessors);
    }
    @Override
    public void register() throws InterruptedException
    {
        workers = new CPUAligner[nrOfProcessors];
        for(int i = 0; i < nrOfProcessors; i++)
        {
            if(sharedContext.forceDoublePrecisionRepr)
            {
                workers[i] = (CPUAligner)new PlainJavaCPUAligner(sharedContext, converter, pyramidDepth);
            }
            else
            {
                workers[i] = (CPUAligner)new HybridPrecisionCPUAligner(sharedContext, converter, pyramidDepth);
            }
        }
        for(int i = 0; i < nrOfProcessors; i++)
        {
            workers[i].start();
        }
    }

    private void calculatePyramidDepth()
    {
        long s = sharedContext.img.dimension(0) < sharedContext.img.dimension(1) ? sharedContext.img.dimension(0) : sharedContext.img.dimension(1);
        while (s >= NGStackReg.MIN_SIZE) {
            s /= 2;
            pyramidDepth++;
        }
    }
    
    boolean getFinishedTransformations()
    {
        return sharedContext.finishedTransformations;
    }

    @Override
    public void waitForFinish()  throws InterruptedException {
        if(workers != null)
        {
            for(int i = 0; i < nrOfProcessors; i++)
            {
                if(workers[i] != null)
                {
                    workers[i].getThread().join();
                }
            }
        }
    }

    @Override
    public void release()
    {
        if(workers != null)
        {
            for(int i = 0; i < nrOfProcessors; i++)
            {
                if(workers[i] != null)
                {
                    workers[i].release(); // Nothing must be called on these instances again
                }
            }
            workers = null;
        }
    }
}
