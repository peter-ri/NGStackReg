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
import ch.unibas.biozentrum.imagejplugins.abstracts.Transformation;
import ch.unibas.biozentrum.imagejplugins.util.RigidBodyTransformation;
import ch.unibas.biozentrum.imagejplugins.util.TranslationTransformation;
import java.io.PrintWriter;
import java.util.concurrent.CyclicBarrier;
import net.imagej.ImgPlus;
import net.imagej.Position;
import net.imagej.axis.DefaultLinearAxis;
import net.imglib2.img.planar.PlanarImg;
import org.scijava.log.LogService;
import org.scijava.app.StatusService;
import org.json.*;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public class SharedContextZT extends AbstractSharedContext {
    Position currentPosition;// WARNING: this variable will be modified internally by the worker threads. This must not be static, otherwise multiple instances would conflict.
    Transformation[][][] transformations;
    Transformation[][][] transformationsZ;
    Transformation[][][] transformationsT;
    
    int axisID;
    final int axisIDT;
    final int axisIDZ;
    final Position pos;
    int[] transformationExtents;
    final int[] transformationExtentsZ = {1,1,1};
    final int[] transformationExtentsT = {1,1,1};
    final LogService logService;
    final StatusService statusService;
    int nrOfAlignments;
    int nrOfTransformations;
    int currentAlignmentCount = 0;
    int currentTransformationCount = 0;
    
    private boolean fwd = true;
    public SharedContextZT(final NGStackReg.TransformationType transformationType, @SuppressWarnings("rawtypes") final ImgPlus img, final Position pos, final int axisIDZ, final int axisIDT, final boolean forceDoublePrecisionRepr, final LogService logService, final StatusService statusService)
    {
        this.transformationType = transformationType;
        this.img = img;
        this.axisIDT = axisIDT;
        this.axisIDZ = axisIDZ;
        this.axisID = this.axisIDT; //First align along T and then propagate these transformations to Z
        this.pos = pos;
        this.currentPosition = new Position(pos);
        this.logService = logService;
        this.statusService = statusService;
        
        // Determine the number of alignments that have to be calculated
        int dims = img.numDimensions();
        if((dims - 2) < 2)
        {
        	throw new RuntimeException("Too few alignment axes.");
        }
        
        nrOfAlignments = (int)this.img.dimension(this.axisID+2); // Already checked for overflow condition in NGStackReg.java
        long axisCounter = nrOfAlignments > 1 ? 1 : 0;
        for(int i = this.axisID + 1; i < (dims - 2); i++)
        {
            nrOfAlignments *= (int)this.img.dimension(i+2);
            axisCounter *= this.img.dimension(i+2);
        }
        if(axisCounter > (long)Integer.MAX_VALUE)
        {
            throw new RuntimeException("Too many images in the stack, (>Integer.MAX_VALUE)");
        }
        if(axisCounter == 0)
        {
            throw new RuntimeException("Cannot align one image");
        }
        nrOfAlignments -= axisCounter;
        
        // Calculate the number of images that have to be transformed
        long temporaryNumberOfImagesToAlign = img.dimension(2);
        // i = 3 because we already have the first dimension in the counter
        for(int i = 3;i < dims;i++)
        {
            if(img.dimension(i) > (long)Integer.MAX_VALUE)
            {
                throw new RuntimeException("Axis is too large, (>Integer.MAX_VALUE)");
            }
            temporaryNumberOfImagesToAlign *= img.dimension(i);
        }
        if(temporaryNumberOfImagesToAlign > (long)Integer.MAX_VALUE)
        {
            throw new RuntimeException("Too many images in the stack, (>Integer.MAX_VALUE)");
        }
        nrOfTransformations = (int)temporaryNumberOfImagesToAlign; // Checked for overflow
        
        if(this.axisID >= this.currentPosition.numDimensions())
        {
            throw new RuntimeException("Axis out of range");
        }
        for(int i = this.axisID + 1; i < this.currentPosition.numDimensions();i++)
        {
            // Zero the top level dimensions
            this.currentPosition.setPosition(0, i);
        }
        
        // T
        for(int i = this.axisIDT; i < 3;i++)
        {
            if(i < this.currentPosition.numDimensions())
            {
                transformationExtentsT[i] = (int)this.currentPosition.dimension(i);
            }
            else
            {
                break;
            }
        }
        
        transformationExtents = transformationExtentsT;
        
        // Z
        for(int i = this.axisIDZ; i < 3;i++)
        {
            if(i < this.currentPosition.numDimensions())
            {
                transformationExtentsZ[i] = (int)this.currentPosition.dimension(i);
            }
            else
            {
                break;
            }
        }
        
        switch(this.transformationType)
        {
            case TRANSLATION:
            	// T
                transformationsT = new TranslationTransformation[transformationExtentsT[2]][transformationExtentsT[1]][transformationExtentsT[0]];
                for(int j = 0;j < transformationExtentsT[2];j++)
                {
                    for(int k = 0;k < transformationExtentsT[1];k++)
                    {
                        for(int i = 0;i < transformationExtentsT[0];i++)
                        {
                            transformationsT[j][k][i] = new TranslationTransformation();// F Z C (not really because if there are less channels this could be different)
                        }
                    }
                }
                transformations = transformationsT;
                // Z
                transformationsZ = new TranslationTransformation[transformationExtentsZ[2]][transformationExtentsZ[1]][transformationExtentsZ[0]];
                for(int j = 0;j < transformationExtentsZ[2];j++)
                {
                    for(int k = 0;k < transformationExtentsZ[1];k++)
                    {
                        for(int i = 0;i < transformationExtentsZ[0];i++)
                        {
                            transformationsZ[j][k][i] = new TranslationTransformation();// F Z C (not really because if there are less channels this could be different)
                        }
                    }
                }
                break;
            case RIGIDBODY:
                // T
                transformationsT = new RigidBodyTransformation[transformationExtentsT[2]][transformationExtentsT[1]][transformationExtentsT[0]];
                for(int j = 0;j < transformationExtentsT[2];j++)
                {
                    for(int k = 0;k < transformationExtentsT[1];k++)
                    {
                        for(int i = 0;i < transformationExtentsT[0];i++)
                        {
                            transformationsT[j][k][i] = new RigidBodyTransformation();// F Z C (not really because if there are less channels this could be different)
                        }
                    }
                }
                transformations = transformationsT;
                // Z
                transformationsZ = new RigidBodyTransformation[transformationExtentsZ[2]][transformationExtentsZ[1]][transformationExtentsZ[0]];
                for(int j = 0;j < transformationExtentsZ[2];j++)
                {
                    for(int k = 0;k < transformationExtentsZ[1];k++)
                    {
                        for(int i = 0;i < transformationExtentsZ[0];i++)
                        {
                            transformationsZ[j][k][i] = new RigidBodyTransformation();// F Z C (not really because if there are less channels this could be different)
                        }
                    }
                }
                break;
            case SCALEDROTATION:
                break;
            case AFFINE:
                break;
        }
        this.forceDoublePrecisionRepr = forceDoublePrecisionRepr;
        finishedTransformations = false;
    }
    
    @Override
    public int getNrOfGPUDevices()
    {
        return nrOfGPUDevices;
    }
    
    private class StackRegTransformationCombinerWorker implements Runnable
    {

        @Override
        public void run() {
            // reset the position to the first frame so we can now go through all frames from the beginning and just skip the reference frames
            currentPosition.first();
            // Combine the transformation with every preceding transformation (down to the reference)
            int p = (int)pos.getLongPosition(axisID);
            int[] initializers = {0,0,0};
            int[] previous = {0,0,0};
            previous[axisID] = 1;
            initializers[axisID] = p + 1;
            if(p == 0)
            {
                for(int j = initializers[2];j < transformationExtents[2];j++)
                {
                    for(int k = initializers[1];k < transformationExtents[1];k++)
                    {
                        for(int i = initializers[0];i < transformationExtents[0];i++)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j-previous[2]][k-previous[1]][i-previous[0]]);
                        }
                    }
                }
            }
            else
            {
                // need to run forward and reverse
                for(int j = initializers[2];j < transformationExtents[2];j++)
                {
                    for(int k = initializers[1];k < transformationExtents[1];k++)
                    {
                        for(int i = initializers[0];i < transformationExtents[0];i++)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j-previous[2]][k-previous[1]][i-previous[0]]);
                        }
                    }
                }
                // The transformationExtents contains lengths and NOT indices, therefore we need to subtract 1 
                initializers[0] = transformationExtents[0] - 1;
                initializers[1] = transformationExtents[1] - 1;
                initializers[2] = transformationExtents[2] - 1;
                initializers[axisID] = p - 1;
                for(int j = initializers[2];j >= 0;j--)
                {
                    for(int k = initializers[1];k >= 0;k--)
                    {
                        for(int i = initializers[0];i >= 0;i--)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j+previous[2]][k+previous[1]][i+previous[0]]);
                        }
                    }
                }
            }
            
            currentPosition = new Position(pos);
            transformations = transformationsZ;
            transformationExtents = transformationExtentsZ;
            axisID = axisIDZ;
            
            int dims = img.numDimensions();
            
            nrOfAlignments = (int)img.dimension(axisID+2); // Already checked for overflow condition in NGStackReg.java
            long axisCounter = nrOfAlignments > 1 ? 1 : 0;
            for(int i = axisID + 1; i < (dims - 2); i++)
            {
                nrOfAlignments *= (int)img.dimension(i+2);
                axisCounter *= img.dimension(i+2);
            }
            if(axisCounter > (long)Integer.MAX_VALUE)
            {
                throw new RuntimeException("Too many images in the stack, (>Integer.MAX_VALUE)");
            }
            if(axisCounter == 0)
            {
                throw new RuntimeException("Cannot align one image");
            }
            nrOfAlignments -= axisCounter;
            
            for(int i = axisID + 1; i < currentPosition.numDimensions();i++)
            {
                // Zero the top level dimensions
                currentPosition.setPosition(0, i);
            }
            
            // Not sure this would work, because I don't know which synchronization lock is held.
            workerSynchronizationBarrier = new CyclicBarrier(workerSynchronizationBarrier.getParties(),new FinalStackRegTransformationCombinerWorker());
        }

    }
    
    private class FinalStackRegTransformationCombinerWorker implements Runnable
    {

        @Override
        public void run() {
        	statusService.showStatus(currentTransformationCount, nrOfTransformations, "Transforming");
        	// reset the position to the first frame so we can now go through all frames from the beginning and just skip the reference frames
            currentPosition.first();
            // Combine the transformation with every preceding transformation (down to the reference)
            int p = (int)pos.getLongPosition(axisID);
            int[] initializers = {0,0,0};
            int[] previous = {0,0,0};
            previous[axisID] = 1;
            initializers[axisID] = p + 1;
            if(p == 0)
            {
                for(int j = initializers[2];j < transformationExtents[2];j++)
                {
                    for(int k = initializers[1];k < transformationExtents[1];k++)
                    {
                        for(int i = initializers[0];i < transformationExtents[0];i++)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j-previous[2]][k-previous[1]][i-previous[0]]);
                        }
                    }
                }
            }
            else
            {
                // need to run forward and reverse
                for(int j = initializers[2];j < transformationExtents[2];j++)
                {
                    for(int k = initializers[1];k < transformationExtents[1];k++)
                    {
                        for(int i = initializers[0];i < transformationExtents[0];i++)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j-previous[2]][k-previous[1]][i-previous[0]]);
                        }
                    }
                }
                // The transformationExtents contains lengths and NOT indices, therefore we need to subtract 1 
                initializers[0] = transformationExtents[0] - 1;
                initializers[1] = transformationExtents[1] - 1;
                initializers[2] = transformationExtents[2] - 1;
                initializers[axisID] = p - 1;
                for(int j = initializers[2];j >= 0;j--)
                {
                    for(int k = initializers[1];k >= 0;k--)
                    {
                        for(int i = initializers[0];i >= 0;i--)
                        {
                            transformations[j][k][i].invert();
                            transformations[j][k][i].transformWith(transformations[j+previous[2]][k+previous[1]][i+previous[0]]);
                        }
                    }
                }
            }
            
            //Now the T transformations still need to be propagated to the Z transformations, but since these are already absolute
            //we don't have to use the reference frame as initial position. Furthermore, the transformations array is in reverse
            //order of significance transformations[1][T][Z] and the previous T transform transformations[1][T][1]
            //or transformations[T][Z][1] and the previous T transform transformations[T][1][1]
            if(transformationsT.length > 1)
            {
            	for(int j = initializers[2];j < transformationExtents[2];j++)
                {
                    for(int k = initializers[1];k < transformationExtents[1];k++)
                    {
                        for(int i = initializers[0];i < transformationExtents[0];i++)
                        {
                            transformations[j][k][i].transformWith(transformationsT[j][0][0]);
                        }
                    }
                }
            }
            else
            {
	            for(int k = 0;k < transformationExtents[1];k++)
	            {
	                for(int i = 0;i < transformationExtents[0];i++)
	                {
	                    transformations[0][k][i].transformWith(transformationsT[0][k][0]);
	                }
	            }
            }
        }
    }
    
    @Override
    void addParties(int participants)
    {
        // WARNING: THIS MUST BE CALLED !!!BEFORE!!! any threads are actually run
        if(workerSynchronizationBarrier == null)
        {
            workerSynchronizationBarrier = new CyclicBarrier(participants,new StackRegTransformationCombinerWorker());
        }
        else
        {
            workerSynchronizationBarrier = new CyclicBarrier(participants + workerSynchronizationBarrier.getParties(),new StackRegTransformationCombinerWorker());
        }
    }
    
    @Override
    synchronized boolean getNextAlignmentTarget(final SharedContextAlignmentTarget target)
    {
        statusService.showStatus(currentAlignmentCount++,nrOfAlignments, "Aligning");
        // Returns true when done
        if (img.getImg() instanceof PlanarImg) {
            if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                throw new RuntimeException("Image plane index out of range.");
            }
            // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
            target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
            if (target.targetArray == null) {
                throw new RuntimeException("Could not get image plane at specified index.");
            }
        }
        if (fwd) {
            switch (currentPosition.numDimensions()) {
                case 2:
                    // Z or C
                    switch (axisID) {
                        case 0:
                            if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                currentPosition.fwd(axisID);
                            } else {
                                // now we either have to go in reverse or to the next level or we are done
                                if (pos.getIntPosition(axisID) > 0) {
                                    // reverse
                                    fwd = false;
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    currentPosition.bck(axisID);
                                } else if (currentPosition.getIntPosition(1) < currentPosition.dimension(1) - 1) {
                                    currentPosition.fwd(1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 1:
                            if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                currentPosition.fwd(axisID);
                            } else {
                                // now either we have to go in reverse or we are done
                                if (pos.getIntPosition(axisID) > 0) {
                                    // reverse
                                    fwd = false;
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    currentPosition.bck(axisID);
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                    }
                    break;
                case 3:
                    // F/T or Z or C
                    switch (axisID) {
                        case 0:
                            if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                currentPosition.fwd(axisID);
                            } else {
                                // now we either have to go in reverse or to the next level or we are done
                                if (pos.getIntPosition(axisID) > 0) {
                                    // reverse
                                    fwd = false;
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    currentPosition.bck(axisID);
                                } else if (currentPosition.getIntPosition(axisID + 1) < currentPosition.dimension(axisID + 1) - 1) {
                                    currentPosition.fwd(axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            //reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else if (currentPosition.getIntPosition(axisID + 2) < currentPosition.dimension(axisID + 2) - 1) {
                                    // next next dimension
                                    currentPosition.fwd(axisID + 2);
                                    currentPosition.setPosition(0, axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            // Because it would be a single image axis otherwise
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 1:
                            if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                currentPosition.fwd(axisID);
                            } else {
                                // now either we have to go in reverse or to the next level or we are done
                                if (pos.getIntPosition(axisID) > 0) {
                                    // reverse
                                    fwd = false;
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    currentPosition.bck(axisID);
                                } else if (currentPosition.getIntPosition(axisID + 1) < currentPosition.dimension(axisID + 1) - 1) {
                                    currentPosition.fwd(axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            //reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 2:
                            if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                currentPosition.fwd(axisID);
                            } else {
                                // now either we have to go in reverse or we are done
                                if (pos.getIntPosition(axisID) > 0) {
                                    // reverse
                                    fwd = false;
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    currentPosition.bck(axisID);
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                    }
                    break;
            }
        } else {
            switch (currentPosition.numDimensions()) {
                case 2:
                    // Z or C
                    switch (axisID) {
                        case 0:
                            if (currentPosition.getIntPosition(axisID) > 0) {
                                currentPosition.bck(axisID);
                            } else {
                                // now either we have to go to the next level or we are done
                                if (currentPosition.getIntPosition(1) < currentPosition.dimension(1) - 1) {
                                    fwd = true;
                                    currentPosition.fwd(1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 1:
                            if (currentPosition.getIntPosition(axisID) > 0) {
                                currentPosition.bck(axisID);
                            } else {
                                // now either we have to go in reverse or we are done
                                currentAlignmentCount = 0;
                                return true;
                            }
                            break;
                    }
                    break;
                case 3:
                    // F/T or Z or C
                    switch (axisID) {
                        case 0:
                            if (currentPosition.getIntPosition(axisID) > 0) {
                                currentPosition.bck(axisID);
                            } else {
                                // now either we have to go to the next level or we are done
                                if (currentPosition.getIntPosition(axisID + 1) < currentPosition.dimension(axisID + 1) - 1) {
                                    fwd = true;
                                    currentPosition.fwd(axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else if (currentPosition.getIntPosition(axisID + 2) < currentPosition.dimension(axisID + 2) - 1) {
                                    fwd = true;
                                    // next next dimension
                                    currentPosition.fwd(axisID + 2);
                                    currentPosition.setPosition(0, axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            // Because it would be a single image axis otherwise
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 1:
                            if (currentPosition.getIntPosition(axisID) > 0) {
                                currentPosition.bck(axisID);
                            } else {
                                // now either we have to go to the next level or we are done
                                if (currentPosition.getIntPosition(axisID + 1) < currentPosition.dimension(axisID + 1) - 1) {
                                    fwd = true;
                                    currentPosition.fwd(axisID + 1);
                                    currentPosition.setPosition(pos.getIntPosition(axisID), axisID);
                                    if (img.getImg() instanceof PlanarImg) {
                                        if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                                            throw new RuntimeException("Image plane index out of range.");
                                        }
                                        // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
                                        target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
                                        if (target.targetArray == null) {
                                            throw new RuntimeException("Could not get image plane at specified index.");
                                        }
                                    }
                                    if (currentPosition.getIntPosition(axisID) < currentPosition.dimension(axisID) - 1) {
                                        currentPosition.fwd(axisID);
                                    } else {
                                        // reverse or done
                                        if (pos.getIntPosition(axisID) > 0) {
                                            // reverse
                                            fwd = false;
                                            // no need to get the image again 
                                            currentPosition.bck(axisID);
                                        } else {
                                            currentAlignmentCount = 0;
                                            return true;
                                        }
                                    }
                                } else {
                                    currentAlignmentCount = 0;
                                    return true;
                                }
                            }
                            break;
                        case 2:
                            if (currentPosition.getIntPosition(axisID) > 0) {
                                currentPosition.bck(axisID);
                            } else {
                                currentAlignmentCount = 0;
                                return true;
                            }
                            break;
                    }
                    break;
            }
        }
        switch (currentPosition.numDimensions()) {
            case 2:
                // Z or C
                switch (axisID) {
                    case 0:
                        target.transformation = transformations[0][currentPosition.getIntPosition(1)][currentPosition.getIntPosition(0)];
                        break;
                    case 1:
                        target.transformation = transformations[0][currentPosition.getIntPosition(1)][0];
                        break;
                }
                break;
            case 3:
                // F/T or Z or C
                switch (axisID) {
                    case 0:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][currentPosition.getIntPosition(1)][currentPosition.getIntPosition(0)];
                        break;
                    case 1:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][currentPosition.getIntPosition(1)][0];
                        break;
                    case 2:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][0][0];
                        break;
                }
                break;
        }
        if (img.getImg() instanceof PlanarImg) {
            if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                throw new RuntimeException("Image plane index out of range.");
            }
            // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
            target.sourceArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
            if (target.sourceArray == null) {
                throw new RuntimeException("Could not get image plane at specified index.");
            }
        }
        return false;
    }
    
    @Override
    synchronized boolean getTransformationForCurrentPosition(final SharedContextAlignmentTarget target)
    {
        if(finishedTransformations) {
            statusService.showStatus(nrOfTransformations, nrOfTransformations, "Transforming");
            return true;
        }
        statusService.showStatus(currentTransformationCount++, nrOfTransformations, "Transforming");
        // No skipping of the reference images
        if (img.getImg() instanceof PlanarImg) {
            if (currentPosition.getIndex() > Integer.MAX_VALUE) {
                throw new RuntimeException("Image plane index out of range.");
            }
            // Grab the current array (this will give us a stable reference to the array so we can leave the sync)
            target.targetArray = ((PlanarImg) img.getImg()).getPlane((int) currentPosition.getIndex()).getCurrentStorageArray();
            if (target.targetArray == null) {
                throw new RuntimeException("Could not get image plane at specified index.");
            }
        }
        // grab the current position as index to the transformation array
        switch(currentPosition.numDimensions())
        {
            case 2:
                switch(axisID)
                {
                    case 0:
                        target.transformation = transformations[0][currentPosition.getIntPosition(1)][currentPosition.getIntPosition(0)];
                        break;
                    case 1:
                        target.transformation = transformations[0][currentPosition.getIntPosition(1)][0];
                        break;
                }
                break;
            case 3:
                switch(axisID)
                {
                    case 0:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][currentPosition.getIntPosition(1)][currentPosition.getIntPosition(0)];
                        break;
                    case 1:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][currentPosition.getIntPosition(1)][0];
                        break;
                    case 2:
                        target.transformation = transformations[currentPosition.getIntPosition(2)][0][0];
                        break;
                }
                break;
        }
        if (currentPosition.hasNext()) {
            currentPosition.fwd();
        } else {
            finishedTransformations = true;
        }
        return false;
    }
    
    @Override
    public void serializeTransformations(final PrintWriter out)
    {
        JSONObject root = new JSONObject();
        JSONObject extents = new JSONObject();
        JSONArray transformationArray = new JSONArray();
        int dims = img.numDimensions();

        for(int i = 2;i < dims;i++)
        {
            if("Channel".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel())) {
                extents.put("Channel", img.dimension(i));
            }
            else if("Z".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel()))  {
                extents.put("Z", img.dimension(i));
            }
            else if("Time".equals(((DefaultLinearAxis)img.axis(i)).type().getLabel())) {
                extents.put("Time", img.dimension(i));
            }
        }
        root.put("Extents", extents);
        root.put("AlignmentAxis", ((DefaultLinearAxis)img.axis(axisID + 2)).type().getLabel());
        switch(dims - 2) {
        case 2:
            // for simplicities sake create a transformation for each image (this may lead to duplications)
            if(axisID == 0) {
                // No propagated transformations
                for(int i = 0; i < transformationExtents[2];i++)
                {
                    for(int j = 0; j < transformationExtents[1];j++)
                    {
                        JSONArray subTransformation = new JSONArray();
                        for(int k = 0;k < transformationExtents[0];k++)
                        {
                            subTransformation.put(transformations[i][j][k].serialize());
                        }
                        transformationArray.put(subTransformation);
                    }
                }
            } else {
                // There are propagated transformations
                for(int i = 0; i < transformationExtents[2];i++)
                {
                    for(int j = 0; j < transformationExtents[1];j++)
                    {
                        JSONArray subTransformation = new JSONArray();
                        for(int k = 0;k < (int)img.dimension(2);k++)
                        {
                            subTransformation.put(transformations[i][j][0].serialize());
                        }
                        transformationArray.put(subTransformation);
                    }
                }
            }
            break;
        case 3:
            // for simplicities sake create a transformation for each image (this may lead to duplications)
            switch(axisID) {
            case 0:
                // No propagated transformations
                for(int i = 0; i < transformationExtents[2];i++)
                {
                    JSONArray subTransformation = new JSONArray();
                    for(int j = 0; j < transformationExtents[1];j++)
                    {
                        JSONArray subSubTransformation = new JSONArray();
                        for(int k = 0;k < transformationExtents[0];k++)
                        {
                            subSubTransformation.put(transformations[i][j][k].serialize());
                        }
                        subTransformation.put(subSubTransformation);
                    }
                    transformationArray.put(subTransformation);
                }
                break;
            case 1:
                for(int i = 0; i < transformationExtents[2];i++)
                {
                    JSONArray subTransformation = new JSONArray();
                    for(int j = 0; j < transformationExtents[1];j++)
                    {
                        JSONArray subSubTransformation = new JSONArray();
                        for(int k = 0;k < (int)img.dimension(2);k++)
                        {
                            subSubTransformation.put(transformations[i][j][0].serialize());
                        }
                        subTransformation.put(subSubTransformation);
                    }
                    transformationArray.put(subTransformation);
                }
                break;
            case 2:
                for(int i = 0; i < transformationExtents[2];i++)
                {
                    JSONArray subTransformation = new JSONArray();
                    for(int j = 0; j < (int)img.dimension(3);j++)
                    {
                        JSONArray subSubTransformation = new JSONArray();
                        for(int k = 0;k < (int)img.dimension(2);k++)
                        {
                            subSubTransformation.put(transformations[i][0][0].serialize());
                        }
                        subTransformation.put(subSubTransformation);
                    }
                    transformationArray.put(subTransformation);
                }
                break;
            }
            break;
        }
        root.put("Tranformations", transformationArray);
        out.write(root.toString());
    }
}
