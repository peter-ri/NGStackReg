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

import java.io.PrintWriter;
import java.util.concurrent.CyclicBarrier;

import ch.unibas.biozentrum.imagejplugins.NGStackReg;
import net.imagej.ImgPlus;

/*
 * Must be in this package, because methods cannot be internal to another package
 */

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public abstract class AbstractSharedContext
{
	NGStackReg.TransformationType transformationType;
    @SuppressWarnings("rawtypes")
	ImgPlus img;
    CyclicBarrier workerSynchronizationBarrier = null;
    int nrOfGPUDevices = 0;//used later to determine the number of GPU feeding cores (yes the GPUs are hungry)
    boolean forceDoublePrecisionRepr;
    boolean finishedTransformations;//This will be used to stop the image transformation infinite loops so don't mess around with the variable
    
	public abstract int getNrOfGPUDevices();
	abstract void addParties(int participants);
	abstract boolean getNextAlignmentTarget(final SharedContextAlignmentTarget target);
	abstract boolean getTransformationForCurrentPosition(final SharedContextAlignmentTarget target);
	public abstract void serializeTransformations(final PrintWriter out);
}