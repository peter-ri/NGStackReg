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

package ch.unibas.biozentrum.imagejplugins.abstracts;

import ch.unibas.biozentrum.imagejplugins.implementations.AbstractSharedContext;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public abstract class RegistrationAndTransformation {
    protected final AbstractSharedContext sharedContext;

    protected RegistrationAndTransformation(final AbstractSharedContext sharedContext)
    {
        this.sharedContext = sharedContext;
    }

    /**
     * Must be called to release the allocated resources, especially system
     * resources allocated outside the reach of java VM.
     */
    public abstract void release();

    /**
     * Calculates the image transformations.
     * <p>
     * The implementations of this method calculate the image transformations
     * and store the transformations in the transformations array.
     * Usually this will start worker threads processing separate image
     * transformations.
     * 
     * @throws InterruptedException
     */
    public abstract void register()  throws InterruptedException, Exception;
    
    public abstract void waitForFinish()  throws InterruptedException;
}
