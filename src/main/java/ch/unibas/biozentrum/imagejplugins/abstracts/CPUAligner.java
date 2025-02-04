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
public abstract class CPUAligner implements Runnable {
    protected final ImageConverter converter;
    protected final int pyramidDepth;
    protected final AbstractSharedContext sharedContext;
    private Thread t = null;
    protected CPUAligner(final AbstractSharedContext sharedContext, final ImageConverter converter, final int pyramidDepth)
    {
        this.converter = converter;
        this.pyramidDepth = pyramidDepth;
        this.sharedContext = sharedContext;
    }
    public void start() {
        t = new Thread(this);
        t.start();
    }
    public Thread getThread() {
        return t;
    }
    public void release()
    {
    	
    }
}
