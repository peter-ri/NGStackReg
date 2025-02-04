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

import java.nio.DoubleBuffer;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public interface ImageConverter {
    public void convertTo(final Object arr, final double[] target);
    public void convertTo(final Object arr, final float[] target);
    public void convertTo(final Object arr, final DoubleBuffer target);
    public void deConvertTo(final double[] target, final Object arr);
    public void deConvertTo(final float[] target, final Object arr);
    public void deConvertTo(final DoubleBuffer target, final Object arr);
}
