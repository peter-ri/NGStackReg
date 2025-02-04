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

package ch.unibas.biozentrum.imagejplugins.util.imageconverters;

import ch.unibas.biozentrum.imagejplugins.abstracts.ImageConverter;
import java.nio.DoubleBuffer;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public class FloatImageConverter implements ImageConverter {
    @Override
    public void convertTo(final Object arr, final double[] target)
    {
        final float[] larr = (float[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the FloatImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            target[i] = (double)larr[i];
        }
    }
    @Override
    public void convertTo(final Object arr, final DoubleBuffer target)
    {
        final float[] larr = (float[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            target.put((double)larr[i]);
        }
        target.rewind();
    }

    @Override
    public void deConvertTo(double[] target, Object arr) {
        final float[] larr = (float[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the FloatImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            larr[i] = (float)target[i];
        }
    }

    @Override
    public void deConvertTo(DoubleBuffer target, Object arr) {
        final float[] larr = (float[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            larr[i] = (float)target.get();
        }
        target.rewind();
    }

    @Override
    public void convertTo(Object arr, float[] target) {
        final float[] larr = (float[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the DoubleImageConverter are not of the same size.");
        }
        System.arraycopy(larr, 0, target, 0, larr.length);
    }

    @Override
    public void deConvertTo(float[] target, Object arr) {
        final float[] larr = (float[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the DoubleImageConverter are not of the same size.");
        }
        System.arraycopy(target, 0, larr, 0, larr.length);
    }
}
