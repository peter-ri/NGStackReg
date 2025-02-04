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
public class ByteImageConverter implements ImageConverter {

    @Override
    public void convertTo(final Object arr, final double[] target)
    {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            target[i] = ((double)larr[i]/((double)Byte.MAX_VALUE));
        }
    }
    @Override
    public void convertTo(final Object arr, final DoubleBuffer target)
    {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            target.put(((double)larr[i]/((double)Byte.MAX_VALUE)));
        }
        target.rewind();
    }
    
    @Override
    public void deConvertTo(double[] target, Object arr) {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target[i] * ((double)Byte.MAX_VALUE);
            if(currentValue < ((double)Byte.MIN_VALUE))
            {
                larr[i] = Byte.MIN_VALUE;
            }
            else if(currentValue > ((double)Byte.MAX_VALUE))
            {
                larr[i] = Byte.MAX_VALUE;
            }
            else
            {
                larr[i] = (byte)Math.round(currentValue);
            }
        }
    }

    @Override
    public void deConvertTo(DoubleBuffer target, Object arr) {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target.get(i) * ((double)Byte.MAX_VALUE);
            if(currentValue < ((double)Byte.MIN_VALUE))
            {
                larr[i] = Byte.MIN_VALUE;
            }
            else if(currentValue > ((double)Byte.MAX_VALUE))
            {
                larr[i] = Byte.MAX_VALUE;
            }
            else
            {
                larr[i] = (byte)Math.round(currentValue);
            }
        }
        target.rewind();
    }

    @Override
    public void convertTo(Object arr, float[] target) {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            target[i] = ((float)larr[i]/((float)Byte.MAX_VALUE));
        }
    }

    @Override
    public void deConvertTo(float[] target, Object arr) {
        final byte[] larr = (byte[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            float currentValue = target[i] * ((float)Byte.MAX_VALUE);
            if(currentValue < ((float)Byte.MIN_VALUE))
            {
                larr[i] = Byte.MIN_VALUE;
            }
            else if(currentValue > ((float)Byte.MAX_VALUE))
            {
                larr[i] = Byte.MAX_VALUE;
            }
            else
            {
                larr[i] = (byte)Math.round(currentValue);
            }
        }
    }
}
