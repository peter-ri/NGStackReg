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
public class ShortImageConverter implements ImageConverter {
    @Override
    public void convertTo(final Object arr, final double[] target)
    {
        final short[] larr = (short[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the ShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            target[i] = ((double)larr[i]/((double)Short.MAX_VALUE));
        }
    }
    @Override
    public void convertTo(final Object arr, final DoubleBuffer target)
    {
        final short[] larr = (short[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            target.put(((double)larr[i]/((double)Short.MAX_VALUE)));
        }
        target.rewind();
    }
    
    @Override
    public void deConvertTo(double[] target, Object arr) {
        final short[] larr = (short[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target[i] * ((double)Short.MAX_VALUE);
            if(currentValue < ((double)Short.MIN_VALUE))
            {
                larr[i] = Short.MIN_VALUE;
            }
            else if(currentValue > ((double)Short.MAX_VALUE))
            {
                larr[i] = Short.MAX_VALUE;
            }
            else
            {
                larr[i] = (short)Math.round(currentValue);
            }
        }
    }

    @Override
    public void deConvertTo(DoubleBuffer target, Object arr) {
        final short[] larr = (short[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target.get(i) * ((double)Short.MAX_VALUE);
            if(currentValue < ((double)Short.MIN_VALUE))
            {
                larr[i] = Short.MIN_VALUE;
            }
            else if(currentValue > ((double)Short.MAX_VALUE))
            {
                larr[i] = Short.MAX_VALUE;
            }
            else
            {
                larr[i] = (short)Math.round(currentValue);
            }
        }
        target.rewind();
    }

    @Override
    public void convertTo(Object arr, float[] target) {
        final short[] larr = (short[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the ShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            target[i] = ((float)larr[i]/((float)Short.MAX_VALUE));
        }
    }

    @Override
    public void deConvertTo(float[] target, Object arr) {
        final short[] larr = (short[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            float currentValue = target[i] * ((float)Short.MAX_VALUE);
            if(currentValue < ((float)Short.MIN_VALUE))
            {
                larr[i] = Short.MIN_VALUE;
            }
            else if(currentValue > ((float)Short.MAX_VALUE))
            {
                larr[i] = Short.MAX_VALUE;
            }
            else
            {
                larr[i] = (short)Math.round(currentValue);
            }
        }
    }
}
