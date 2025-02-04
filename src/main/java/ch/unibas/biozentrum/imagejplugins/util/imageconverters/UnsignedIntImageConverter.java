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
public class UnsignedIntImageConverter implements ImageConverter {
    @Override
    public void convertTo(Object arr, double[] target)
    {
        final int[] larr = (int[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedIntImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            if(larr[i] < (int)0)
            {
                target[i] = ((double)larr[i] + 4294967296.0) / 4294967295.0;
            }
            else
            {
                target[i] = ((double)larr[i]) / 4294967295.0;
            }
        }
    }
    
    @Override
    public void convertTo(final Object arr, final DoubleBuffer target)
    {
        final int[] larr = (int[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            if(larr[i] < (int)0)
            {
                target.put((((double)larr[i] + 4294967296.0) / 4294967295.0));
            }
            else
            {
                target.put((((double)larr[i]) / 4294967295.0));
            }
        }
        target.rewind();
    }
    
    @Override
    public void deConvertTo(double[] target, Object arr) {
        final int[] larr = (int[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target[i] * 4294967295.0;
            if(currentValue < 0.0)
            {
                larr[i] = 0;
            }
            else if(currentValue > 4294967295.0)
            {
                larr[i] = -1;
            }
            else
            {
                larr[i] = (int)(currentValue + 0.5); // because target[i] is never negative this is rounding by truncation
            }
        }
    }

    @Override
    public void deConvertTo(DoubleBuffer target, Object arr) {
        final int[] larr = (int[])arr;
        if(larr.length != target.limit())
        {
            throw new RuntimeException("The arrays passed to the ByteImageConverter are not of the same size.");
        }
        target.rewind();
        for(int i = 0; i < larr.length;i++)
        {
            double currentValue = target.get(i) * 4294967295.0;
            if(currentValue < 0.0)
            {
                larr[i] = 0;
            }
            else if(currentValue > 4294967295.0)
            {
                larr[i] = -1;
            }
            else
            {
                larr[i] = (int)(currentValue + 0.5);// because target[i] is never negative this is rounding by truncation
            }
        }
        target.rewind();
    }

    @Override
    public void convertTo(Object arr, float[] target) {
        final int[] larr = (int[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedIntImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            if(larr[i] < (int)0)
            {
                target[i] = ((float)larr[i] + 4294967296.0f) / 4294967295.0f;
            }
            else
            {
                target[i] = ((float)larr[i]) / 4294967295.0f;
            }
        }
    }

    @Override
    public void deConvertTo(float[] target, Object arr) {
        final int[] larr = (int[])arr;
        if(larr.length != target.length)
        {
            throw new RuntimeException("The arrays passed to the UnsignedShortImageConverter are not of the same size.");
        }
        for(int i = 0; i < larr.length;i++)
        {
            float currentValue = target[i] * 4294967295.0f;
            if(currentValue < 0.0f)
            {
                larr[i] = 0;
            }
            else if(currentValue > 4294967295.0f)
            {
                larr[i] = -1;
            }
            else
            {
                larr[i] = (int)(currentValue + 0.5f); // because target[i] is never negative this is rounding by truncation
            }
        }
    }
}
