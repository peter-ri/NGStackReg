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

package ch.unibas.biozentrum.imagejplugins.util;

import ch.unibas.biozentrum.imagejplugins.util.imageconverters.ByteImageConverter;
import net.imagej.ImgPlus;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import ch.unibas.biozentrum.imagejplugins.abstracts.ImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.DoubleImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.FloatImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.IntImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.LongImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.ShortImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedByteImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedIntImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedLongImageConverter;
import ch.unibas.biozentrum.imagejplugins.util.imageconverters.UnsignedShortImageConverter;
/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public class ImageConverterFactory {
    public static ImageConverter getImageConverter(final ImgPlus img)
    {
        if(img.firstElement() instanceof ByteType)
        {
            return (ImageConverter)new ByteImageConverter();
        }
        else if(img.firstElement() instanceof UnsignedByteType)
        {
            return (ImageConverter)new UnsignedByteImageConverter();
        }
        else if(img.firstElement() instanceof ShortType)
        {
            return (ImageConverter)new ShortImageConverter();
        }
        else if(img.firstElement() instanceof UnsignedShortType)
        {
            return (ImageConverter)new UnsignedShortImageConverter();
        }
        else if(img.firstElement() instanceof IntType)
        {
            return (ImageConverter)new IntImageConverter();
        }
        else if(img.firstElement() instanceof UnsignedIntType)
        {
            return (ImageConverter)new UnsignedIntImageConverter();
        }
        else if(img.firstElement() instanceof LongType)
        {
            return (ImageConverter)new LongImageConverter();
        }
        else if(img.firstElement() instanceof UnsignedLongType)
        {
            return (ImageConverter)new UnsignedLongImageConverter();
        }
        else if(img.firstElement() instanceof FloatType)
        {
            return (ImageConverter)new FloatImageConverter();
        }
        else if(img.firstElement() instanceof DoubleType)
        {
            return (ImageConverter)new DoubleImageConverter();
        }
        else
        {
            throw new RuntimeException("The ImageConverterFactory does not support the requested image type.");
        }
    }
    
}
