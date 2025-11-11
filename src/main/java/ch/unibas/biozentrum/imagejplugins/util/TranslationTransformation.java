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

import ch.unibas.biozentrum.imagejplugins.abstracts.Transformation;
import org.json.JSONObject;

/**
 * @author Peter D. Ringel
 * @version 1.0.0
 *
 */
public class TranslationTransformation implements Transformation {
    public double offsetx = 0.0;
    public double offsety = 0.0;

    @Override
    public void transformWith(Transformation t) {
        if(t instanceof TranslationTransformation)
        {
            // Combining a translation transformation is simply adding the translations
            offsetx += ((TranslationTransformation) t).offsetx;
            offsety += ((TranslationTransformation) t).offsety;
        }
        else
        {
            throw new RuntimeException("RigidBodyTransformation cannot be transformed with a different transformation type.");
        }
    }

    @Override
    public void invert() {
        offsetx = -offsetx;
        offsety = -offsety;
    }

    @Override
    public JSONObject serialize() {
    	//TODO:
    	//FIXME: The transformations are applied to the access vector and thus the images are transformed inversely, this should be reflected in the output
        JSONObject retval = new JSONObject();
        JSONObject transformation = new JSONObject();
        // Save doubles as string, because JSON does not officially support double precision
        transformation.put("OffsetX", Double.toString(offsetx));
        transformation.put("OffsetY", Double.toString(offsety));
        retval.put("Translation", transformation);
        return retval;
    }
    
    @Override
    public Transformation copy()
    {
    	TranslationTransformation retval = new TranslationTransformation();
    	retval.offsetx = offsetx;
    	retval.offsety = offsety;
    	return retval;
    }
    
    @Override
    public Square transform(Square square)
    {
        //x1, y1 = 0,0
        square.x1 = square.x3 = -offsetx;
        square.y1 = square.y2 = -offsety;
        //square x2 = width; y2 = 0 
        square.x2 -= offsetx;
        //square x3 = 0; y3 = height
        square.y3 -= offsety;
        //square x4 = width; y4 = height
        square.x4 -= offsetx;
        square.y4 -= offsety;
        return square;
    }
    
    @Override
    public void translate(final double offsetx, final double offsety)
    {
    	this.offsetx += offsetx;
    	this.offsety += offsety;
    }
}
