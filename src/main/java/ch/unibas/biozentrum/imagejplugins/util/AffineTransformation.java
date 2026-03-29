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
public class AffineTransformation implements Transformation {
    public double offsetx = 0.0;
    public double offsety = 0.0;
    public double a11 = 1.0;
    public double a12 = 0.0;
    public double a21 = 0.0;
    public double a22 = 1.0;

    @Override
    public void transformWith(Transformation t) {
        if(t instanceof AffineTransformation)
        {
        	AffineTransformation lt = (AffineTransformation)t;
            double la11 = (a11 * lt.a11) + (a21 * lt.a12);
            double la12 = (a12 * lt.a11) + (a22 * lt.a12);
            double loffsetx = (offsetx * lt.a11) + (offsety * lt.a12) + lt.offsetx;
            double la21 = (a11 * lt.a21) + (a21 * lt.a22);
            double la22 = (a12 * lt.a21) + (a22 * lt.a22);
            double loffsety = (offsetx * lt.a21) + (offsety * lt.a22) + lt.offsety;
            a11 = la11;
            a12 = la12;
            a21 = la21;
            a22 = la22;
            offsetx = loffsetx;
            offsety = loffsety;
        }
        else
        {
            throw new RuntimeException("AffineTransformation cannot be transformed with a different transformation type.");
        }
    }

    @Override
    public void invert() {
    	double a11xa22 = a11 * a22;
    	double a12xa21 = a12 * a21;
    	double d1 = a11xa22 - a12xa21;
    	double la11 = a22 / d1;
    	double la12 = a12 / (-d1);
    	double loffsetx = ((a12 * offsety) - (a22 * offsetx)) / d1;
    	double la21 = a21 / (-d1);
    	double la22 = a11 / d1;
    	double loffsety = ((a11 * offsety) - (a21 * offsetx)) / (-d1);
    	a11 = la11;
    	a12 = la12;
    	offsetx = loffsetx;
    	a21 = la21;
    	a22 = la22;
    	offsety = loffsety;
    }

    @Override
    public JSONObject serialize() {
    	//TODO:
    	//FIXME: The transformations are applied to the access vector and thus the images are transformed inversely, this should be reflected in the output
        JSONObject retval = new JSONObject();
        JSONObject transformation = new JSONObject();
        // Save doubles as string, because JSON does not officially support double precision
        transformation.put("a11", Double.toString(a11));
        transformation.put("a12", Double.toString(a12));
        transformation.put("a21", Double.toString(a21));
        transformation.put("a22", Double.toString(a22));
        transformation.put("OffsetX", Double.toString(offsetx));
        transformation.put("OffsetY", Double.toString(offsety));
        retval.put("Affine", transformation);
        return retval;
    }
    
    @Override
    public Transformation copy()
    {
    	AffineTransformation retval = new AffineTransformation();
    	retval.offsetx = offsetx;
    	retval.offsety = offsety;
    	retval.a11 = a11;
    	retval.a12 = a12;
    	retval.a21 = a21;
    	retval.a22 = a22;
    	return retval;
    }
    
    @Override
	public Square transform(Square square)
    {
    	/*AffineTransformation cp = (AffineTransformation)copy();
    	cp.invert();
    	//(0,0)
    	square.x1 = cp.offsetx;
    	square.y1 = cp.offsety;
    	//(width, 0)
    	double lx = (cp.a11 * square.x2) + cp.offsetx;
    	double ly = (cp.a21 * square.x2) + cp.offsety;
    	square.x2 = lx;
    	square.y2 = ly;
    	//(0, height)
    	lx = (cp.a12 * square.y3) + cp.offsetx;
    	ly = (cp.a22 * square.y3) + cp.offsety;
    	square.x3 = lx;
    	square.y3 = ly;
    	//(width, height)
    	lx = (cp.a11 * square.x4) + (cp.a12 * square.y4) + cp.offsetx;
    	ly = (cp.a21 * square.x4) + (cp.a22 * square.y4) + cp.offsety;
    	square.x3 = lx;
    	square.y3 = ly;
        return square;*/
    	
    	AffineTransformation cp = (AffineTransformation)copy();
    	cp.invert();
    	//(0,0)
    	square.x1 = cp.offsetx;
    	square.y1 = cp.offsety;
    	//(width, 0)
    	double lx = (cp.a11 * square.x2) + cp.offsetx;
    	double ly = (cp.a21 * square.x2) + cp.offsety;
    	square.x2 = lx;
    	square.y2 = ly;
    	//(0, height)
    	lx = (cp.a12 * square.y3) + cp.offsetx;
    	ly = (cp.a22 * square.y3) + cp.offsety;
    	square.x3 = lx;
    	square.y3 = ly;
    	//(width, height) — fixed: was writing to x3/y3 (copy-paste bug), must write to x4/y4
    	lx = (cp.a11 * square.x4) + (cp.a12 * square.y4) + cp.offsetx;
    	ly = (cp.a21 * square.x4) + (cp.a22 * square.y4) + cp.offsety;
    	square.x4 = lx;
    	square.y4 = ly;
        return square;
    }
    
    @Override
    public void translate(final double offsetx, final double offsety)
    {
    	/*this.offsetx += offsetx;
    	this.offsety += offsety;*/
    	
    	/*
         * The canvas origin is shifted by (offsetx, offsety) in output image space.
         * New access-vector offset: t_new = t + A·(dx,dy)
         * where A = | a11  a12 |
         *           | a21  a22 |
         *
         * t_new_x = this.offsetx + a11*dx + a12*dy
         * t_new_y = this.offsety + a21*dx + a22*dy
         */
    	this.offsetx += a11 * offsetx + a12 * offsety;
    	this.offsety += a21 * offsetx + a22 * offsety;
    }
}
