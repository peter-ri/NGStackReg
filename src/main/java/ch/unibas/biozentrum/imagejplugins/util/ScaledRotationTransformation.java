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
public class ScaledRotationTransformation implements Transformation {
    public double offsetx = 0.0;
    public double offsety = 0.0;
    public double angle = 0.0;
    public double scale = 1.0;

    @Override
    public void transformWith(Transformation t) {
        if(t instanceof ScaledRotationTransformation)
        {
            angle += ((ScaledRotationTransformation) t).angle;
            scale *= ((ScaledRotationTransformation) t).scale;
            double s = Math.sin(-((ScaledRotationTransformation) t).angle);
            double c = Math.cos(-((ScaledRotationTransformation) t).angle);
            double tmpoffsetx = offsetx;
            offsetx = (c * offsetx - s * offsety) * ((ScaledRotationTransformation) t).scale + ((ScaledRotationTransformation) t).offsetx;
            offsety = (s * tmpoffsetx + c * offsety) * ((ScaledRotationTransformation) t).scale + ((ScaledRotationTransformation) t).offsety;
        }
        else
        {
            throw new RuntimeException("ScaledRotationTransformation cannot be transformed with a different transformation type.");
        }
    }

    @Override
    public void invert() {
    	double s = Math.sin(-angle);
        double c = Math.cos(-angle);
        double tmpoffsetx = offsetx;
        //The scale is already inverted
        offsetx = (-c * offsetx - s * offsety) / scale;
        offsety = (s * tmpoffsetx - c * offsety) / scale;
        angle = -angle;
        scale = 1.0/scale;
    }

    @Override
    public JSONObject serialize() {
    	//TODO:
    	//FIXME: The transformations are applied to the access vector and thus the images are transformed inversely, this should be reflected in the output
        JSONObject retval = new JSONObject();
        JSONObject transformation = new JSONObject();
        // Save doubles as string, because JSON does not officially support double precision
        transformation.put("Scale", Double.toString(scale));
        transformation.put("Rotation", Double.toString(angle));
        transformation.put("OffsetX", Double.toString(offsetx));
        transformation.put("OffsetY", Double.toString(offsety));
        retval.put("ScaledRotation", transformation);
        return retval;
    }
    
    @Override
    public Transformation copy()
    {
    	ScaledRotationTransformation retval = new ScaledRotationTransformation();
    	retval.offsetx = offsetx;
    	retval.offsety = offsety;
    	retval.angle = angle;
    	retval.scale = scale;
    	return retval;
    }
    
    @Override
	public Square transform(Square square)
    {
    	ScaledRotationTransformation cp = (ScaledRotationTransformation)copy();
    	cp.invert();
    	/*
    	 * After invert(): cp.angle = -angle_orig, cp.scale = 1/scale_orig,
    	 * cp.offsetx/y = t_inv
    	 *
    	 * Forward map: output = (1/scale_orig) * R(-angle_orig) * source + t_inv
    	 *
    	 * sin(-cp.angle) = sin(angle_orig), cos(-cp.angle) = cos(angle_orig)
    	 * so s = sin(angle_orig), c = cos(angle_orig).
    	 *
    	 * (1/scale_orig) = cp.scale  ->  multiply by cp.scale
    	 *
    	 * output_x = (c*sx - s*sy) * cp.scale + cp.offsetx
    	 * output_y = (s*sx + c*sy) * cp.scale + cp.offsety
    	 */
    	double s = Math.sin(-cp.angle);
        double c = Math.cos(-cp.angle);
        //x1, y1 = 0,0
        square.x1 = cp.offsetx;
        square.y1 = cp.offsety;
        //square x2 = width; y2 = 0 
        double tmp = square.x2;
        square.x2 = tmp * c * cp.scale + cp.offsetx;
        square.y2 = tmp * s * cp.scale + cp.offsety;
        //square x3 = 0; y3 = height
        square.x3 = cp.offsetx - square.y3 * s * cp.scale;
        square.y3 = square.y3 * c * cp.scale + cp.offsety;
        //square x4 = width; y4 = height
        tmp = square.x4;
        square.x4 = (tmp * c - square.y4 * s) * cp.scale + cp.offsetx;
        square.y4 = (tmp * s + square.y4 * c) * cp.scale + cp.offsety;
        return square;
    }
    
    @Override
    public void translate(final double offsetx, final double offsety)
    {
    	/*
         * The canvas origin is shifted by (offsetx, offsety) in output image space.
         * New access-vector offset: t_new = t + A·(dx,dy)
         * where A = scale * R(angle).
         *
         * Using s = sin(-angle) = -sin(angle), c = cos(-angle) = cos(angle)
         *   R(angle) applied to (dx,dy):
         *     x-component: cos(angle)·dx + sin(angle)·dy  = c·dx - s·dy
         *     y-component: -sin(angle)·dx + cos(angle)·dy = s·dx + c·dy
         *
         * So:
         *   t_new_x = this.offsetx + scale * (c*dx - s*dy)
         *   t_new_y = this.offsety + scale * (s*dx + c*dy)
         */
        double s = Math.sin(-angle);
        double c = Math.cos(-angle);
        this.offsetx += scale * (c * offsetx - s * offsety);
        this.offsety += scale * (s * offsetx + c * offsety);
    }
}
