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
public class RigidBodyTransformation implements Transformation {
    public double offsetx = 0.0;
    public double offsety = 0.0;
    public double angle = 0.0;

    @Override
    public void transformWith(Transformation t) {
        if(t instanceof RigidBodyTransformation)
        {
            /*
            Combining a rigid body transformation is simply adding the rotations
            and transforming and adding the transformed offsets
            */
            angle += ((RigidBodyTransformation) t).angle;
            double s = Math.sin(((RigidBodyTransformation) t).angle);
            double c = Math.cos(((RigidBodyTransformation) t).angle);
            double tmpoffsetx = offsetx;
            offsetx = c * offsetx - s * offsety + ((RigidBodyTransformation) t).offsetx;
            offsety = s * tmpoffsetx + c * offsety + ((RigidBodyTransformation) t).offsety;
        }
        else
        {
            throw new RuntimeException("RigidBodyTransformation cannot be transformed with a different transformation type.");
        }
    }

    @Override
    public void invert() {
        double s = Math.sin(angle);
        double c = Math.cos(angle);
        double tmpoffsetx = offsetx;
        offsetx = -c * offsetx + s * offsety;
        offsety = -c * offsety - s * tmpoffsetx;
        angle = -angle;
    }

    @Override
    public JSONObject serialize() {
        JSONObject retval = new JSONObject();
        JSONObject transformation = new JSONObject();
        // Save doubles as string, because JSON does not officially support double precision
        transformation.put("Rotation", Double.toString(angle));
        transformation.put("OffsetX", Double.toString(offsetx));
        transformation.put("OffsetY", Double.toString(offsety));
        retval.put("RigidBody", transformation);
        return retval;
    }
    
    @Override
    public Transformation copy()
    {
    	RigidBodyTransformation retval = new RigidBodyTransformation();
    	retval.offsetx = offsetx;
    	retval.offsety = offsety;
    	retval.angle = angle;
    	return retval;
    }
}
