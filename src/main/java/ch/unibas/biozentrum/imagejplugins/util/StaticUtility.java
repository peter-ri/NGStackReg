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

public final class StaticUtility {
	private StaticUtility() { }
	
	public static int roundUp(long maxGroupSize, long SIMDwidth, long globalSize) {
        int localSize =(int)((maxGroupSize % SIMDwidth) == 0 ? maxGroupSize : maxGroupSize - (maxGroupSize % SIMDwidth)); // round down to the nearest SIMDwidth multiple
        return (int)((globalSize % localSize) == 0 ? globalSize : globalSize + localSize - (globalSize % localSize));
    }
	
	// TODO: potentially use a math library function instead of including the
    // copyrighted function here
    // The copyright of this function is attributed to the original authors
    // Th&eacute;venaz P, Ruttimann UE &amp; Unser M (1998) A pyramid approach to
    // subpixel registration based on intensity. <i>IEEE transactions on image
    // processing : a publication of the IEEE Signal Processing Society</i>
    // <b>7</b>:27–41 (DOI: <a href="http://dx.doi.org/10.1109/83.650848">10.1109/83.650848</a>)
    // it is used here as part of creating a derivative work under the GPLv3
	public static void invertGauss (final double[][] matrix) {
        final int n = matrix.length;
        final double[][] inverse = new double[n][n];
        for (int i = 0; (i < n); i++) {
                double max = matrix[i][0];
                double absMax = Math.abs(max);
                for (int j = 0; (j < n); j++) {
                        inverse[i][j] = 0.0;
                        if (absMax < Math.abs(matrix[i][j])) {
                                max = matrix[i][j];
                                absMax = Math.abs(max);
                        }
                }
                inverse[i][i] = 1.0 / max;
                for (int j = 0; (j < n); j++) {
                        matrix[i][j] /= max;
                }
        }
        for (int j = 0; (j < n); j++) {
                double max = matrix[j][j];
                double absMax = Math.abs(max);
                int k = j;
                for (int i = j + 1; (i < n); i++) {
                        if (absMax < Math.abs(matrix[i][j])) {
                                max = matrix[i][j];
                                absMax = Math.abs(max);
                                k = i;
                        }
                }
                if (k != j) {
                        final double[] partialLine = new double[n - j];
                        final double[] fullLine = new double[n];
                        System.arraycopy(matrix[j], j, partialLine, 0, n - j);
                        System.arraycopy(matrix[k], j, matrix[j], j, n - j);
                        System.arraycopy(partialLine, 0, matrix[k], j, n - j);
                        System.arraycopy(inverse[j], 0, fullLine, 0, n);
                        System.arraycopy(inverse[k], 0, inverse[j], 0, n);
                        System.arraycopy(fullLine, 0, inverse[k], 0, n);
                }
                for (k = 0; (k <= j); k++) {
                        inverse[j][k] /= max;
                }
                for (k = j + 1; (k < n); k++) {
                        matrix[j][k] /= max;
                        inverse[j][k] /= max;
                }
                for (int i = j + 1; (i < n); i++) {
                        for (k = 0; (k <= j); k++) {
                                inverse[i][k] -= matrix[i][j] * inverse[j][k];
                        }
                        for (k = j + 1; (k < n); k++) {
                                matrix[i][k] -= matrix[i][j] * matrix[j][k];
                                inverse[i][k] -= matrix[i][j] * inverse[j][k];
                        }
                }
        }
        for (int j = n - 1; (1 <= j); j--) {
                for (int i = j - 1; (0 <= i); i--) {
                        for (int k = 0; (k <= j); k++) {
                                inverse[i][k] -= matrix[i][j] * inverse[j][k];
                        }
                        for (int k = j + 1; (k < n); k++) {
                                matrix[i][k] -= matrix[i][j] * matrix[j][k];
                                inverse[i][k] -= matrix[i][j] * inverse[j][k];
                        }
                }
        }
        for (int i = 0; (i < n); i++) {
                System.arraycopy(inverse[i], 0, matrix[i], 0, n);
        }
    }
	
    // TODO: potentially use a math library function instead of including the
    // copyrighted function here
    // The copyright of this function is attributed to the original authors
    // Th&eacute;venaz P, Ruttimann UE &amp; Unser M (1998) A pyramid approach to
    // subpixel registration based on intensity. <i>IEEE transactions on image
    // processing : a publication of the IEEE Signal Processing Society</i>
    // <b>7</b>:27–41 (DOI: <a href="http://dx.doi.org/10.1109/83.650848">10.1109/83.650848</a>)
    // it is used here as part of creating a derivative work under the GPLv3
    public static double[] matrixMultiply(final double[][] matrix, final double[] vector) {
        final double[] result = new double[matrix.length];
        for (int i = 0; (i < matrix.length); i++) {
            result[i] = 0.0;
            for (int j = 0; (j < vector.length); j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return (result);
    }
}