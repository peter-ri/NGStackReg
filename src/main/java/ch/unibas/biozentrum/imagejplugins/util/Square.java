package ch.unibas.biozentrum.imagejplugins.util;

public final class Square {
	/*
	 * (top left 0,0 bottom right width, height)
	 * x1,y1   x2,y2
	 * x3,y3   x4,y4
	 */
	public double x1, y1;
	public double x2, y2;
	public double x3, y3;
	public double x4, y4;
	public Square(long width, long height) {
		x1 = x3 = y1 = y2 = 0.0;
		x2 = x4 = (double)width;
		y3 = y4 = (double)height;
	}
	
	public void set(long width, long height) {
		x1 = x3 = y1 = y2 = 0.0;
		x2 = x4 = (double)width;
		y3 = y4 = (double)height;
	}
	
	public double minX() {
		return Math.min(Math.min(Math.min(x1, x2), x3), x4);
	}
	
	public double minY() {
		return Math.min(Math.min(Math.min(y1, y2), y3), y4);
	}
	
	public double maxX() {
		return Math.max(Math.max(Math.max(x1, x2), x3), x4);
	}
	
	public double maxY() {
		return Math.max(Math.max(Math.max(y1, y2), y3), y4);
	}
}
