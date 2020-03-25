package org.tensorflow.lite.examples.detection.env;

import android.graphics.Bitmap;

public class ScaleImage {

    public static Bitmap resizeBitmap(Bitmap source){
        int width = source.getWidth();
        int height = source.getHeight();

        int select_side = select_side(width,height);
        float scale_factor = scale_factor(select_side);


        float scale_width=scale_side(width,scale_factor);
        float scale_height=scale_side(height,scale_factor);

        Bitmap resizedBitmap = Bitmap.createScaledBitmap(source,(int)scale_width,(int)scale_height, false);

        return resizedBitmap;
    }

    private static float scale_side(int side, float scale_factor) {
        return (float)side/scale_factor;
    }

    private static float scale_factor(int scale_side) {
        return (float)scale_side/450;
    }

    private static int select_side(int width, int height) {
        if(width>height)
            return width;
        return height;
    }
}
