/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.DisplayMetrics;
import android.util.Size;
import android.util.SparseArray;
import android.util.TypedValue;
import android.view.Display;
import android.view.WindowManager;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.barcode.Barcode;
import com.google.android.gms.vision.barcode.BarcodeDetector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;


public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(3264, 1836 );
    public static float desireScreenRate=0;

    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;
    private Classifier detector;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private boolean computingDetection = false;
    private long timestamp = 0;
    private Matrix frameToCropTransform;
    private static Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private BorderedText borderedText;
    public static Matrix getTransformMatrix(){
        return cropToFrameTransform;
    }
    public static Map<String,Classifier.Recognition> cacheResults = new HashMap<>();
    public static int cacheCounter = 0;
    public ScheduledExecutorService scheduleTaskExecutor;
    final AtomicBoolean clear = new AtomicBoolean(false);

    public void createClearTimer(){
        scheduleTaskExecutor = Executors.newScheduledThreadPool(1);
        scheduleTaskExecutor.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {runOnUiThread(() -> {
                if(clear.get()) {
                    List<Classifier.Recognition> list = new ArrayList<Classifier.Recognition>(cacheResults.values());
                    tracker.trackResults(list, 0);
                    trackingOverlay.postInvalidate();
                    cacheResults.clear();
                    cacheCounter = 0;
                    System.out.println("Clear");
                }
            });
            }
        }, 0, 2, TimeUnit.SECONDS); // or .MINUTES, .HOURS etc.


    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);
        tracker = new MultiBoxTracker(this);
        int cropSize = TF_OD_API_INPUT_SIZE;
        try {
            detector = TFLiteObjectDetectionAPIModel.create(getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE, TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform = ImageUtils.getTransformationMatrix(previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });
        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
        createClearTimer();
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        //trackingOverlay.postInvalidate();

       if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(() -> {

            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap,rgbFrameBitmap);
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            //final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();
            for (final Classifier.Recognition result : results) {
                final RectF location = result.getLocation();
                result.setLocation(location);
                //mappedRecognitions.add(result);
                Bitmap bitmap = getCropBitmap(rgbFrameBitmap,result.getLocation(),false);
                String barcode = decode(bitmap);
                if(barcode==null){
                    bitmap = getCropBitmap(rgbFrameBitmap,result.getLocation(),true);
                    barcode = decode(bitmap);
                    if(barcode!=null) {
                        cacheResults.put(barcode, result);
                        CameraActivity.addBarcode(barcode);
                        clear.set(false);
                    }
                }else{
                    cacheResults.put(barcode,result);
                    CameraActivity.addBarcode(barcode);
                    clear.set(false);
                }
            }
            if(results.size()>0){
                List<Classifier.Recognition> list = new ArrayList<Classifier.Recognition>(cacheResults.values());
                tracker.trackResults(list, currTimestamp);
                trackingOverlay.postInvalidate();
            }else{
                clear.set(true);
            }
            computingDetection=false;
        });
    }

    private Bitmap getCropBitmap(Bitmap source, RectF cropRectF,boolean threshold) {
        float width=0;
        float height=0;

        if(cropRectF.width()+75<source.getWidth()) {
            width = cropRectF.width() + 75;
        }else{
            width = source.getWidth();
        }

        if(cropRectF.height()+75<source.getHeight()) {
            height = cropRectF.height() + 75;
        }else {
            height = source.getHeight();
        }

        Bitmap resultBitmap = Bitmap.createBitmap((int) width, (int)height, Bitmap.Config.ARGB_8888);
        Canvas cavas = new Canvas(resultBitmap);

        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
        paint.setColor(Color.WHITE);

        cavas.drawRect(new RectF(0, 0, width, height), paint);

        Matrix matrix = new Matrix();
        matrix.postTranslate(-cropRectF.left+50, -cropRectF.top+50);
        cavas.drawBitmap(source, matrix, paint);

        if(threshold) {
            Mat imageMat = new Mat();
            Utils.bitmapToMat(resultBitmap, imageMat);
            Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_BGR2GRAY);
            Core.normalize(imageMat, imageMat, 0, 255, Core.NORM_MINMAX);

            Mat kernel = new Mat(new org.opencv.core.Size(1, 1), CvType.CV_8U, new Scalar(255));
            Imgproc.morphologyEx(imageMat, imageMat, Imgproc.MORPH_OPEN, kernel);
            Imgproc.morphologyEx(imageMat, imageMat, Imgproc.MORPH_CLOSE, kernel);
            Imgproc.adaptiveThreshold(imageMat, imageMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY, 41, 20);

            Utils.matToBitmap(imageMat, resultBitmap);
        }

        return resultBitmap;
    }

    private String decode(Bitmap bMap){
        Frame frame = new Frame.Builder().setBitmap(bMap).build();
        SparseArray<Barcode> barcodes = CameraActivity.barcodeDetector.detect(frame);
        if(barcodes.size()>0) {
            Barcode thisCode = barcodes.valueAt(0);
            return thisCode.rawValue;
        }else{
            return null;
        }
    }


    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        setScreenResolution(this);
        System.out.println(desireScreenRate);
        return DESIRED_PREVIEW_SIZE;
    }

    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }

    private static void setScreenResolution(Context context)
    {
        WindowManager wm = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        Display display = wm.getDefaultDisplay();
        DisplayMetrics metrics = new DisplayMetrics();
        display.getMetrics(metrics);
        int width = metrics.widthPixels;
        int height = metrics.heightPixels;
        desireScreenRate=(float)height/(float)width;
    }

}
