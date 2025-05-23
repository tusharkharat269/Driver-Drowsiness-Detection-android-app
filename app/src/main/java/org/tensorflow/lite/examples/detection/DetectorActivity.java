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

import static org.tensorflow.lite.examples.detection.tflite.YOLOv11Classifier.*;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;

import org.tensorflow.lite.examples.detection.tflite.YOLOClassifier;

import org.tensorflow.lite.examples.detection.tflite.Classifier;

import org.tensorflow.lite.examples.detection.tracking.DriverSafetyAlarmSystem;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {

    private static final String Tag = "DetectorActivity";
    private static final Logger LOGGER = new Logger();

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.25f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

//    private YoloV5Classifier detector;

//    private YOLOv11Classifier detector;
//    private DrowsinessClassifier detector2;

    //    private YOLOv11Classifier detector;
    private YOLOClassifier model1;
    private YOLOClassifier model2;

    private DriverSafetyAlarmSystem safetyAlarmSystem;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    // Add these fields to your class

    private DetectionTracker detectionTracker;

//    private ActivityMonitor activityMonitor;
//    private FatigueMonitor fatigueMonitor;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        safetyAlarmSystem = new DriverSafetyAlarmSystem(this);


        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);
        detectionTracker = new DetectionTracker();

        try {

            model1 = DetectorFactory.getDetector(getAssets(), 0);
            model2 = DetectorFactory.getDetector(getAssets(), 1);


            LOGGER.i("tflite model intialized");
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = 720;

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
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
    }


    @Override
    protected void processImage() {
        final long currTimestamp = System.currentTimeMillis();
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @RequiresApi(api = Build.VERSION_CODES.N)
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
//                        final List<Classifier.Recognition> results1 = model1.recognizeImage(croppedBitmap);
//                        final List<Classifier.Recognition> results2 = model2.recognizeImage(croppedBitmap);

                        ExecutorService executor = Executors.newFixedThreadPool(2); // Use a thread pool

                        Future<List<Recognition>> future1 = executor.submit(() -> model1.recognizeImage(croppedBitmap));
                        Future<List<Recognition>> future2 = executor.submit(() -> model2.recognizeImage(croppedBitmap));

// Wait for both results
                        List<Recognition> results1 = null;
                        try {
                            results1 = future1.get();
                        } catch (ExecutionException | InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                        List<Recognition> results2 = null;
                        try {
                            results2 = future2.get();
                        } catch (ExecutionException | InterruptedException e) {
                            throw new RuntimeException(e);
                        }

                        List<Classifier.Recognition> rawResults = new ArrayList<>();
                        rawResults.addAll(results1);
                        rawResults.addAll(results2);

                        executor.shutdown(); // Clean up

                        final List<Classifier.Recognition> results = detectionTracker.updateTrackedObjects(rawResults);


                        for (Classifier.Recognition result : results) {
                            Log.d("RAW_DETECTION", "Label: " + result.getTitle() +
                                    " Conf: " + result.getConfidence() +
                                    " Box: " + result.getLocation().toShortString());
                        }

                        Log.e("CHECK", "run: " + results.size());

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);

                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }

//                            String className = result.getTitle().toLowerCase().split(" #")[0];
//                            if (safetyAlarmSystem.detectionStates.containsKey(className)) {
//                                Log.i(Tag,"classnames: "+ className);
//                                safetyAlarmSystem.processDetections(className, timestamp);
//                            }


                        }

                        tracker.trackResults(mappedRecognitions, currTimestamp);
                        trackingOverlay.postInvalidate();

                        safetyAlarmSystem.processDetections(mappedRecognitions, currTimestamp);

                        computingDetection = false;
                    }
                });
    }




    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
//        runInBackground(() -> detector.setUseNNAPI(isChecked));
//        runInBackground(() -> detector2.setUseNNAPI(isChecked));
        runInBackground(() -> model1.setUseNNAPI(isChecked));
        runInBackground(() -> model2.setUseNNAPI(isChecked));

    }

    @Override
    protected void setNumThreads(final int numThreads) {
//        runInBackground(() -> detector.setNumThreads(numThreads));
//        runInBackground(() -> detector2.setNumThreads(numThreads));
        runInBackground(() -> model1.setNumThreads(numThreads));
        runInBackground(() -> model2.setNumThreads(numThreads));

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (croppedBitmap != null) croppedBitmap.recycle();
        if (cropCopyBitmap != null) cropCopyBitmap.recycle();
        safetyAlarmSystem.release();
    }

}
