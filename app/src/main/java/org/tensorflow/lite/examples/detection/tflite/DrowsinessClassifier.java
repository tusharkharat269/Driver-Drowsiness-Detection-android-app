package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * YOLO11n Drowsiness Model Classifier for the second model
 */
public class DrowsinessClassifier implements Classifier {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "DrowsinessClassifier";

    // Model configuration for drowsiness model
    private static final int INPUT_SIZE = 640; // Updated input size for the new model
    private static final int NUM_CHANNELS = 3;
    private static final int NUM_BYTES_PER_CHANNEL = 4; // Float32
    private static final int BYTES_PER_PIXEL = NUM_CHANNELS * NUM_BYTES_PER_CHANNEL;

    // Classes from the model
    private static final String[] LABELS = {
            "drowsy","active","yawn"
    };

    // Output configuration based on model specs (1, 7, 8400)
    private static final int OUTPUT_DIMS = 7; // Number of values per detection
    private static final int OUTPUT_SIZE = 8400; // Number of possible detections

    // Detection parameters
    private float confidenceThreshold = 0.45f;
    private float iouThreshold = 0.4f;
    private static final float PADDING_FACTOR = 0.03f;

    private ByteBuffer imgData;
    private int[] intValues;
    private MappedByteBuffer tfliteModel;
    private Interpreter tfLite;
    private Interpreter.Options tfLiteOptions;
    private GpuDelegate gpuDelegate;
    private NnApiDelegate nnApiDelegate;

    // Tracking aspect ratio and padding for accurate coordinate conversion
    private float imageAspectRatio = 1.0f;
    private float xPad = 0.0f;
    private float yPad = 0.0f;
    private float scale = 1.0f;

    public float getObjThresh() {
        return confidenceThreshold;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }

    public static DrowsinessClassifier create(final AssetManager assetManager, final String modelFilename) throws IOException {
        final DrowsinessClassifier classifier = new DrowsinessClassifier();

        try {
            classifier.tfliteModel = Utils.loadModelFile(assetManager, modelFilename);
            classifier.tfLiteOptions = new Interpreter.Options();
            classifier.tfLiteOptions.setNumThreads(4);

            // Configure GPU usage
            boolean useGPU = false;
            if (useGPU) {
                GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
                gpuOptions.setPrecisionLossAllowed(true);
                classifier.gpuDelegate = new GpuDelegate(gpuOptions);
                classifier.tfLiteOptions.addDelegate(classifier.gpuDelegate);
                LOGGER.i("GPU Delegate enabled for drowsiness model");
            }

            // Configure NNAPI
            boolean useNNAPI = false;
            if (useNNAPI) {
                NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                nnApiOptions.setAllowFp16(true);
                classifier.nnApiDelegate = new NnApiDelegate(nnApiOptions);
                classifier.tfLiteOptions.addDelegate(classifier.nnApiDelegate);
                LOGGER.i("NNAPI Delegate enabled for drowsiness model");
            }

            // Initialize TFLite Interpreter
            classifier.tfLite = new Interpreter(classifier.tfliteModel, classifier.tfLiteOptions);

            // Pre-allocate input/output buffers
            classifier.imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4);
            classifier.imgData.order(ByteOrder.nativeOrder());
            classifier.intValues = new int[INPUT_SIZE * INPUT_SIZE];

            // Configure thresholds based on model type
            if (modelFilename.contains("fp16")) {
                classifier.confidenceThreshold = 0.45f;
                classifier.iouThreshold = 0.4f;
            } else {
                classifier.confidenceThreshold = 0.5f;
                classifier.iouThreshold = 0.4f;
            }

            LOGGER.i("Drowsiness model loaded successfully: " + modelFilename);

            // Warmup the model
//            classifier.warmupModel();

        } catch (Exception e) {
            LOGGER.e(e, "Error initializing DrowsinessClassifier");
            throw new RuntimeException(e);
        }

        return classifier;
    }

    @Override
    public void enableStatLogging(boolean debug) { /* ... */ }

    @Override
    public String getStatString() { return ""; }

    @Override
    public void close() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        if (nnApiDelegate != null) {
            nnApiDelegate.close();
            nnApiDelegate = null;
        }
    }

    @Override
    public void setNumThreads(int numThreads) {
        if (tfLiteOptions != null) {
            tfLiteOptions.setNumThreads(numThreads);
            if (tfLite != null) {
                close();
                tfLite = new Interpreter(tfliteModel, tfLiteOptions);
            }
        }
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLiteOptions != null) {
            if (isChecked && nnApiDelegate == null) {
                NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                nnApiOptions.setAllowFp16(true);
                nnApiDelegate = new NnApiDelegate(nnApiOptions);
                tfLiteOptions.addDelegate(nnApiDelegate);
                if (tfLite != null) {
                    close();
                    tfLite = new Interpreter(tfliteModel, tfLiteOptions);
                }
                LOGGER.i("NNAPI Delegate enabled for drowsiness model");
            } else if (!isChecked && nnApiDelegate != null) {
                nnApiDelegate.close();
                nnApiDelegate = null;
                if (tfLite != null) {
                    close();
                    tfLite = new Interpreter(tfliteModel, tfLiteOptions);
                }
                LOGGER.i("NNAPI Delegate disabled for drowsiness model");
            }
        }
    }

    // Process outputs from the model format [1, 7, 8400] -> [batch, dimensions, detections]
    public List<Recognition> postprocess(float[][][] output, int imageWidth, int imageHeight) {
        List<Recognition> results = new ArrayList<>();

        // Store image dimensions for proper scaling
        float scaleX = (float) imageWidth / INPUT_SIZE;
        float scaleY = (float) imageHeight / INPUT_SIZE;

        // Apply letterboxing correction
        // This is crucial for accurate coordinates when aspect ratios don't match
        float scaleFactor = Math.min(
                (float) INPUT_SIZE / imageWidth,
                (float) INPUT_SIZE / imageHeight);

        float xPadding = (INPUT_SIZE - imageWidth * scaleFactor) / 2.0f;
        float yPadding = (INPUT_SIZE - imageHeight * scaleFactor) / 2.0f;

        // Process each detection from model output
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            // Get coordinates (normalized [0,1])
            float centerX = output[0][0][i];
            float centerY = output[0][1][i];
            float width = output[0][2][i];
            float height = output[0][3][i];

            // Skip invalid detections early
            if (centerX <= 0 || centerY <= 0 || width <= 0 || height <= 0 ||
                    centerX >= 1 || centerY >= 1 || width >= 1 || height >= 1) {
                continue;
            }

            // Find class with highest probability
            float maxClassProb = 0;
            int classId = -1;

            for (int j = 0; j < LABELS.length; j++) {
                float classProb = output[0][j + 4][i]; // Class probs start at index 4
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j;
                }
            }

            // Calculate final confidence
            float confidence = maxClassProb;

            // Filter low confidence detections
            if (confidence > confidenceThreshold && classId >= 0) {
                // Convert from normalized coordinates to INPUT_SIZE space
                // Account for letterboxing (padding) when model preserves aspect ratio
                float x1 = (centerX - (width / 2.0f)) * INPUT_SIZE;
                float y1 = (centerY - (height / 2.0f)) * INPUT_SIZE;
                float x2 = (centerX + (width / 2.0f)) * INPUT_SIZE;
                float y2 = (centerY + (height / 2.0f)) * INPUT_SIZE;

                // Adjust for letterboxing padding
                x1 = (x1 - xPadding) / scaleFactor;
                y1 = (y1 - yPadding) / scaleFactor;
                x2 = (x2 - xPadding) / scaleFactor;
                y2 = (y2 - yPadding) / scaleFactor;

                // Ensure boxes stay within image bounds (with small padding)
                float paddingX = imageWidth * PADDING_FACTOR;
                float paddingY = imageHeight * PADDING_FACTOR;

                x1 = Math.max(paddingX, Math.min(x1, imageWidth - paddingX));
                y1 = Math.max(paddingY, Math.min(y1, imageHeight - paddingY));
                x2 = Math.max(paddingX, Math.min(x2, imageWidth - paddingX));
                y2 = Math.max(paddingY, Math.min(y2, imageHeight - paddingY));

                // Skip invalid boxes
                if (x2 <= x1 || y2 <= y1) continue;

                // Check if box is too small (minimum size check)
                float boxWidth = x2 - x1;
                float boxHeight = y2 - y1;
                float minSize = Math.min(imageWidth, imageHeight) * 0.03f; // 3% of smaller dimension

                if (boxWidth < minSize || boxHeight < minSize) continue;

                RectF location = new RectF(x1, y1, x2, y2);

                // Add to results
                results.add(new Recognition(
                        "" + classId,
                        LABELS[classId],
                        confidence,
                        location,
                        classId
                ));

                // Debug log to verify box coordinates
                Log.d(TAG, "Detection: class=" + LABELS[classId] +
                        ", conf=" + String.format("%.2f", confidence) +
                        ", box=" + location.toShortString());
            }
        }

        // Apply Non-Maximum Suppression
        return nonMaxSuppression(results);
    }

    private List<Recognition> nonMaxSuppression(List<Recognition> detections) {
        if (detections.isEmpty()) {
            return detections;
        }

        // Group detections by class
        Map<Integer, List<Recognition>> classMap = new HashMap<>();
        for (Recognition detection : detections) {
            int classId = detection.getDetectedClass();
            if (!classMap.containsKey(classId)) {
                classMap.put(classId, new ArrayList<>());
            }
            classMap.get(classId).add(detection);
        }

        List<Recognition> result = new ArrayList<>();

        // Process each class independently
        for (List<Recognition> classDetections : classMap.values()) {
            // Sort by confidence
            Collections.sort(classDetections,
                    (d1, d2) -> Float.compare(d2.getConfidence(), d1.getConfidence()));

            List<Recognition> nmsResult = new ArrayList<>();

            while (!classDetections.isEmpty()) {
                // Take top-scoring box
                Recognition main = classDetections.remove(0);
                nmsResult.add(main);
                RectF mainBox = main.getLocation();

                List<Integer> indicesToRemove = new ArrayList<>();

                // Find all overlapping boxes
                for (int i = 0; i < classDetections.size(); i++) {
                    Recognition current = classDetections.get(i);
                    float iou = calculateIoU(mainBox, current.getLocation());

                    if (iou > iouThreshold) {
                        indicesToRemove.add(i);
                    }
                }

                // Remove overlapping boxes in reverse order
                for (int i = indicesToRemove.size() - 1; i >= 0; i--) {
                    classDetections.remove((int) indicesToRemove.get(i));
                }
            }

            result.addAll(nmsResult);
        }

        return result;
    }

    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Preprocess the image and get letterbox parameters
        float[] letterboxInfo = convertBitmapToByteBuffer(bitmap);
        scale = letterboxInfo[0];
        xPad = letterboxInfo[1];
        yPad = letterboxInfo[2];

        // Run inference
        long startTime = SystemClock.uptimeMillis();

        // Create output array that matches the model's output shape [1, 7, 8400]
        float[][][] output = new float[1][7][8400];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, output);

        try {
            tfLite.runForMultipleInputsOutputs(new Object[]{imgData}, outputMap);
            Log.d(TAG, "Drowsiness inference completed successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error during drowsiness inference: " + e.getMessage(), e);
            return new ArrayList<>();
        }

        long inferenceTime = SystemClock.uptimeMillis() - startTime;
        LOGGER.i("Drowsiness inference time: " + inferenceTime + "ms");

        // Process model output to get detections
        int imageWidth = bitmap.getWidth();
        int imageHeight = bitmap.getHeight();
        imageAspectRatio = (float) imageWidth / imageHeight;

        // Store these values for postprocessing
        this.xPad = xPad;
        this.yPad = yPad;
        this.scale = scale;

        return postprocess(output, imageWidth, imageHeight);
    }

    /**
     * Image preprocessing with letterboxing
     */
    protected float[] convertBitmapToByteBuffer(Bitmap bitmap) {
        // Reset the buffer position
        imgData.rewind();

        // Calculate scaling to maintain aspect ratio
        float scale = Math.min(
                (float) INPUT_SIZE / bitmap.getWidth(),
                (float) INPUT_SIZE / bitmap.getHeight());

        // Calculate padding for letterboxing
        int scaledWidth = Math.round(bitmap.getWidth() * scale);
        int scaledHeight = Math.round(bitmap.getHeight() * scale);
        float xPad = (INPUT_SIZE - scaledWidth) / 2f;
        float yPad = (INPUT_SIZE - scaledHeight) / 2f;

        // Create intermediate bitmap with correct scale
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(
                bitmap, scaledWidth, scaledHeight, true);

        // Create target bitmap with padding (letterboxing)
        Bitmap letterboxedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(letterboxedBitmap);
        canvas.drawColor(Color.BLACK); // Fill with black (background padding)

        // Center the resized image
        canvas.drawBitmap(resizedBitmap, xPad, yPad, null);

        // Extract pixel values
        letterboxedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // Process each pixel - normalize to [0,1]
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[pixel++];

                // Extract and normalize RGB values to [0,1]
                imgData.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
                imgData.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
                imgData.putFloat((pixelValue & 0xFF) / 255.0f);
            }
        }

        // Clean up temporary bitmaps
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle();
        }
        if (letterboxedBitmap != resizedBitmap) {
            letterboxedBitmap.recycle();
        }

        // Return letterboxing information for coordinate correction
        return new float[]{scale, xPad, yPad};
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        // Handle edge cases
        if (boxA == null || boxB == null ||
                boxA.width() <= 0 || boxA.height() <= 0 ||
                boxB.width() <= 0 || boxB.height() <= 0) {
            return 0.0f;
        }

        float xA = Math.max(boxA.left, boxB.left);
        float yA = Math.max(boxA.top, boxB.top);
        float xB = Math.min(boxA.right, boxB.right);
        float yB = Math.min(boxA.bottom, boxB.bottom);

        // Compute the area of intersection
        float intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);

        // Compute the area of both rectangles
        float boxAArea = boxA.width() * boxA.height();
        float boxBArea = boxB.width() * boxB.height();

        // Compute IoU
        float unionArea = boxAArea + boxBArea - intersectionArea;
        float iou = unionArea > 0 ? intersectionArea / unionArea : 0;

        return iou;
    }

    private void warmupModel() {
        // Create a blank input of correct dimensions
        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE * 3; i++) {
            imgData.putFloat(0.0f);
        }

        // Create output buffer
        float[][][] dummyOutput = new float[1][7][8400];

        // Run inference on dummy data
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, dummyOutput);

        try {
            tfLite.runForMultipleInputsOutputs(new Object[]{imgData}, outputMap);
            LOGGER.i("Drowsiness model warmup completed");
        } catch (Exception e) {
            LOGGER.e("Drowsiness model warmup failed: " + e.getMessage());
        }
    }
}