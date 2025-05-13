package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Unified YOLO classifier for both drowsiness and distraction detection
 * Supports multiple input sizes, output dimensions, and model types
 */
public class YOLOClassifier implements Classifier {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "YOLOClassifier";

    // Model configuration
    private final int inputSize;
    private static final int NUM_CHANNELS = 3;
    private static final int NUM_BYTES_PER_CHANNEL = 4; // Float32
    private static final int BYTES_PER_PIXEL = NUM_CHANNELS * NUM_BYTES_PER_CHANNEL;
    private static final float PADDING_FACTOR = 0.03f; // 3% padding

    // Output configuration
    private final int outputWidth;  // Number of values per detection
    private final int outputHeight; // Number of possible detections

    // Detection parameters
//    private final float confidenceThreshold;
    private final float[] confidenceThresholds;

    private final float iouThreshold;

    // Labels for classification
    private final String[] labels;

    // TFLite execution resources
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

    // Model filename for identification
    private final String modelFilename;

    private YOLOClassifier(String modelFilename, int inputSize, String[] labels,
                           int outputWidth, int outputHeight,
                           float confidenceThresholds[], float iouThreshold) {
        this.modelFilename = modelFilename;
        this.inputSize = inputSize;
        this.labels = labels;
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        this.confidenceThresholds = confidenceThresholds;
        this.iouThreshold = iouThreshold;
    }

    /**
     * Creates a classifier with the provided configuration
     */
    public static YOLOClassifier create(
            final AssetManager assetManager,
            final DetectorFactory.ModelConfig config) throws IOException {

        final YOLOClassifier classifier = new YOLOClassifier(
                config.modelFilename,
                config.inputSize,
                config.labels,
                config.outputWidth,
                config.outputHeight,
                config.confidenceThresholds,
                config.iouThreshold
        );

        try {
            classifier.tfliteModel = Utils.loadModelFile(assetManager, config.modelFilename);
            classifier.tfLiteOptions = new Interpreter.Options();

            // Enable multithreading for better performance
            classifier.tfLiteOptions.setNumThreads(4);

            // Option 1: GPU Delegate for better performance on compatible devices
            boolean useGPU = false;
            if (useGPU) {
                GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
                gpuOptions.setPrecisionLossAllowed(true); // Allow FP16 for better performance
                classifier.gpuDelegate = new GpuDelegate(gpuOptions);
                classifier.tfLiteOptions.addDelegate(classifier.gpuDelegate);
                LOGGER.i("GPU Delegate enabled for " + config.modelFilename);
            }

            // Option 2: NNAPI for Android 8.1+ (API 27+)
            boolean useNNAPI = false;
            if (useNNAPI) {
                NnApiDelegate.Options nnApiOptions = new NnApiDelegate.Options();
                nnApiOptions.setAllowFp16(true);
                classifier.nnApiDelegate = new NnApiDelegate(nnApiOptions);
                classifier.tfLiteOptions.addDelegate(classifier.nnApiDelegate);
                LOGGER.i("NNAPI Delegate enabled for " + config.modelFilename);
            }

            // Initialize TFLite Interpreter
            classifier.tfLite = new Interpreter(classifier.tfliteModel, classifier.tfLiteOptions);

            // Pre-allocate input/output buffers with proper sizes
            classifier.imgData = ByteBuffer.allocateDirect(1 * config.inputSize * config.inputSize * 3 * 4);
            classifier.imgData.order(ByteOrder.nativeOrder());
            classifier.intValues = new int[config.inputSize * config.inputSize];

            LOGGER.i("Model loaded successfully: " + config.modelFilename);

            // Warmup the model with dummy data to improve first inference time
            classifier.warmupModel();

        } catch (Exception e) {
            LOGGER.e(e, "Error initializing YOLOClassifier for " + config.modelFilename);
            throw new RuntimeException(e);
        }

        return classifier;
    }

    @Override
    public float getObjThresh() {
        return 0;
    }

    public int getInputSize() {
        return inputSize;
    }

    @Override
    public void enableStatLogging(boolean debug) { /* Not implemented */ }

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
                LOGGER.i("NNAPI Delegate enabled for " + modelFilename);
            } else if (!isChecked && nnApiDelegate != null) {
                nnApiDelegate.close();
                nnApiDelegate = null;
                if (tfLite != null) {
                    close();
                    tfLite = new Interpreter(tfliteModel, tfLiteOptions);
                }
                LOGGER.i("NNAPI Delegate disabled for " + modelFilename);
            }
        }
    }

    /**
     * Process model output to get detections
     */
    public List<Recognition> postprocess(float[][][] output, int imageWidth, int imageHeight) {
        List<Recognition> results = new ArrayList<>();

        // Store image dimensions for proper scaling
        float scaleX = (float) imageWidth / inputSize;
        float scaleY = (float) imageHeight / inputSize;

        // Apply letterboxing correction
        float scaleFactor = Math.min(
                (float) inputSize / imageWidth,
                (float) inputSize / imageHeight);

        float xPadding = (inputSize - imageWidth * scaleFactor) / 2.0f;
        float yPadding = (inputSize - imageHeight * scaleFactor) / 2.0f;

        // Process each detection from model output
        for (int i = 0; i < outputHeight; i++) {
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

            for (int j = 0; j < labels.length; j++) {
                float classProb = output[0][j + 4][i]; // Class probs start at index 4
                if (classProb > maxClassProb) {
                    maxClassProb = classProb;
                    classId = j;
                }
            }

            // Calculate final confidence
            float confidence = maxClassProb;

            // Filter low confidence detections
            if (confidence > confidenceThresholds[classId] && classId >= 0) {
                // Convert from normalized coordinates to INPUT_SIZE space
                float x1 = (centerX - (width / 2.0f)) * inputSize;
                float y1 = (centerY - (height / 2.0f)) * inputSize;
                float x2 = (centerX + (width / 2.0f)) * inputSize;
                float y2 = (centerY + (height / 2.0f)) * inputSize;

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
                        labels[classId],
                        confidence,
                        location,
                        classId
                ));

                // Debug log to verify box coordinates
                Log.d(TAG, "Detection: class=" + labels[classId] +
                        ", conf=" + String.format("%.2f", confidence) +
                        ", box=" + location.toShortString());
            }
        }

        // Apply Non-Maximum Suppression
//        return results;
        return performNMS(results);
    }

    /**
     * Enhanced NMS implementation that can use Weighted Box Fusion
     */
    private List<Recognition> performNMS(List<Recognition> detections) {
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
                List<Recognition> overlappingBoxes = new ArrayList<>();
                List<Float> ious = new ArrayList<>();

                // Find all overlapping boxes
                for (int i = 0; i < classDetections.size(); i++) {
                    Recognition current = classDetections.get(i);
                    float iou = calculateIoU(mainBox, current.getLocation());

                    if (iou > iouThreshold) {
                        indicesToRemove.add(i);
                        overlappingBoxes.add(current);
                        ious.add(iou);
                    }
                }

                // Apply Weighted Box Fusion if there are overlaps
                if (!overlappingBoxes.isEmpty()) {
                    // Start with current box values weighted by confidence
                    float x1Weighted = mainBox.left * main.getConfidence();
                    float y1Weighted = mainBox.top * main.getConfidence();
                    float x2Weighted = mainBox.right * main.getConfidence();
                    float y2Weighted = mainBox.bottom * main.getConfidence();
                    float totalWeight = main.getConfidence();

                    // Add weights from all overlapping boxes
                    for (Recognition box : overlappingBoxes) {
                        float weight = box.getConfidence();
                        totalWeight += weight;

                        x1Weighted += box.getLocation().left * weight;
                        y1Weighted += box.getLocation().top * weight;
                        x2Weighted += box.getLocation().right * weight;
                        y2Weighted += box.getLocation().bottom * weight;
                    }

                    // Calculate final weighted coordinates
                    mainBox.left = x1Weighted / totalWeight;
                    mainBox.top = y1Weighted / totalWeight;
                    mainBox.right = x2Weighted / totalWeight;
                    mainBox.bottom = y2Weighted / totalWeight;

                    // Use maximum confidence as the final confidence
                    float maxConfidence = main.getConfidence();
                    for (Recognition box : overlappingBoxes) {
                        maxConfidence = Math.max(maxConfidence, box.getConfidence());
                    }
                    main.setConfidence(maxConfidence);
                }

                // Remove overlapping boxes in reverse order (to maintain indices)
                for (int i = indicesToRemove.size() - 1; i >= 0; i--) {
                    classDetections.remove((int) indicesToRemove.get(i));
                }
            }

            result.addAll(nmsResult);
        }

        return result;
    }

    /**
     * Process bitmap image and run inference
     */
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Preprocess the image and get letterbox parameters
        float[] letterboxInfo = convertBitmapToByteBuffer(bitmap);
        scale = letterboxInfo[0];
        xPad = letterboxInfo[1];
        yPad = letterboxInfo[2];

        // Run inference
        long startTime = SystemClock.uptimeMillis();

        // Log input information for debugging
        int[] inputShape = tfLite.getInputTensor(0).shape();
        int[] outputShape = tfLite.getOutputTensor(0).shape();

        // Match array shape to the model output shape
        float[][][] output = new float[1][outputWidth][outputHeight];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, output);

        try {
            tfLite.runForMultipleInputsOutputs(new Object[]{imgData}, outputMap);
            Log.d(TAG, "Inference completed successfully for " + modelFilename);
        } catch (Exception e) {
            Log.e(TAG, "Error during inference: " + e.getMessage(), e);
            return new ArrayList<>();
        }

        long inferenceTime = SystemClock.uptimeMillis() - startTime;
        LOGGER.i("Inference time: " + inferenceTime + "ms for " + modelFilename);

        // Process model output to get detections
        int imageWidth = bitmap.getWidth();
        int imageHeight = bitmap.getHeight();
        imageAspectRatio = (float) imageWidth / imageHeight;

        // Store these values for postprocessing (aspect ratio correction)
        this.xPad = xPad;
        this.yPad = yPad;
        this.scale = scale;

        return postprocess(output, imageWidth, imageHeight);
    }

    /**
     * Convert bitmap to proper input format with letterboxing
     */
    protected float[] convertBitmapToByteBuffer(Bitmap bitmap) {
        // Reset the buffer position
        imgData.rewind();

        // Calculate scaling to maintain aspect ratio
        float scale = Math.min(
                (float) inputSize / bitmap.getWidth(),
                (float) inputSize / bitmap.getHeight());

        // Calculate padding for letterboxing
        int scaledWidth = Math.round(bitmap.getWidth() * scale);
        int scaledHeight = Math.round(bitmap.getHeight() * scale);
        float xPad = (inputSize - scaledWidth) / 2f;
        float yPad = (inputSize - scaledHeight) / 2f;

        // Create intermediate bitmap with correct scale
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(
                bitmap, scaledWidth, scaledHeight, true);

        // Create target bitmap with padding (letterboxing)
        Bitmap letterboxedBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(letterboxedBitmap);
        canvas.drawColor(Color.BLACK); // Fill with black (background padding)

        // Center the resized image
        canvas.drawBitmap(resizedBitmap, xPad, yPad, null);

        // Extract pixel values
        letterboxedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize);

        // Process each pixel - normalize to [0,1]
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
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

    /**
     * Calculate Intersection over Union between two bounding boxes
     */
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

    /**
     * Warm up the model to reduce first inference latency
     */
    private void warmupModel() {
        // Create a blank input of correct dimensions
        imgData.rewind();
        for (int i = 0; i < inputSize * inputSize * 3; i++) {
            imgData.putFloat(0.0f);
        }

        // Create output buffer with correct dimensions
        float[][][] dummyOutput = new float[1][outputWidth][outputHeight];

        // Run inference on dummy data
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, dummyOutput);

        try {
            tfLite.runForMultipleInputsOutputs(new Object[]{imgData}, outputMap);
            LOGGER.i("Model warmup completed for " + modelFilename);
        } catch (Exception e) {
            LOGGER.e("Model warmup failed for " + modelFilename + ": " + e.getMessage());
        }
    }
}