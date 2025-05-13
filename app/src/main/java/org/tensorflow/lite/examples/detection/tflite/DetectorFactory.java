package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetManager;
import android.util.Log;

import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Factory class that manages YOLO detector instances
 * Allows reusing a single optimized detector framework for multiple models
 */
public class DetectorFactory {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "DetectorFactory";

    // Cache loaded detectors to avoid reloading identical models
    private static final Map<String, YOLOClassifier> detectorCache = new HashMap<>();

    // Model configurations
    public static final int MODEL_TYPE_DISTRACTION = 0;
    public static final int MODEL_TYPE_DROWSINESS = 1;

    // Configuration for distraction model
    private static final ModelConfig DISTRACTION_CONFIG = new ModelConfig(
            "final_last_yolo11_float32.tflite",
            416,
            new String[]{"open_eyes", "distracted", "closed_eyes", "head_drop", "phone", "smoking", "y"},
            11,
            3549,
//            0.35f,
            new float[] {0.25F, 0.5F, 0.1F, 0.05F, 0.6F, 0.2F, 0.8F},
            0.5f
    );

    // Configuration for drowsiness model
    private static final ModelConfig DROWSINESS_CONFIG = new ModelConfig(
            "keggle-model_float32.tflite",
            640,
            new String[]{"drowsy", "active", "yawning"},
            7,
            8400,
//            0.45f,
            new float[] {0.4F, 0.70F, 0.75F},
            0.6f
    );

    /**
     * Get a detector instance for the specified model type
     * @param assets Android asset manager
     * @param modelType Type of model (MODEL_TYPE_DISTRACTION or MODEL_TYPE_DROWSINESS)
     * @return Configured YOLOClassifier for the requested model
     * @throws IOException if model loading fails
     */
    public static YOLOClassifier getDetector(AssetManager assets, int modelType) throws IOException {
        ModelConfig config;

        // Select appropriate model configuration
        switch (modelType) {
            case MODEL_TYPE_DISTRACTION:
                config = DISTRACTION_CONFIG;
                break;
            case MODEL_TYPE_DROWSINESS:
                config = DROWSINESS_CONFIG;
                break;
            default:
                throw new IllegalArgumentException("Unknown model type: " + modelType);
        }

        // Check if we already have this model loaded
        String cacheKey = config.modelFilename;
        if (detectorCache.containsKey(cacheKey)) {
            LOGGER.i("Returning cached detector for " + cacheKey);
            return detectorCache.get(cacheKey);
        }

        // Otherwise, create a new detector
        Log.i(TAG, "Creating new detector for " + config.modelFilename);
        YOLOClassifier detector = YOLOClassifier.create(assets, config);

        // Cache it for future use
        detectorCache.put(cacheKey, detector);
        return detector;
    }

    /**
     * Clear the detector cache to free up memory
     */
    public static void clearCache() {
        for (YOLOClassifier detector : detectorCache.values()) {
            detector.close();
        }
        detectorCache.clear();
    }

    /**
     * Model configuration class to store parameters for each model type
     */
    public static class ModelConfig {
        public final String modelFilename;
        public final int inputSize;
        public final String[] labels;
        public final int outputWidth;
        public final int outputHeight;
//        public final float confidenceThreshold;
        public final float[] confidenceThresholds;

        public final float iouThreshold;

        public ModelConfig(String modelFilename, int inputSize, String[] labels,
                           int outputWidth, int outputHeight,
                           float[] confidenceThresholds, float iouThreshold) {
            this.modelFilename = modelFilename;
            this.inputSize = inputSize;
            this.labels = labels;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.confidenceThresholds = confidenceThresholds;
            this.iouThreshold = iouThreshold;
        }
    }
}