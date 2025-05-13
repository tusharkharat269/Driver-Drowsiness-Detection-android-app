package org.tensorflow.lite.examples.detection.yolo11_tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Handler;
import android.os.Looper;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

public class YOLOv11ClassifierOld {

    // Model configuration
    public static final int INPUT_SIZE = 416;
    private static final int NUM_CLASSES = 7;
    private static final float CONFIDENCE_THRESHOLD = 0.2f;
    private static final float NMS_THRESHOLD = 0.5f;
    private static final int NUM_DETECTIONS = 3549;

    private Interpreter interpreter;
    private int[] inputShape;
    private int[] outputShape;

    // Class labels
    private final String[] classes = {
            "awake", "distracted", "drowsy",
            "head drop", "phone", "smoking", "yawn"
    };

    // Class colors (BGR format)
    private final int[] classColors = {
            0xFF0000, 0x00FFFF, 0xFF00FF,
            0x0000FF, 0x00FF00, 0xFFA500, 0xFF69B4
    };

    public static class Detection {
        private RectF rect;

        private float confidence;
        private int classId;

        public Detection(RectF rect, float confidence, int classId) {
            this.rect = rect;
            this.confidence = confidence;
            this.classId = classId;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }

        public RectF getLocation() {
            return rect;
        }

        public void setLocation(RectF rect) {
            this.rect = rect;
        }

        public int getClassId() {
            return classId;
        }

        public void setClassId(int classId) {
            this.classId = classId;
        }
    }

    public YOLOv11ClassifierOld(AssetManager assetManager, String modelPath) throws IOException {
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath));
        inputShape = interpreter.getInputTensor(0).shape();
        outputShape = interpreter.getOutputTensor(0).shape();
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public List<Detection> detect(Bitmap bitmap, int originalWidth, int originalHeight) {
        // Preprocess input
        ByteBuffer inputBuffer = preprocessBitmap(bitmap);

        // Output buffer [1, 11, 3549]
        float[][][] output = new float[1][11][NUM_DETECTIONS];

        // Run inference
        interpreter.run(inputBuffer, output);

        // Postprocess results
        return postprocess(output[0], originalWidth, originalHeight);
    }

    private ByteBuffer preprocessBitmap(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            // Extract RGB components and normalize
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }

        return inputBuffer;
    }

    private List<Detection> postprocess(float[][] outputs, int originalWidth, int originalHeight) {
        List<Detection> detections = new ArrayList<>();
        final float scaleX = (float) originalWidth / INPUT_SIZE;
        final float scaleY = (float) originalHeight / INPUT_SIZE;

        for (int i = 0; i < NUM_DETECTIONS; i++) {
            // Parse detection parameters
            float x = outputs[0][i];  // Center X
            float y = outputs[1][i];  // Center Y
            float w = outputs[2][i];  // Width
            float h = outputs[3][i];  // Height

            // Get class probabilities
            float[] classScores = new float[NUM_CLASSES];
            for (int c = 0; c < NUM_CLASSES; c++) {
                classScores[c] = outputs[4 + c][i];
            }

            // Find best class
            int classId = -1;
            float maxConfidence = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                if (classScores[c] > maxConfidence) {
                    maxConfidence = classScores[c];
                    classId = c;
                }
            }

            if (maxConfidence > CONFIDENCE_THRESHOLD && classId != -1) {
                // Convert to image coordinates
                float left = (x - w/2) * scaleX;
                float top = (y - h/2) * scaleY;
                float right = (x + w/2) * scaleX;
                float bottom = (y + h/2) * scaleY;

                // Clip to image bounds
                left = Math.max(0, left);
                top = Math.max(0, top);
                right = Math.min(originalWidth, right);
                bottom = Math.min(originalHeight, bottom);

                detections.add(new Detection(
                        new RectF(left, top, right, bottom),
                        maxConfidence,
                        classId
                ));
            }
        }

        // Apply Non-Max Suppression
        return nms(detections);
    }

    private List<Detection> nms(List<Detection> detections) {
        List<Detection> results = new ArrayList<>();
        PriorityQueue<Detection> priorityQueue = new PriorityQueue<>(
                50,
                (a, b) -> Float.compare(b.confidence, a.confidence)
        );

        priorityQueue.addAll(detections);

        while (!priorityQueue.isEmpty()) {
            Detection[] current = new Detection[]{priorityQueue.poll()};
            results.add(current[0]);

            List<Detection> removed = new ArrayList<>();
            for (Detection det : priorityQueue) {
                if (iou(current[0].rect, det.rect) > NMS_THRESHOLD) {
                    removed.add(det);
                }
            }
            priorityQueue.removeAll(removed);
        }

        return results;
    }

    private float iou(RectF a, RectF b) {
        float areaA = a.width() * a.height();
        float areaB = b.width() * b.height();

        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.right, b.right);
        float bottom = Math.min(a.bottom, b.bottom);

        if (right < left || bottom < top) return 0;

        float intersection = (right - left) * (bottom - top);
        return intersection / (areaA + areaB - intersection);
    }

    public String getClassName(int classId) {
        return classes[classId];
    }

    public int getClassColor(int classId) {
        return classColors[classId];
    }
    public void close() {
        interpreter.close();
    }




    private void runInBackground(Runnable task) {
        new Handler(Looper.getMainLooper()).post(task);
    }
    public void setUseNNAPI(boolean useNNAPI) {
//        runInBackground(() -> {
//            tfliteOptions.setUseNNAPI(useNNAPI);
////            recreateInterpreter();
//        });
    }

    public void setNumThreads(int numThreads) {
//        runInBackground(() -> {
//            tfliteOptions.setNumThreads(numThreads);
////            recreateInterpreter();
//        });
    }
}
