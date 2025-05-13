package org.tensorflow.lite.examples.detection;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.core.app.NotificationCompat;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.DetectorFactory;
import org.tensorflow.lite.examples.detection.tflite.YOLOClassifier;
import org.tensorflow.lite.examples.detection.tracking.DriverSafetyAlarmSystem;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class BackgroundCameraService extends Service {

    private static final String TAG = "BackgroundCameraService";
    private static final String CHANNEL_ID = "CameraServiceChannel";
    private static final int NOTIFICATION_ID = 1;

    private CameraManager cameraManager;
    private String cameraId;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private ImageReader imageReader;
    private DriverSafetyAlarmSystem safetyAlarmSystem;
    private PowerManager.WakeLock wakeLock;

    // Processing related variables
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private int previewWidth = 0;
    private int previewHeight = 0;

    // Binder for activity communication
    private final IBinder binder = new LocalBinder();
    private DetectionTracker detectionTracker;
    private YOLOClassifier model1;
    private YOLOClassifier model2;
    private int sensorOrientation;

    public class LocalBinder extends Binder {
        BackgroundCameraService getService() {
            return BackgroundCameraService.this;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    // Camera state callback
    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            Log.d(TAG, "Camera opened");
            cameraDevice = camera;
            createCaptureSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            Log.d(TAG, "Camera disconnected");
            camera.close();
            cameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            Log.e(TAG, "Camera error: " + error);
            camera.close();
            cameraDevice = null;
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();

        // Initialize safety alarm system
        safetyAlarmSystem = new DriverSafetyAlarmSystem(this);
        detectionTracker = new DetectionTracker();

        // Get wake lock to keep CPU running
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,
                "BackgroundCamera:WakeLock");
        wakeLock.acquire();

        // Start background thread
        startBackgroundThread();

        // Initialize camera
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        createNotificationChannel();
        Notification notification = buildNotification();
        startForeground(NOTIFICATION_ID, notification);

        try {
            openCamera();
        } catch (CameraAccessException e) {
            Log.e(TAG, "Could not open camera", e);
        }

        // If service gets killed, restart it
        return START_STICKY;
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Camera Background Service",
                    NotificationManager.IMPORTANCE_HIGH
            );
            channel.setDescription("Camera is being accessed in the background");
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(channel);
        }
    }

    private Notification buildNotification() {
        Intent notificationIntent = new Intent(this, CameraActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Camera Running")
                .setContentText("Background camera detection is active")
                .setSmallIcon(android.R.drawable.ic_menu_camera)
                .setContentIntent(pendingIntent)
                .build();
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (backgroundThread != null) {
            backgroundThread.quitSafely();
            try {
                backgroundThread.join();
                backgroundThread = null;
                backgroundHandler = null;
            } catch (InterruptedException e) {
                Log.e(TAG, "Error stopping background thread", e);
            }
        }
    }
    private static final Logger LOGGER = new Logger();

    private void openCamera() throws CameraAccessException {
        // Choose front-facing camera as in your original code
        try {

            model1 = DetectorFactory.getDetector(getAssets(), 0);
            model2 = DetectorFactory.getDetector(getAssets(), 1);


            LOGGER.i("tflite model intialized");
        } catch (final Exception e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");

        }
        for (String id : cameraManager.getCameraIdList()) {
            CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(id);
            sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
            Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);

            // Use front camera (same as your original code)
            if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                cameraId = id;
                break;
            }
        }

        if (cameraId == null) {
            // Fallback to first camera
            String[] cameraIds = cameraManager.getCameraIdList();
            if (cameraIds.length > 0) {
                cameraId = cameraIds[0];
            } else {
                Log.e(TAG, "No cameras available");
                return;
            }
        }

        // Set up image reader for processing frames
        CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);
        StreamConfigurationMap map = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
            Log.e(TAG, "Cannot get stream configuration map");
            return;
        }

        // Choose appropriate preview size (640x480 is a good starting point)
        Size[] outputSizes = map.getOutputSizes(ImageReader.class);
        Size previewSize = chooseBestSize(outputSizes, 1280, 720);
        previewWidth = previewSize.getWidth();
        previewHeight = previewSize.getHeight();

        Log.d(TAG, "Selected camera preview size: " + previewWidth + "x" + previewHeight);

        imageReader = ImageReader.newInstance(
                previewWidth, previewHeight, ImageFormat.YUV_420_888, 2);
        imageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
            @Override
            public void onImageAvailable(ImageReader reader) {
                Log.d(TAG, "onImageAvailable called - frame received"); // Debug log

                Image image = null;
                try {
                    image = reader.acquireLatestImage();
                    if (image == null) {
                        Log.w(TAG, "Null image received from camera");
                        return;
                    }

                    if (isProcessingFrame) {
                        image.close();
                        return;
                    }
                    isProcessingFrame = true;

                    // Process image here
                    Image.Plane[] planes = image.getPlanes();
                    fillBytes(planes, yuvBytes);
                    yRowStride = planes[0].getRowStride();

                    // Initialize RGB buffer if needed
                    if (rgbBytes == null) {
                        rgbBytes = new int[previewWidth * previewHeight];
                    }

                    // Convert YUV to RGB
                    ImageUtils.convertYUV420ToARGB8888(
                            yuvBytes[0],
                            yuvBytes[1],
                            yuvBytes[2],
                            previewWidth,
                            previewHeight,
                            yRowStride,
                            planes[1].getRowStride(),
                            planes[1].getPixelStride(),
                            rgbBytes);

                    // Process the RGB data
                    Log.d(TAG, "Processing frame - calling processImage()");
                    processImage(rgbBytes);

                } catch (Exception e) {
                    Log.e(TAG, "Exception processing image: " + e.getMessage(), e);
                    isProcessingFrame = false;
                } finally {
                    if (image != null) {
                        image.close();
                    }
                }
            }
        }, backgroundHandler);

        // Open camera
        Log.d(TAG, "Opening camera with ID: " + cameraId);
        cameraManager.openCamera(cameraId, stateCallback, backgroundHandler);
    }

    private void createCaptureSession() {
        try {
            if (cameraDevice == null || imageReader == null) {
                Log.e(TAG, "Cannot create capture session, camera or imageReader is null");
                return;
            }

            List<Surface> surfaces = new ArrayList<>();
            Surface readerSurface = imageReader.getSurface();
            surfaces.add(readerSurface);

            cameraDevice.createCaptureSession(surfaces, new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession session) {
                    if (cameraDevice == null) {
                        return;
                    }

                    captureSession = session;
                    try {
                        CaptureRequest.Builder requestBuilder = cameraDevice.createCaptureRequest(
                                CameraDevice.TEMPLATE_PREVIEW);
                        requestBuilder.addTarget(imageReader.getSurface());

                        // Auto focus
                        requestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

                        // Start capture
                        Log.d(TAG, "Starting camera capture session");
                        captureSession.setRepeatingRequest(
                                requestBuilder.build(), null, backgroundHandler);

                    } catch (CameraAccessException e) {
                        Log.e(TAG, "Camera access exception in createCaptureSession", e);
                    }
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                    Log.e(TAG, "Failed to configure camera capture session");
                }
            }, backgroundHandler);

        } catch (CameraAccessException e) {
            Log.e(TAG, "Exception creating capture session", e);
        }
    }

    private Size chooseBestSize(Size[] sizes, int idealWidth, int idealHeight) {
        if (sizes == null || sizes.length == 0) {
            return new Size(idealWidth, idealHeight);
        }

        Size bestSize = sizes[0];
        int bestScore = Integer.MAX_VALUE;

        for (Size size : sizes) {
            int score = Math.abs(size.getWidth() - idealWidth) +
                    Math.abs(size.getHeight() - idealHeight);
            if (score < bestScore) {
                bestScore = score;
                bestSize = size;
            }
        }

        return bestSize;
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    private void processImage(int[] rgbBytes) {
        final long currTimestamp = System.currentTimeMillis();

        Bitmap rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

        int cropSize = 720;
        Bitmap croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        
        

        Matrix frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation, true
        );

        Matrix cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        backgroundHandler.post(() -> {
            try {
                ExecutorService executor = Executors.newFixedThreadPool(2);
                Future<List<Classifier.Recognition>> future1 = executor.submit(() -> model1.recognizeImage(croppedBitmap));
                Future<List<Classifier.Recognition>> future2 = executor.submit(() -> model2.recognizeImage(croppedBitmap));

                List<Classifier.Recognition> results = new ArrayList<>();
                results.addAll(future1.get());
                results.addAll(future2.get());
                executor.shutdown();

                for (Classifier.Recognition result : results) {
                    RectF location = result.getLocation();
                    if (location != null) {
                        cropToFrameTransform.mapRect(location);
                        result.setLocation(location);
                        Log.d(TAG, "Detected: " + result.getTitle() + " Conf: " + result.getConfidence());
                    }
                }

                if (safetyAlarmSystem != null) {
                    safetyAlarmSystem.processDetections(results, currTimestamp);
                }

            } catch (Exception e) {
                Log.e(TAG, "Error during detection: " + e.getMessage(), e);
            } finally {
                isProcessingFrame = false;
            }
        });
    }

    /**
     * Called when the service is being removed from recent apps
     */
    @Override
    public void onTaskRemoved(Intent rootIntent) {
        Log.d(TAG, "Service task removed (app cleared from recent apps)");

        // Make sure camera is properly stopped
        closeCamera();

        // Stop the service itself
        stopSelf();

        super.onTaskRemoved(rootIntent);
    }

    @Override
    public void onDestroy() {
        // Clean up resources
        closeCamera();
        stopBackgroundThread();

        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        super.onDestroy();

    }

    private void closeCamera() {

        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }

        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }

        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }
}