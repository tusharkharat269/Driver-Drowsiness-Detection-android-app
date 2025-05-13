package org.tensorflow.lite.examples.detection.tracking;

//import android.app.Notification;
//import android.app.NotificationChannel;
//import android.app.NotificationManager;
import android.content.Context;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.widget.Toast;

//import androidx.core.app.NotificationCompat;

import org.tensorflow.lite.examples.detection.R;
import org.tensorflow.lite.examples.detection.tflite.Classifier;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * A system that tracks detections over time and triggers alarms based on
 * configurable thresholds for each detection class.
 */
public class DriverSafetyAlarmSystem {
    private static final String TAG = "DriverSafetyAlarm";

    // Constants for notification channels
    private static final String CHANNEL_ID = "driver_safety_channel";
    private static final String CHANNEL_NAME = "Driver Safety Alerts";
    private static final int NOTIFICATION_ID = 1001;


    // Detection classes
    public static final String CLASS_DISTRACTED = "distracted";
    public static final String CLASS_DROWSY = "drowsy";
    public static final String CLASS_PHONE = "phone";
    public static final String CLASS_SMOKING = "smoking";
    public static final String CLASS_YAWNING = "yawning";

    // Cooldown periods to avoid alert spamming (in milliseconds)
    private static final long ALERT_COOLDOWN_MS = 10000;        // 10 seconds between alerts of the same type

    private static final float confidence = 0.1f;

    // Context for notifications and sounds
    private final Context context;

    // Media player for playing alert sounds
    private MediaPlayer mediaPlayer;

    // Handler for posting to main thread
    private final Handler mainHandler;

    // Detection state tracking
    public Map<String, detectionStates> detectionMap;

    // Last alert timestamps to implement cooldowns
    private final Map<String, Long> lastAlertTimes;


    // Notification manager
//    private NotificationManager notificationManager;

    // Track if system is currently enabled
    private boolean isEnabled = true;
    private MediaPlayer alertPlayer;

    static class detectionStates{
        ArrayDeque<Long> timestamps = new ArrayDeque<>();
        float window_threshold;
        Long lastAlertTime;

        detectionStates(float window_threshold){
            this.window_threshold = window_threshold;
        }

    }

    /**
     * Constructor
     * @param context Application context
     */
    public DriverSafetyAlarmSystem(Context context) {
        this.context = context;
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.detectionMap = new HashMap<>();
        this.lastAlertTimes = new HashMap<>();

        initializeMediaPlayer();

        detectionMap = new HashMap<>();
        detectionMap.put(CLASS_DISTRACTED, new detectionStates(0f));
        detectionMap.put(CLASS_DROWSY, new detectionStates(0f));
        detectionMap.put(CLASS_PHONE, new detectionStates(0f));
        detectionMap.put(CLASS_SMOKING, new detectionStates(0f));
        detectionMap.put(CLASS_YAWNING, new detectionStates(0f));


        // Create notification channel
//        createNotificationChannel();
        Log.i(TAG, "DriverSafetyAlarmSystem intitated..................................................");
    }

    /**
     * Enable or disable the alarm system
     * @param enabled True to enable, false to disable
     */

    /**
     * Process detected objects to check for alarming conditions
//     * @param recognitions List of detected objects
     * @param currTimestamp Current timestamp
     */
    public void processDetections(List<Classifier.Recognition> res, long currTimestamp) {
        if (res == null || res.isEmpty()) return;
        if (!isEnabled) return;

        for (Classifier.Recognition detected : res) {
            String className = detected.getTitle().toLowerCase().split(" #")[0];
            if (!detectionMap.containsKey(className)) continue;

            // Initialize deque if missing
            detectionStates classState = detectionMap.get(className);
            if (classState == null) continue;
            ArrayDeque<Long> timestamps = classState.timestamps;

            synchronized(timestamps){
                timestamps.add(currTimestamp);
                Log.e(TAG, className + ">>>" + timestamps.toString());


                Iterator<Long> iterator = timestamps.iterator();
                //            long totalTime = 0;
                while (iterator.hasNext()) {
                    Long timestamp = iterator.next();
                    if (currTimestamp - timestamp > TimeUnit.MINUTES.toMillis(1)) {
                        iterator.remove();
                    } else {
                        // If we find a timestamp that's newer than the cutoff, we can stop removing
                        // since ArrayDeque is ordered by insertion time
                        break;
                    }
                }
                Log.i(TAG, "Detections in last minute for " + className + ": " + timestamps.size());
                if (timestamps.size() > 15) {
                    if (classState.lastAlertTime != null && currTimestamp - classState.lastAlertTime < ALERT_COOLDOWN_MS) {
                        // Still in cooldown period, don't trigger again
                        Log.i(TAG, "not cooldown yet -----------------------------------------------------");
                        continue;
                    }
                    // Update last alert time
                    classState.lastAlertTime = currTimestamp;
                    triggerAlert(className, getAlertMessageForClass(className));
                }
            }
        }
    }


    private String getAlertMessageForClass(String className) {
        switch (className) {
            case CLASS_DISTRACTED:
                return "Distraction detected! Please focus on the road.";
            case CLASS_DROWSY:
                return "Drowsiness detected! Please take a break.";
            case CLASS_PHONE:
                return "Phone usage detected! Keep your hands on the wheel.";
            case CLASS_SMOKING:
                return "Smoking detected! This may impair your driving.";
            default:
                return "Unsafe driving behavior detected!";
        }
    }

    /**
     * Trigger an alert for a specific detection class
     */
    private void triggerAlert(String className, String message) {

        Log.i(TAG, "Triggering alert: " + className + " - " + message);

        // Play sound and show notification
        playSoundAlert(className);
        vibrate();
//        showNotification(className, message);

        // Show toast on UI thread
        mainHandler.post(() -> Toast.makeText(context, message, Toast.LENGTH_LONG).show());
    }

    public void playSoundAlert(String message) {
        try {
            // Play sound
            if (alertPlayer != null && !alertPlayer.isPlaying()) {
                alertPlayer.seekTo(0);
                alertPlayer.start();
            }

            // Show toast
//            mainHandler.post(() -> Toast.makeText(context, message, Toast.LENGTH_LONG).show());

        } catch (Exception e) {
            Log.e(TAG, "Alert error: " + e.getMessage());
        }
    }

    private void initializeMediaPlayer() {
        try {
            alertPlayer = MediaPlayer.create(context, R.raw.alarm);
            alertPlayer.setLooping(false);
            Log.d(TAG, "MediaPlayer initialized");
        } catch (Exception e) {
            Log.e(TAG, "MediaPlayer init failed: " + e.getMessage());
        }
    }

    public void vibrate() {
        try {
            Vibrator vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
            if (vibrator == null || !vibrator.hasVibrator()) {
                return;
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                vibrator.vibrate(1000);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error during vibration", e);
        }
    }

    /**
     * Play appropriate sound alert for detection class
     */
//    private void playSoundAlert(String className) {
//        // Release previous media player if exists
//        if (mediaPlayer != null) {
//            mediaPlayer.release();
//        }
//
//        // Select sound resource
//        int soundResource;
//        switch (className) {
//            case CLASS_DISTRACTED:
//                soundResource = SOUND_DISTRACTED;
//                break;
//            case CLASS_DROWSY:
//            case CLASS_HEAD_DROP:
//            case CLASS_CLOSED_EYES:
//                soundResource = SOUND_DROWSY;
//                break;
//            case CLASS_PHONE:
//                soundResource = SOUND_PHONE;
//                break;
//            case CLASS_SMOKING:
//                soundResource = SOUND_SMOKING;
//                break;
//            case CLASS_YAWNING:
//                soundResource = SOUND_YAWNING;
//                break;
//            default:
//                // Fallback to default system alert sound
//                try {
//                    Uri defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
//                    mediaPlayer = MediaPlayer.create(context, defaultSoundUri);
//                    mediaPlayer.start();
//                } catch (Exception e) {
//                    Log.e(TAG, "Error playing default alert sound", e);
//                }
//                return;
//        }
//
//        try {
//            // Play selected sound resource
//            mediaPlayer = MediaPlayer.create(context, soundResource);
//            if (mediaPlayer != null) {
//                mediaPlayer.setOnCompletionListener(MediaPlayer::release);
//                mediaPlayer.start();
//            }
//        } catch (Exception e) {
//            Log.e(TAG, "Error playing alert sound", e);
//        }
//    }

    /**
     * Show notification for alert
     */
//    private void showNotification(String className, String message) {
//        if (notificationManager == null) {
//            notificationManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
//        }
//
//        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ID)
//                .setSmallIcon(R.drawable.ic_notification)
//                .setContentTitle("Driver Safety Alert")
//                .setContentText(message)
//                .setPriority(NotificationCompat.PRIORITY_HIGH)
//                .setCategory(NotificationCompat.CATEGORY_ALARM)
//                .setAutoCancel(true);
//
//        // Make it high priority on android 7 and below
//        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
//            builder.setPriority(Notification.PRIORITY_HIGH);
//        }
//
//        notificationManager.notify(NOTIFICATION_ID, builder.build());
//    }
//
//    /**
//     * Create notification channel (required for Android O and above)
//     */
//    private void createNotificationChannel() {
//        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//            NotificationChannel channel = new NotificationChannel(
//                    CHANNEL_ID,
//                    CHANNEL_NAME,
//                    NotificationManager.IMPORTANCE_HIGH);
//
//            channel.setDescription("Driver safety alerts and warnings");
//            channel.enableVibration(true);
//
//            // Configure sound
//            AudioAttributes audioAttributes = new AudioAttributes.Builder()
//                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
//                    .setUsage(AudioAttributes.USAGE_ALARM)
//                    .build();
//            channel.setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION), audioAttributes);
//
//            NotificationManager notificationManager = context.getSystemService(NotificationManager.class);
//            if (notificationManager != null) {
//                notificationManager.createNotificationChannel(channel);
//            }
//        }
//    }

    /**
     * Vibrate device for haptic feedback
     */


    /**
     * Release resources
     */
    public void release() {
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }

    /**
     * Class to track detection state for a specific detection class
     */

}