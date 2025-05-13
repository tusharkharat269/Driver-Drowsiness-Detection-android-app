
package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.examples.detection.tracking.DriverSafetyAlarmSystem;

/**
 * Combined Driver Safety Application that integrates both
 * regular camera detection and background camera service capabilities
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CAMERA = 100;
    private static final int PERMISSION_REQUEST_BACKGROUND_LOCATION = 101;

    // UI Elements
    private Button startForegroundButton;
    private Button stopForegroundButton;
    private Button startDetectionButton;
    private Button testAlertButton;
    private TextView serviceStatusTextView;

    // Background Service connection components
    private BackgroundCameraService cameraService;
    private boolean isServiceBound = false;

    // Service connection for background camera operations
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            BackgroundCameraService.LocalBinder binder = (BackgroundCameraService.LocalBinder) service;
            cameraService = binder.getService();
            isServiceBound = true;
            updateServiceUI(true);
            Log.d(TAG, "Service connected");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            isServiceBound = false;
            updateServiceUI(false);
            Log.d(TAG, "Service disconnected");
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        startForegroundButton = findViewById(R.id.startForegroundButton);
        stopForegroundButton = findViewById(R.id.stopForegroundButton);
        startDetectionButton = findViewById(R.id.startDetectionButton);
        testAlertButton = findViewById(R.id.testAlertButton);
        serviceStatusTextView = findViewById(R.id.serviceStatusTextView);

        // Set up button click listeners for background service
        startForegroundButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkAndRequestPermissions();
            }
        });

        stopForegroundButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopCameraService();
            }
        });

        // Button to launch the full camera detection activity
        startDetectionButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startCameraDetectionActivity();
            }
        });

        // Test alert button functionality
        testAlertButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                testAlert();
            }
        });

        // Update UI initially
        updateServiceUI(false);
    }

    /**
     * Updates the UI based on the service running state
     */
    private void updateServiceUI(boolean isRunning) {
        startForegroundButton.setEnabled(!isRunning);
        stopForegroundButton.setEnabled(isRunning);
        serviceStatusTextView.setText(isRunning ?
                "Background Camera Service: Running" :
                "Background Camera Service: Stopped");
    }

    /**
     * Check and request necessary permissions before starting service
     */
    private void checkAndRequestPermissions() {
        boolean cameraPermissionGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;

        if (!cameraPermissionGranted) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    PERMISSION_REQUEST_CAMERA);
            return;
        }

        // For Android 10 and above, request background location permission if needed
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            boolean backgroundLocationGranted = ContextCompat.checkSelfPermission(
                    this, Manifest.permission.ACCESS_BACKGROUND_LOCATION)
                    == PackageManager.PERMISSION_GRANTED;

            if (!backgroundLocationGranted) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.ACCESS_BACKGROUND_LOCATION},
                        PERMISSION_REQUEST_BACKGROUND_LOCATION);
                return;
            }
        }

        // All permissions granted, start the service
        startCameraService();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case PERMISSION_REQUEST_CAMERA:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Check next permission or start service
                    checkAndRequestPermissions();
                } else {
                    Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show();
                }
                break;
            case PERMISSION_REQUEST_BACKGROUND_LOCATION:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    startCameraService();
                } else {
                    Toast.makeText(this,
                            "Background location permission required for full functionality",
                            Toast.LENGTH_LONG).show();
                    // You might still want to start the service with limited functionality
                    startCameraService();
                }
                break;
        }
    }

    /**
     * Start the background camera service
     */
    private void startCameraService() {
        Intent serviceIntent = new Intent(this, BackgroundCameraService.class);

        // Start the service as a foreground service
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent);
        } else {
            startService(serviceIntent);
        }

        // Bind to the service
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE);

        Toast.makeText(this, "Background camera service started", Toast.LENGTH_SHORT).show();
    }

    /**
     * Stop the background camera service
     */
    private void stopCameraService() {
        if (isServiceBound) {
            // Unbind from the service
            unbindService(serviceConnection);
            isServiceBound = false;
        }

        // Stop the service
        Intent serviceIntent = new Intent(this, BackgroundCameraService.class);
        stopService(serviceIntent);

        updateServiceUI(false);
        Toast.makeText(this, "Background camera service stopped", Toast.LENGTH_SHORT).show();
    }

    /**
     * Launch the camera detection activity
     */
    private void startCameraDetectionActivity() {
        // Launch the original CameraActivity implementation
        Intent intent = new Intent(this, DetectorActivity.class);
        startActivity(intent);
    }

    /**
     * Test the alert functionality
     */
    private void testAlert() {
        if (isServiceBound && cameraService != null) {
            // Use the service's alert method if available
//            cameraService.testAlert();
            Log.i(TAG,"test alert not working");
        } else {
            // Create a temporary driver safety alarm system for testing
            DriverSafetyAlarmSystem safetyAlarmSystem = new DriverSafetyAlarmSystem(this);
            safetyAlarmSystem.playSoundAlert("Testing alarm System");
            Toast.makeText(this, "Testing alert System.....", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopCameraService();
        // Make sure to unbind from service to prevent leaks
        if (isServiceBound) {
            unbindService(serviceConnection);
            isServiceBound = false;
        }
    }
}