package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.Build;
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

public class BackgroundCameraActivity extends AppCompatActivity{

    private static final String TAG = "CameraActivity";
    private static final int PERMISSION_REQUEST_CAMERA = 100;
    private static final int PERMISSION_REQUEST_BACKGROUND_LOCATION = 101;

    private BackgroundCameraService cameraService;
    private boolean isServiceBound = false;
    private Button startServiceButton;
    private Button stopServiceButton;
    private TextView serviceStatusTextView;

    // Define the connection to the service
    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            BackgroundCameraService.LocalBinder binder = (BackgroundCameraService.LocalBinder) service;
            cameraService = binder.getService();
            isServiceBound = true;
            updateUI(true);
            Log.d(TAG, "Service connected");
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            isServiceBound = false;
            updateUI(false);
            Log.d(TAG, "Service disconnected");
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        // Initialize views
        startServiceButton = findViewById(R.id.startServiceButton);
        stopServiceButton = findViewById(R.id.stopServiceButton);
        serviceStatusTextView = findViewById(R.id.serviceStatusTextView);

        // Set up button click listeners
        startServiceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkAndRequestPermissions();
            }
        });

        stopServiceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                stopCameraService();
            }
        });

        // Update UI initially
        updateUI(false);
    }

    private void updateUI(boolean isRunning) {
        startServiceButton.setEnabled(!isRunning);
        stopServiceButton.setEnabled(isRunning);
        serviceStatusTextView.setText(isRunning ?
                "Camera Service: Running" :
                "Camera Service: Stopped");
    }

    private void checkAndRequestPermissions() {
        boolean cameraPermissionGranted = ContextCompat.checkSelfPermission(
                this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;

        if (!cameraPermissionGranted) {
            ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.CAMERA},
                    PERMISSION_REQUEST_CAMERA);
            return;
        }

        // For Android 10 and above, request background location permission if needed
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
            boolean backgroundLocationGranted = ContextCompat.checkSelfPermission(
                    this, android.Manifest.permission.ACCESS_BACKGROUND_LOCATION)
                    == PackageManager.PERMISSION_GRANTED;

            if (!backgroundLocationGranted) {
                ActivityCompat.requestPermissions(this,
                        new String[]{android.Manifest.permission.ACCESS_BACKGROUND_LOCATION},
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

        if (requestCode == PERMISSION_REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Check next permission or start service
                checkAndRequestPermissions();
            } else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == PERMISSION_REQUEST_BACKGROUND_LOCATION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCameraService();
            } else {
                Toast.makeText(this,
                        "Background location permission required for full functionality",
                        Toast.LENGTH_LONG).show();
                // You might still want to start the service with limited functionality
                startCameraService();
            }
        }
    }

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

        Toast.makeText(this, "Camera service started", Toast.LENGTH_SHORT).show();
    }

    private void stopCameraService() {
        if (isServiceBound) {
            // Unbind from the service
            unbindService(serviceConnection);
            isServiceBound = false;
        }

        // Stop the service
        Intent serviceIntent = new Intent(this, BackgroundCameraService.class);
        stopService(serviceIntent);

        updateUI(false);
        Toast.makeText(this, "Camera service stopped", Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        // Make sure to unbind from service to prevent leaks
        if (isServiceBound) {
            unbindService(serviceConnection);
            isServiceBound = false;
        }
    }
}