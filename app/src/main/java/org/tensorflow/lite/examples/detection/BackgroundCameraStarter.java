package org.tensorflow.lite.examples.detection;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;

public class BackgroundCameraStarter {
    private static final String TAG = "BackgroundCameraStarter";

    private Context context;
    private BackgroundCameraService cameraService;
    private boolean isBound = false;

    private ServiceConnection serviceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            BackgroundCameraService.LocalBinder binder =
                    (BackgroundCameraService.LocalBinder) service;
            cameraService = binder.getService();
            isBound = true;
            Log.d(TAG, "Background camera service connected");
            Toast.makeText(context, "Background Camera Started", Toast.LENGTH_SHORT).show();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            cameraService = null;
            isBound = false;
            Log.d(TAG, "Background camera service disconnected");
            Toast.makeText(context, "Background Camera Stopped", Toast.LENGTH_SHORT).show();
        }
    };

    public BackgroundCameraStarter(Context context) {
        this.context = context;
    }

    public void startBackgroundCamera() {
        // Create service intent
        Intent serviceIntent = new Intent(context, BackgroundCameraService.class);

        // Start service based on Android version
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent);
        } else {
            context.startService(serviceIntent);
        }

        // Bind to service
        context.bindService(
                serviceIntent,
                serviceConnection,
                Context.BIND_AUTO_CREATE
        );
    }

    public void stopBackgroundCamera() {
        // Unbind service if bound
        if (isBound) {
            context.unbindService(serviceConnection);
            isBound = false;
        }

        // Stop the service
        Intent serviceIntent = new Intent(context, BackgroundCameraService.class);
        context.stopService(serviceIntent);
    }

    // Check if background camera service is running
    public boolean isRunning() {
        return isBound && cameraService != null;
    }

    // Optional method to get the service instance
    public BackgroundCameraService getService() {
        return cameraService;
    }

}