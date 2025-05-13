package org.tensorflow.lite.examples.detection;

import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.examples.detection.tflite.Classifier.Recognition;

import java.util.ArrayList;
import java.util.List;

public class DetectionTracker {
    private static final String TAG = "DetectionTracker";

    public static class TrackedDetection {
        private Recognition recognition;
        private float alpha = 0.7f; // Reduced EMA factor for smoother transitions
        private int trackId;
        private int missedFrames = 0;
        private int maxMissedFrames = 2; // Increased persistence
        private RectF velocity = new RectF(0, 0, 0, 0); // Track velocity for prediction
        private RectF lastBox; // Store previous position
        private float confidenceDecayRate = 0.25f; // Confidence decay when missed

        public TrackedDetection(Recognition recognition, int trackId) {
            this.recognition = recognition;
            this.trackId = trackId;
            this.lastBox = new RectF(recognition.getLocation());
        }

        public boolean update(Recognition newDetection) {
            if (newDetection == null) {
                missedFrames++;

                // Apply velocity prediction when detection is missing
                if (missedFrames <= 5) { // Only predict for a few frames
                    RectF box = recognition.getLocation();

                    // Apply velocity to predict new location
                    box.left += velocity.left;
                    box.top += velocity.top;
                    box.right += velocity.right;
                    box.bottom += velocity.bottom;

                    // Constrain to reasonable limits
                    ensureValidBox(box);

                    // Decay confidence each missed frame
                    recognition.setConfidence(recognition.getConfidence() * confidenceDecayRate);
                }

                return missedFrames <= maxMissedFrames;
            }

            // Detection found, reset counter
            missedFrames = 0;

            // Calculate velocity (movement between frames)
            RectF oldBox = recognition.getLocation();
            RectF newBox = newDetection.getLocation();

            // Calculate velocities (movement per frame)
            velocity.left = newBox.left - lastBox.left;
            velocity.top = newBox.top - lastBox.top;
            velocity.right = newBox.right - lastBox.right;
            velocity.bottom = newBox.bottom - lastBox.bottom;

            // Dampen velocity for stability (optional)
            velocity.left *= 0.8f;
            velocity.top *= 0.8f;
            velocity.right *= 0.8f;
            velocity.bottom *= 0.8f;

            // Store current box for next velocity calculation
            lastBox.set(newBox);

            // Adaptive EMA smoothing - less smoothing for high confidence detections
            float adaptiveAlpha = alpha;
            if (newDetection.getConfidence() > 0.2f) {
                adaptiveAlpha = Math.max(0.5f, alpha); // Less smoothing for high confidence
            } else if (newDetection.getConfidence() < 0.05f) {
                adaptiveAlpha = Math.min(0.9f, alpha + 0.1f); // More smoothing for low confidence
            }

            // Update tracking with EMA smoothing
            oldBox.left = adaptiveAlpha * oldBox.left + (1 - adaptiveAlpha) * newBox.left;
            oldBox.top = adaptiveAlpha * oldBox.top + (1 - adaptiveAlpha) * newBox.top;
            oldBox.right = adaptiveAlpha * oldBox.right + (1 - adaptiveAlpha) * newBox.right;
            oldBox.bottom = adaptiveAlpha * oldBox.bottom + (1 - adaptiveAlpha) * newBox.bottom;

            // Ensure box stays valid
            ensureValidBox(oldBox);

            // Smooth confidence with bias toward higher values
            float oldConf = recognition.getConfidence();
            float newConf = newDetection.getConfidence();

            // Prefer higher confidence (asymmetric smoothing)
            if (newConf > oldConf) {
                // Rise quickly
                recognition.setConfidence(0.3f * oldConf + 0.7f * newConf);
            } else {
                // Fall slowly
                recognition.setConfidence(0.8f * oldConf + 0.2f * newConf);
            }

            return true;
        }

        // Ensure box has valid dimensions and is properly formed
        private void ensureValidBox(RectF box) {
            // Minimum box dimensions (prevent collapse)
            float minWidth = 10;
            float minHeight = 10;

            // If box width is too small, expand from center
            if (box.width() < minWidth) {
                float center = (box.left + box.right) / 2;
                box.left = center - minWidth/2;
                box.right = center + minWidth/2;
            }

            // If box height is too small, expand from center
            if (box.height() < minHeight) {
                float center = (box.top + box.bottom) / 2;
                box.top = center - minHeight/2;
                box.bottom = center + minHeight/2;
            }

            // Ensure proper order (left < right, top < bottom)
            if (box.left > box.right) {
                float temp = box.left;
                box.left = box.right;
                box.right = temp;
            }

            if (box.top > box.bottom) {
                float temp = box.top;
                box.top = box.bottom;
                box.bottom = temp;
            }
        }

        public Recognition getRecognition() {
            return recognition;
        }

        public int getTrackId() {
            return trackId;
        }

        public int getMissedFrames() {
            return missedFrames;
        }
    }

    private List<TrackedDetection> trackedObjects = new ArrayList<>();
    private int nextTrackId = 1;
    private float iouMatchThreshold = 0.3f;  // Lowered to make tracking more persistent
    private boolean useClassMatching = true; // Match only objects of same class
    private int frameWidth = 0;
    private int frameHeight = 0;

    // Set frame dimensions for boundary enforcement
    public void setFrameDimensions(int width, int height) {
        this.frameWidth = width;
        this.frameHeight = height;
        Log.d(TAG, "Frame dimensions set: " + width + "x" + height);
    }

    public List<Recognition> updateTrackedObjects(List<Recognition> newDetections) {
        // Match detections with existing tracks
        boolean[] matched = new boolean[newDetections.size()];
        List<TrackedDetection> updatedTracks = new ArrayList<>();

        // First, update existing tracks with matched detections
        for (TrackedDetection track : trackedObjects) {
            Recognition trackBox = track.getRecognition();

            // Find best matching detection
            int bestMatch = -1;
            float bestIoU = iouMatchThreshold;

            for (int i = 0; i < newDetections.size(); i++) {
                if (matched[i]) continue;

                Recognition detection = newDetections.get(i);

                // Match only if same class when enabled
                if (useClassMatching && trackBox.getDetectedClass() != detection.getDetectedClass()) {
                    continue;
                }

                // Calculate IoU between current track and this detection
                float iou = calculateIoU(trackBox.getLocation(), detection.getLocation());

                // Check if this is the best match so far
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatch = i;
                }
            }

            // Update track with the best match
            boolean stillValid;
            if (bestMatch >= 0) {
                // Mark this detection as matched
                matched[bestMatch] = true;

                // Update the track with the matched detection
                stillValid = track.update(newDetections.get(bestMatch));
            } else {
                // No match found, update with null to mark as missed
                stillValid = track.update(null);
            }

            // Keep only valid tracks
            if (stillValid) {
                updatedTracks.add(track);
            }
        }

        // Create new tracks for unmatched detections
        for (int i = 0; i < newDetections.size(); i++) {
            if (!matched[i]) {
                // Skip low confidence detections when creating new tracks
                Recognition detection = newDetections.get(i);
                if (detection.getConfidence() < 0.1f) { // Higher threshold for new tracks
                    continue;
                }

                // Create a new track
                TrackedDetection newTrack = new TrackedDetection(detection, nextTrackId++);
                updatedTracks.add(newTrack);

                Log.d(TAG, "Created new track ID " + (nextTrackId-1) +
                        " for class " + detection.getTitle() +
                        " with confidence " + detection.getConfidence());
            }
        }

        // Update the list of tracked objects
        trackedObjects = updatedTracks;

        // Create list of recognitions with track IDs
        List<Recognition> result = new ArrayList<>();
        for (TrackedDetection track : trackedObjects) {
            Recognition rec = track.getRecognition();

            // Tag recognition with track ID
            String title = rec.getTitle() + " #" + track.getTrackId();

            // Create a new recognition with the track ID embedded in title
            Recognition trackedRec = new Recognition(
                    rec.getId(),
                    title,
                    rec.getConfidence(),
                    rec.getLocation(),
                    rec.getDetectedClass()
            );

            result.add(trackedRec);
        }

        // Log tracking status
        Log.d(TAG, "Tracking " + trackedObjects.size() + " objects");

        return result;
    }

    private float calculateIoU(RectF boxA, RectF boxB) {
        // Calculate area of intersection
        float xOverlap = Math.max(0, Math.min(boxA.right, boxB.right) - Math.max(boxA.left, boxB.left));
        float yOverlap = Math.max(0, Math.min(boxA.bottom, boxB.bottom) - Math.max(boxA.top, boxB.top));
        float intersectionArea = xOverlap * yOverlap;

        // Calculate area of both bounding boxes
        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);

        // Calculate area of union
        float unionArea = boxAArea + boxBArea - intersectionArea;

        // Return IoU
        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    // Helper method to enforce frame boundaries for all tracks
    public void enforceFrameBoundaries() {
        if (frameWidth <= 0 || frameHeight <= 0) return;

        for (TrackedDetection track : trackedObjects) {
            RectF box = track.getRecognition().getLocation();

            // Constrain box to frame dimensions
            box.left = Math.max(0, Math.min(box.left, frameWidth - 1));
            box.top = Math.max(0, Math.min(box.top, frameHeight - 1));
            box.right = Math.max(box.left + 1, Math.min(box.right, frameWidth));
            box.bottom = Math.max(box.top + 1, Math.min(box.bottom, frameHeight));
        }
    }

    // Reset tracking (e.g., when switching cameras)
    public void reset() {
        trackedObjects.clear();
        nextTrackId = 1;
        Log.d(TAG, "Tracking reset");
    }
}