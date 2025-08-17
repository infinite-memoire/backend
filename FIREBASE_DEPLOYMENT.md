# Firebase Security Rules Deployment Guide

## Overview
This guide explains how to deploy the Firebase security rules to fix the issue where users cannot save audio recordings without selecting a chapter.

## Files Created
- `firestore.rules` - Firestore database security rules
- `storage.rules` - Firebase Storage security rules  
- `firebase.json` - Firebase project configuration
- `firestore.indexes.json` - Database indexes for performance

## Prerequisites
1. Install Firebase CLI:
   ```bash
   npm install -g firebase-tools
   ```

2. Login to Firebase:
   ```bash
   firebase login
   ```

3. Initialize Firebase project (if not already done):
   ```bash
   firebase init
   ```
   - Select Firestore and Storage
   - Choose your existing Firebase project
   - Accept the default files (they will be overwritten by our rules)

## Deployment Steps

### 1. Deploy Firestore Rules
```bash
firebase deploy --only firestore:rules
```

### 2. Deploy Storage Rules
```bash
firebase deploy --only storage
```

### 3. Deploy Both Together
```bash
firebase deploy --only firestore:rules,storage
```

### 4. Deploy Everything (including indexes)
```bash
firebase deploy
```

## Security Rules Summary

### Firestore Rules (`firestore.rules`)
- **User Documents**: Users can read/write their own user document (`/users/{userId}`)
- **Chapters**: Users can create/manage chapters under their user document, including "Uncategorized"
- **Recordings**: Users can create/manage recordings within their chapters
- **Audio Files**: Users can manage their audio files
- **Global Collections**: Authenticated users can access global collections (audio_files, transcripts, etc.)

### Storage Rules (`storage.rules`)
- **User Audio**: Users can upload/read audio files in `/users/{userId}/audio/`
- **Chapter Recordings**: Users can upload/read recordings in `/users/{userId}/chapters/{chapterId}/recordings/`
- **Global Storage**: Authenticated users can access global audio and recordings paths

## Testing the Rules

### 1. Using Firebase Emulator (Local Testing)
```bash
# Start emulators
firebase emulators:start

# Your app should connect to:
# Firestore: localhost:8080
# Storage: localhost:9199
# UI: localhost:4000
```

### 2. Production Testing
After deploying rules to production, test by:
1. Authenticating a user in your frontend
2. Trying to save a recording without selecting a chapter
3. Verify the "Uncategorized" chapter is created automatically
4. Confirm the recording is saved successfully

## Troubleshooting

### Common Issues
1. **Permission Denied**: Ensure user is properly authenticated
2. **Rules Not Applied**: Wait 1-2 minutes after deployment for rules to propagate
3. **Path Mismatch**: Verify your frontend uses the correct Firestore paths matching the rules

### Debugging Rules
Use the Firebase Console Rules Playground to test specific operations:
1. Go to Firebase Console → Firestore → Rules
2. Click "Rules Playground"
3. Test read/write operations with different user IDs

## Frontend Path Requirements

Ensure your frontend saves data using these paths:
- **User Document**: `/users/{userId}`
- **Chapters**: `/users/{userId}/chapters/{chapterId}` 
- **Recordings**: `/users/{userId}/chapters/{chapterId}/recordings/{recordingId}`
- **Storage**: `/users/{userId}/chapters/{chapterId}/recordings/{filename}`

The "Uncategorized" chapter should use a consistent ID like `"uncategorized"` or generate a UUID.
