# Error Handling Documentation

## Overview

This document describes the comprehensive error handling implemented in the Fetal Head Segmentation demo application.

---

## Backend Error Handling

### 1. Invalid Image Format

**Scenario**: User uploads a non-image file or corrupted image

**Implementation**:

- File validation in upload endpoint
- PIL Image.verify() to detect corrupted files
- Re-opening file after verification

**Response**:

```json
{
  "success": false,
  "error": "Corrupted or invalid image file. Please upload a valid image."
}
```

**Status Code**: 400 Bad Request

---

### 2. File Too Large

**Scenario**: User uploads a file larger than 16 MB

**Implementation**:

- Flask config: `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024`
- Automatic rejection by Flask
- Custom error handler for 413 status

**Response**:

```json
{
  "success": false,
  "error": "File too large. Maximum size is 16 MB."
}
```

**Status Code**: 413 Request Entity Too Large

---

### 3. Corrupted Image Data

**Scenario**: Image file is partially corrupted or in invalid format

**Implementation**:

- PIL verification before processing
- Separate try-catch blocks for image operations
- Format conversion error handling

**Response**:

```json
{
  "success": false,
  "error": "Failed to convert image format: <specific error>"
}
```

**Status Code**: 400 Bad Request

---

### 4. Model Inference Failure

**Scenario**: Model fails during inference (GPU memory, invalid dimensions, etc.)

**Implementation**:

- Try-catch around inference engine call
- Separate handling for RuntimeError (GPU issues)
- Generic exception handler for other errors

**Response**:

```json
{
  "success": false,
  "error": "Model inference failed. This may be due to GPU memory issues or invalid image dimensions."
}
```

**Status Code**: 500 Internal Server Error

---

### 5. Visualization Creation Failure

**Scenario**: Error creating overlay visualization

**Implementation**:

- Try-catch around create_overlay function
- Prevents partial success (inference works but visualization fails)

**Response**:

```json
{
  "success": false,
  "error": "Failed to create visualization: <specific error>"
}
```

**Status Code**: 500 Internal Server Error

---

### 6. Request Timeout

**Scenario**: Request takes too long to process

**Implementation**:

- Timeout configuration in app config
- Custom error handler for 408 status
- Client-side timeout in fetch requests

**Response**:

```json
{
  "success": false,
  "error": "Request timeout. The operation took too long to complete."
}
```

**Status Code**: 408 Request Timeout

---

### 7. Invalid Endpoint (404)

**Scenario**: User accesses non-existent endpoint

**Implementation**:

- Flask error handler for 404

**Response**:

```json
{
  "error": "Endpoint not found"
}
```

**Status Code**: 404 Not Found

---

### 8. Internal Server Error (500)

**Scenario**: Unexpected server error

**Implementation**:

- Generic error handler for unhandled exceptions
- Logs error for debugging

**Response**:

```json
{
  "error": "Internal server error"
}
```

**Status Code**: 500 Internal Server Error

---

## Frontend Error Handling

### 1. File Type Validation

**Implementation**:

- Client-side validation before upload
- Checks against allowed MIME types: JPEG, PNG, GIF, BMP, WebP

**User Feedback**:

- Red alert banner with error message
- File input reset automatically

---

### 2. File Size Validation

**Implementation**:

- Check file size before upload (16 MB max)
- Prevents unnecessary network requests

**User Feedback**:

```
File size exceeds 16 MB. Please select a smaller image.
```

---

### 3. File Read Error

**Implementation**:

- FileReader error handler
- Detects corrupted files during preview generation

**User Feedback**:

```
Failed to read file. The file may be corrupted.
```

---

### 4. Network Errors

**Implementation**:

- RTK Query error handling
- Detects network failures, timeout, server errors
- Automatic retry logic (up to 2 retries)

**User Feedback**:

```
Network error. Please check your connection and try again.
```

**Features**:

- Retry button in error alert
- Retry counter display
- Automatic retry with exponential backoff

---

### 5. Timeout Handling

**Implementation**:

- 30-second timeout in RTK Query baseQuery
- Specific error message for timeout

**User Feedback**:

```
Request timeout. The server took too long to respond. Please try again.
```

---

### 6. Server Errors (4xx, 5xx)

**Implementation**:

- Parse backend error messages
- Display user-friendly messages based on status code

**Error Messages**:

- **400**: Extract backend error message or show generic validation error
- **408**: Timeout message
- **413**: File too large message
- **500**: Server error message

---

### 7. Loading States

**Implementation**:

- RTK Query `isLoading` state
- Disabled buttons during processing
- Loading spinner with "Processing..." text

**User Feedback**:

- Button shows spinner icon
- Text changes to "Processing..." or "Retrying..."
- Prevents double submissions

---

### 8. React Error Boundary

**Implementation**:

- Class component that catches JavaScript errors
- Wraps entire application
- Logs errors to console

**Features**:

- Fallback UI when error occurs
- "Try Again" button to reset error state
- "Refresh Page" button for full reload
- Development mode shows error stack trace

**User Feedback**:

```
Something went wrong
The application encountered an unexpected error. Please try refreshing the page.
```

---

## Testing Error Handling

### Backend Tests

Run the test script to verify all backend error scenarios:

```bash
cd demo/backend
python test_error_handling.py
```

**Test Coverage**:

- ✓ Health check endpoint
- ✓ Upload without file
- ✓ Corrupted image upload
- ✓ Invalid endpoint (404)
- ✓ Valid image processing
- ✓ Large file upload (optional)
- ✓ Request timeout (optional)

---

### Frontend Tests

**Manual Testing Checklist**:

1. **File Type Validation**:

   - [ ] Upload .txt file → Should show error
   - [ ] Upload .pdf file → Should show error
   - [ ] Upload .jpg file → Should work
   - [ ] Upload .png file → Should work

2. **Network Error**:

   - [ ] Stop backend server
   - [ ] Try uploading → Should show network error
   - [ ] Retry button should appear
   - [ ] Click retry → Should attempt again

3. **Corrupted File**:

   - [ ] Upload corrupted image → Should show error
   - [ ] Error message should be clear

4. **Loading States**:

   - [ ] Upload button disabled while processing
   - [ ] Spinner shows during upload
   - [ ] Results display after completion

5. **Error Boundary**:
   - [ ] Simulate JavaScript error
   - [ ] Fallback UI should display
   - [ ] Try Again button should reset state

---

## Error Logging

### Backend Logging

All errors are logged to console with:

- Error type
- Error message
- Stack trace (for debugging)

Example:

```python
print(f"Error processing image: {str(e)}")
```

### Frontend Logging

Errors are logged to browser console:

```javascript
console.error('Upload failed:', err);
```

In production, consider adding:

- Error reporting service (e.g., Sentry)
- User analytics for error tracking

---

## Best Practices Implemented

1. **Fail Fast**: Validate input early (file type, size) before processing
2. **Specific Error Messages**: Clear, actionable error messages for users
3. **Graceful Degradation**: Application doesn't crash on errors
4. **Retry Logic**: Automatic and manual retry for transient failures
5. **Loading States**: Clear visual feedback during processing
6. **Error Boundaries**: Catch React errors to prevent white screen
7. **Type Safety**: TypeScript ensures proper error handling
8. **Separation of Concerns**: Backend validates, frontend displays

---

## Future Improvements

1. **Error Analytics**: Track error frequency and types
2. **Rate Limiting**: Prevent abuse with too many requests
3. **Image Format Conversion**: Auto-convert unsupported formats
4. **Progress Indicators**: Show upload/processing progress
5. **Offline Support**: Queue uploads when offline
6. **Error Recovery**: Automatic recovery for certain error types
7. **Better Logging**: Structured logging with log levels
8. **Health Monitoring**: Periodic health checks with alerts

---

## Summary

✅ **Backend**:

- Invalid image format ✓
- Corrupted image data ✓
- Model inference failure ✓
- File too large ✓
- Request timeout ✓
- 404/500 error handlers ✓

✅ **Frontend**:

- Network errors with retry logic ✓
- Timeout handling ✓
- Loading states ✓
- Error boundaries ✓
- File validation ✓
- User-friendly error messages ✓

**Task 8.2 Complete** ✓
