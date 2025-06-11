# AI Model Auto-Selection Feature

## Overview

We've added a new feature that automatically checks which AI models are available when the application starts and selects the best one to use. This ensures the application always uses an available and optimal model without requiring manual configuration.

## Features Added

### 1. Automatic Model Selection on Startup

- The application now checks which AI models are working when it starts
- Models are tried in order of preference (best quality first)
- The first working model is automatically selected and configured
- If no models are working, the default configuration is used

### 2. Model Manager Service

- Created a new `ModelManager` class to handle model availability checks
- Implemented efficient model testing that minimizes API calls
- Defined a priority order of preferred models based on quality
- Added logging to track model selection

### 3. Application Initialization

- Created a startup utility to run initialization tasks
- Integrated model selection into application startup
- Added session state tracking of the selected model

### 4. Settings UI Enhancements

- Added AI Settings tab in the Settings section
- Shows current model and initialization status
- Provides a button to manually check and select models
- Added model information and cache management

## How It Works

1. **On Application Startup:**
   - The `initialize_application()` function runs
   - `ModelManager` tests each preferred model in sequence
   - The first working model is selected and set in the configuration
   - The selected model is displayed in the settings tab

2. **Manual Check:**
   - Users can click "Check Available AI Models" in Settings
   - The system will test all models again
   - The best available model is selected and displayed
   - AI cache is cleared to ensure fresh results

3. **Preferred Models (in order):**
   - gpt-4o-mini (highest quality)
   - gpt-3.5-turbo-16k (large context window)
   - gpt-3.5-turbo-0125
   - gpt-3.5-turbo-1106
   - gpt-3.5-turbo

## Testing Results

Testing confirmed that:
- The system correctly identifies working models
- It selects the best available model based on priority
- The settings UI correctly displays model information
- The manual check feature works as expected

## Files Modified

1. `services/model_manager.py` - New file for managing model selection
2. `utils/startup.py` - New file for application initialization
3. `main.py` - Updated to use startup utilities and display model info
4. `test_model_selection.py` - Test script for the feature

## Future Enhancements

1. Add ability to manually set a preferred model
2. Implement periodic checks for model availability
3. Add notifications when models become unavailable
4. Support for new models as they become available 