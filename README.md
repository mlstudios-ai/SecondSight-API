# SecondSight API - Remote Model Inference Service

FastAPI-based remote inference service for the **SecondSight** assistive AI application. This component provides scene description capabilities using Vision Language Models (VLMs) for individuals with visual impairment.

> Scene Description model API is for lower spec iPhones and earlier version of SecondSight.
> New version now uses FastVLM that requires significantly more memory and storage.

## Overview

SecondSight-API is one of three core components in the SecondSight system:

3. **[SecondSight](https://github.com/mlstudios-ai/SecondSight)** - iOS client application in SwiftUIdeployment using ClearML
2. **SecondSight-API** - Remote model inference endpoint (this repository)
1. **[SecondSight-MLOps](https://github.com/mlstudios-ai/SecondSight-MLOps)** - Automated ML pipelines for model training and 

For more information, visit the [main project repository](https://github.com/mlstudios-ai/SecondSight).

## Architecture

This API component serves as the **Scene Description (Remote)** inference service in the SecondSight architecture. It hosts multiple Vision Language Models (VLMs) for generating contextual descriptions of scenes and identified hazards.

### Key Features

- **Multi-Model Support**: Three VLM models available for scene description
  - **EnigmaAI**: Custom fine-tuned Vision Encoder-Decoder model (ViT + GPT-2)
  - **LLaVA-1.5-7B**: Large vision-language model from Hugging Face
  - **Aya Vision 8B**: Multilingual vision-language model from Cohere
- **Application-Scoped Model Loading**: All three models loaded once at application startup
- **Asynchronous Inference**: FastAPI endpoints with async request handling
- **CORS Enabled**: Configured for cross-origin requests from iOS client
- **No-Cache Middleware**: Ensures fresh responses for real-time scene descriptions
- **REST API Communication**: Multi-part form data (image + text prompt)

## Functional Requirements

The Scene Description service provides:

- Processing of still images captured from the iOS device camera
- Text prompt integration for context-aware descriptions
- Detailed descriptions of identified hazards and their spatial context
- General scene descriptions when no hazards are detected
- Support for multiple VLM backends for comparison and flexibility
- Image validation and error handling

## Performance Requirements

- **Response Time**: Remote API inference responds within 500ms under normal network conditions
- **Model Performance**: Minimum 80% BLEU/CIDEr scores for generated descriptions
- **Generation Length**: Max 50 tokens (15-50 depending on model) for concise descriptions
- **Temperature**: 0.7 for balanced creativity and consistency
- **Device Optimization**: Automatic device detection (CUDA/MPS/CPU) via utility functions

## Technology Stack

- **Framework**: FastAPI with Uvicorn
- **Frontend**: Jinja2 templates + static files for web interface
- **Model Framework**: PyTorch with Transformers library
- **Model Architectures**:
  - Vision Encoder-Decoder (ViT + GPT-2) - Custom EnigmaAI model
  - LLaVA (LlavaForConditionalGeneration) - Hugging Face
  - Aya Vision (AutoModelForImageTextToText) - Cohere
- **Image Processing**: PIL (Pillow)
- **Model Optimization**: FP16 precision, low CPU memory usage
- **Dependency Management**: Conda environment with Python 3.11

## Environment Setup

1. Create a new conda environment:

    ```sh
    conda create -p venv/ python==3.11
    conda activate venv/
    ```

2. Install required dependencies:

    ```sh
    pip3 install -r api/requirements.txt 
    ```

## Running the Server

Start the FastAPI server:

```sh
uvicorn api.main:app --reload
```

## API Endpoints

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Root Landing Page**: `http://localhost:8000/`

### Scene Description Endpoints

All endpoints accept `multipart/form-data` POST requests with:
- **file**: Image file (validated to be image/* MIME type)
- **prompt**: Text prompt describing what to look for or context

#### 1. EnigmaAI Model (Default)
```http
POST /api/scene/describe
```
Uses the custom fine-tuned Vision Encoder-Decoder model.

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/scene/describe" \
  -F "file=@image.jpg" \
  -F "prompt=Describe the scene and any hazards"
```

#### 2. LLaVA Model
```http
POST /api/llava/scene/describe
```
Uses the LLaVA-1.5-7B model from Hugging Face.

#### 3. Aya Vision Model
```http
POST /api/aya/scene/describe
```
Uses the Aya Vision 8B multilingual model from Cohere.

### Response Format

All endpoints return a JSON response:
```json
{
  "detail": "A description of the scene including any hazards detected."
}
```

### Error Handling

- **400 Bad Request**: Invalid file type or empty prompt
- **500 Internal Server Error**: Model inference or image processing failure

## Model Architecture & Implementation

### Model Loading Strategy

All three models are loaded at **application startup** using FastAPI's lifespan context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all models into application state
    app.state.enigmaai = ModelFactory.get_model(...)
    app.state.llava = ModelFactory.get_model(...)
    app.state.aya = ModelFactory.get_model(...)
    yield
    # Shutdown: Clean up resources
    app.state.enigmaai.finalize()
    app.state.llava.finalize()
    app.state.aya.finalize()
```

This approach:
- Loads models once, not per request
- Stores models in application state for efficient access
- Properly cleans up GPU/MPS memory on shutdown
- Uses dependency injection to pass models to route handlers

### Model Implementations

#### 1. EnigmaAI Vision (Custom Model)
- **Architecture**: Vision Encoder-Decoder (ViT feature extractor + GPT-2 decoder)
- **Location**: `api/models/SceneModel/`
- **Max Tokens**: 15 tokens (concise descriptions)
- **Special Features**: Custom tokenizer with [PAD] token, temperature 0.7

#### 2. LLaVA Vision
- **Architecture**: LlavaForConditionalGeneration
- **Source**: `llava-hf/llava-1.5-7b-hf` from Hugging Face
- **Max Tokens**: 50 tokens
- **Precision**: FP16 with low CPU memory usage
- **Prompt Format**: `USER: <image>\n{prompt}\nASSISTANT:`

#### 3. Aya Vision
- **Architecture**: AutoModelForImageTextToText
- **Source**: `CohereLabs/aya-vision-8b` from Hugging Face
- **Max Tokens**: 50 tokens
- **Precision**: FP16 with device auto-mapping
- **Special Features**: Multilingual support

### Model Factory Pattern

The codebase uses a Factory pattern for model instantiation:

```python
ModelFactory.get_model(
    ModelFactory.ModelType.SCENE,
    ModelFactory.ModelName.ENIGMAAI
)
```

This provides:
- Type-safe model selection via Enums
- Centralized error handling
- Consistent initialization across model types
- Easy extension for new models

### Resource Management

Each model implements proper cleanup:
- **CPU Migration**: Models moved to CPU before deletion
- **Cache Clearing**: CUDA/MPS cache explicitly emptied
- **Garbage Collection**: Force GC after model deletion
- **Processor Cleanup**: Tokenizers and processors properly released

## System Integration

### Communication Flow

```
iOS App → Multipart Form Data (image + prompt) → FastAPI Endpoint 
  → Model Selection (enigmaai/llava/aya) → VLM Inference → JSON Response → iOS App
```

### Request Processing Pipeline

1. **Upload**: iOS client sends multipart form data with image file and text prompt
2. **Validation**: 
   - File MIME type checked for `image/*`
   - Prompt validated for non-empty content
   - File size validation (commented out but available)
3. **Image Processing**: 
   - Image bytes read and converted to PIL Image
   - Saved to `api/static/scene/` directory for reference
4. **Model Inference**:
   - Image + prompt formatted per model requirements
   - Model dependency injected from application state
   - Inference run with torch.no_grad() for efficiency
5. **Response Extraction**:
   - Full model output decoded
   - Assistant response extracted from formatted output
   - JSON response returned to client

### Integration Requirements

- **CORS**: Enabled for all origins (configure for production)
- **Network**: Internet connectivity required for API access
- **Image Format**: Any PIL-supported image format (JPEG, PNG, etc.)
- **Prompt Format**: Plain text, incorporated into model-specific templates
- **Response Caching**: Disabled via NoCacheMiddleware for real-time inference

## Security & Privacy

- **Minimal Data Storage**: Video/image data is not stored beyond processing duration
- **Minimal Permissions**: Only requires image data for inference
- **No User Data Collection**: Application does not store user data
- **iOS Security Compliance**: Adheres to iOS security frameworks and guidelines

## Code Structure

```
SecondSight-API/
├── api/
│   ├── main.py                    # FastAPI app initialization & lifespan management
│   ├── routers/
│   │   ├── api.py                 # Scene description endpoints
│   │   └── index.py               # Landing page route
│   ├── src/secondsight/
│   │   ├── model.py               # Model classes & factory
│   │   └── util.py                # Device detection utilities
│   ├── models/
│   │   └── SceneModel/            # Custom EnigmaAI model files
│   ├── static/                    # Static assets & saved images
│   └── templates/                 # Jinja2 HTML templates
├── requirements.txt               # Python dependencies
└── README.md
```

## Key Dependencies

```
fastapi[standard,docs,dev]         # Web framework
torch==2.5.0                       # Deep learning framework
transformers==4.51.3               # Hugging Face models
ultralytics==8.3.146               # YOLO (for future detection API)
pillow                             # Image processing
python-multipart                   # Form data handling
bitsandbytes                       # Quantization support
```

## Limitations

- **Small Dataset**: custom curated dataset approximately 2k samples
- **Startup Time**: Models load at cold start, causing initial delay (30-60 seconds)

## Future Considerations

From Level Design Document v0.2:

- **Microservices Architecture**: Separate model inference services for resource allocation
- **Hazard Detection API**: Add YOLO-based hazard detection endpoint for lower-spec devices
- **Scalability**: Implement model caching and load balancing for concurrent requests
- **Multi-device Support**: Android client support
- **Model Optimization**: Quantization (4-bit/8-bit) for reduced memory footprint
- **CI/CD Integration**: Automated model deployment from ClearML pipelines
- **Metrics & Monitoring**: Response time tracking, model performance logging

## Performance Metrics (Target)

- **Model Accuracy**: 80%+ BLEU/CIDEr scores
- **Response Time**: <500ms under normal conditions
- **Uptime Target**: 99.5%
- **Description Length**: 15-50 tokens for concise, accessible output

## Development Notes

- **Device Auto-Detection**: Utility function detects CUDA/MPS/CPU automatically
- **Prompt Engineering**: All models use USER/ASSISTANT format for consistency
- **Temperature**: 0.7 for balanced creativity and determinism
- **Token Limits**: Configured per model (15 for EnigmaAI, 50 for LLaVA/Aya)
- **Error Handling**: Custom exceptions (ModelError, ModelLoadError, PredictionError)