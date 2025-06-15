import os
import io
from PIL import Image
from pathlib import Path
from huggingface_hub import InferenceClient
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.responses import JSONResponse
from secondsight.model import ModelFactory, SceneModel
    
router = APIRouter()

def get_enigmaai_model(request: Request) -> SceneModel:
    """Dependency to get the model from application state."""
    return request.app.state.enigmaai

def get_llava_model(request: Request) -> SceneModel:
    """Dependency to get the model from application state."""
    return request.app.state.llava
    
@router.post("/api/scene/describe")
async def infer(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: SceneModel = Depends(get_enigmaai_model)
):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate file size (e.g., max 10MB)
        image_bytes = await file.read()
        # if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
        #     raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Validate prompt
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Process the image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            save_path = "/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/project/SecondSight-API/api/static/scene"
            file_location = os.path.join(save_path, file.filename)
            with open(file_location, "wb") as buffer:
                buffer.write(image_bytes)
        
            response = model.predict(image, prompt)
            return {"detail": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @router.post("/api/aya/scene/describe")
# async def ayaInfer(
#     file: UploadFile = File(...),
#     prompt: str = Form(...),
#     model: SceneModel = Depends(get_llava_model)):
#     return infer(file, prompt, model)

@router.post("/api/llava/scene/describe")
async def llavaInfer(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: SceneModel = Depends(get_llava_model)):
    
    restuls = await infer(file, prompt, model)
    return restuls
