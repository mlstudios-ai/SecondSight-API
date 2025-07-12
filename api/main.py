from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from api.routers import api, index
from secondsight.model import ModelFactory
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        app.state.enigmaai  = ModelFactory.get_model(
            ModelFactory.ModelType.SCENE,
            ModelFactory.ModelName.ENIGMAAI)
        print(f"EnigmaAI model initialised")
        
        app.state.llava  = ModelFactory.get_model(
            ModelFactory.ModelType.SCENE,
            ModelFactory.ModelName.LLAVA)
        print(f"LLaVA model initialised")
        
        app.state.aya  = ModelFactory.get_model(
            ModelFactory.ModelType.SCENE,
            ModelFactory.ModelName.AYA)
        print(f"AYA model initialised")
        
    except Exception as e:
        print(f"Failed to initialise model: {str(e)}")
        raise
    yield
    # Shutdown
    app.state.enigmaai.finalize()
    app.state.llava.finalize()
    app.state.aya.finalize()

app = FastAPI(lifespan=lifespan)

# Allow CORS for frontend (if you plan to connect to this via a frontend app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# no cache response to prevent stale content
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Add the middleware to your FastAPI application
app.add_middleware(NoCacheMiddleware)

app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

# routers to other webpages
app.include_router(api.router, tags=["api"])
app.include_router(index.router)
