from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from aura_analyzer import analyze_3_images
import json

app = FastAPI(title="Aura Health API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Aura Health Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analysis")
async def create_analysis(
    front: UploadFile = File(...),
    left: UploadFile = File(...), 
    right: UploadFile = File(...),
    user_id: str = "default"
):
    """Основной эндпоинт для анализа 3 изображений"""
    try:
        # Проверяем все файлы
        for file, angle in [(front, "front"), (left, "left"), (right, "right")]:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"{angle} file must be an image")
        
        # Читаем байты всех изображений
        front_bytes = await front.read()
        left_bytes = await left.read()
        right_bytes = await right.read()
        
        # Проверяем что файлы не пустые
        for bytes_data, angle in [(front_bytes, "front"), (left_bytes, "left"), (right_bytes, "right")]:
            if len(bytes_data) == 0:
                raise HTTPException(status_code=400, detail=f"{angle} file is empty")
        
        # Анализируем 3 изображения
        result = analyze_3_images(
            front_bytes=front_bytes,
            left_bytes=left_bytes, 
            right_bytes=right_bytes
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        reload=False
    )