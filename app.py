from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from aura_analyzer import AuraHealthAnalyzer
import json
import os

app = FastAPI(title="Aura Health API", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация анализатора
analyzer = AuraHealthAnalyzer(use_huggingface=True)

@app.get("/")
async def root():
    return {"message": "Aura Health Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "celebrity_count": len(analyzer.celebrity_database)}

@app.post("/analyze/single")
async def analyze_single_photo(
    file: UploadFile = File(...),
    user_id: str = "default"
):
    """Анализ одного фото"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        result = analyzer.analyze_from_bytes(
            image_bytes=image_bytes,
            user_id=user_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/3d")
async def analyze_3d_scan(
    front: UploadFile = File(...),
    left: UploadFile = File(...), 
    right: UploadFile = File(...),
    user_id: str = "default"
):
    """Анализ 3 ракурсов (фронт, лево, право)"""
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
        result = analyzer.analyze_3d_scan(
            front_bytes=front_bytes,
            left_bytes=left_bytes, 
            right_bytes=right_bytes,
            user_id=user_id
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