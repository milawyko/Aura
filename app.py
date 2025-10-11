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
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
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

@app.get("/questionnaire")
async def get_questionnaire():
    """Возвращает анкету для фронтенда"""
    return analyzer.get_web_questionnaire()

@app.post("/analyze")
async def analyze_photo(
    file: UploadFile = File(...),
    user_id: str = "default"
):
    """Основной эндпоинт для анализа фото"""
    try:
        # Проверяем тип файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Читаем байты изображения
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Анализируем изображение
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

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    """Возвращает историю анализов пользователя"""
    try:
        history_data = analyzer.get_trends_data(user_id)
        return history_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/scan/{scan_id}/radar")
async def get_radar_data(scan_id: str):
    """Возвращает данные для радар-диаграммы"""
    try:
        radar_data = analyzer.get_radar_chart_data(scan_id)
        if "error" in radar_data:
            raise HTTPException(status_code=404, detail=radar_data["error"])
        return radar_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get radar data: {str(e)}")

@app.get("/celebrities")
async def get_celebrities():
    """Возвращает список доступных знаменитостей"""
    celebrities = [
        {
            "name": celeb_data["name"],
            "description": f"Сравнение с {celeb_data['name']}"
        }
        for celeb_data in analyzer.celebrity_database.values()
    ]
    return {"celebrities": celebrities}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Только для разработки
    )