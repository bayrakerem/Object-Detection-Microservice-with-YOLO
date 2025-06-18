from fastapi import FastAPI

app = FastAPI(
    title="Object Detection API",
    description="A microservice for object detection using YOLO",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 