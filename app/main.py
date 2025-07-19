# app/make_preprocessing.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.params import Depends

from app.auth import basic_auth
from app.routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .models import get_model
    get_model()      # startup 시 모델 적재
    yield            # shutdown 시 정리할 게 있으면 여기에
                     # (없으면 그냥 pass)
app = FastAPI(
    title="Pullback ML API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router,dependencies=[Depends(basic_auth)])



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # ↙️  디버깅 땐 False 권장(아래 설명)
    )
