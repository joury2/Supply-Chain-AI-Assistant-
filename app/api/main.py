# app/api/main.py
"""
Production FastAPI Backend for AI Forecasting System
FIXED: Import paths and module structure
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd
import io
import os
import sys
import uuid
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
import jwt
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
EXPECTED_API_TOKEN = os.getenv("FASTAPI_API_TOKEN", "forecasting-api-token-v1")

# Initialize FastAPI
app = FastAPI(
    title="AI Forecasting API",
    description="Production-ready forecasting API with model selection, validation, and interpretation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# In-memory storage
sessions = {}  # session_id -> {agent, dataset, created_at, user_id}
forecast_jobs = {}  # job_id -> {status, result, created_at}
rate_limit_store = {}  # user_id -> {count, reset_time}

# ============================================================================
# Pydantic Models
# ============================================================================

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    dataset_rows: int
    dataset_columns: int
    status: str

class DatasetUploadResponse(BaseModel):
    session_id: str
    rows: int
    columns: List[str]
    frequency: str
    message: str

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    session_id: str
    has_forecast_data: bool = False
    forecast_data: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    session_id: str
    horizon: int = Field(..., ge=1, le=365)
    business_context: Optional[Dict[str, Any]] = None

class ForecastResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AnalysisRequest(BaseModel):
    session_id: str

class AnalysisResponse(BaseModel):
    status: str
    validation: Dict[str, Any]
    compatible_models: List[Dict[str, Any]]
    selected_model: Dict[str, Any]
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# ============================================================================
# Authentication
# ============================================================================

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    
    if token != EXPECTED_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )
    
    return token[:10]  # Return user_id

def create_jwt_token(user_id: str):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def check_rate_limit(user_id: str, limit: int = 100):
    """Simple rate limiting"""
    now = datetime.now()
    
    if user_id not in rate_limit_store:
        rate_limit_store[user_id] = {
            'count': 1,
            'reset_time': now + timedelta(hours=1)
        }
        return True
    
    user_data = rate_limit_store[user_id]
    
    if now > user_data['reset_time']:
        rate_limit_store[user_id] = {
            'count': 1,
            'reset_time': now + timedelta(hours=1)
        }
        return True
    
    if user_data['count'] >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    user_data['count'] += 1
    return True

# ============================================================================
# Helper Functions
# ============================================================================

def get_session(session_id: str):
    """Get session or raise 404"""
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return sessions[session_id]

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "llm": "available"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    services = {}
    
    # Check database
    try:
        import sqlite3
        conn = sqlite3.connect('supply_chain.db')
        conn.execute("SELECT 1")
        conn.close()
        services['database'] = "healthy"
    except:
        services['database'] = "unhealthy"
    
    # Check LLM API key
    services['llm'] = "healthy" if os.getenv("NVIDIA_API_KEY") else "no_api_key"
    
    # Check vector store
    services['vector_store'] = "healthy" if os.path.exists('vector_store/index.faiss') else "missing"
    
    overall = "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    
    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "services": services
    }

@app.get("/api/v1/sessions", response_model=List[SessionInfo])
async def list_sessions(user_id: str = Depends(verify_token)):
    """List all active sessions"""
    session_list = []
    
    for session_id, session_data in sessions.items():
        if session_data['user_id'] == user_id:
            dataset = session_data['dataset']
            session_list.append(SessionInfo(
                session_id=session_id,
                created_at=session_data['created_at'].isoformat(),
                dataset_rows=len(dataset),
                dataset_columns=len(dataset.columns),
                status="active"
            ))
    
    return session_list

@app.post("/api/v1/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token)
):
    """Upload dataset"""
    check_rate_limit(user_id)
    
    try:
        # Read file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format"
            )
        
        # Parse dates
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # Initialize agent
        from app.services.llm.simplified_forecast_agent import SimplifiedForecastAgent
        
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="NVIDIA_API_KEY not configured"
            )
        
        agent = SimplifiedForecastAgent(nvidia_api_key=nvidia_api_key)
        agent.upload_dataset(df)
        
        # Create session
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            'agent': agent,
            'dataset': df,
            'created_at': datetime.now(),
            'user_id': user_id
        }
        
        frequency = agent._detect_frequency()
        
        logger.info(f"✅ Dataset uploaded: {len(df)} rows, session: {session_id}")
        
        return DatasetUploadResponse(
            session_id=session_id,
            rows=len(df),
            columns=list(df.columns),
            frequency=frequency,
            message="Dataset uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty")
    except pd.errors.ParserError:
        raise HTTPException(400, "Cannot parse file")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "Upload failed")
    
    
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(verify_token)
):
    """Chat with AI"""
    check_rate_limit(user_id, limit=200)
    
    try:
        session = get_session(request.session_id)
        agent = session['agent']
        
        response = agent.ask_question(request.message)
        
        intent = "unknown"
        if agent.conversation_history:
            intent = agent.conversation_history[-1].get('intent', 'unknown')
        
        forecast_data = agent.get_current_forecast_data()
        
        return ChatResponse(
            response=response,
            intent=intent,
            session_id=request.session_id,
            has_forecast_data=forecast_data is not None,
            forecast_data=forecast_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_dataset(
    request: AnalysisRequest,
    user_id: str = Depends(verify_token)
):
    """Analyze dataset"""
    check_rate_limit(user_id)
    
    try:
        session = get_session(request.session_id)
        agent = session['agent']
        
        response = agent._analyze_dataset()
        result = agent.current_result
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis failed"
            )
        
        validation = result.get('rule_analysis', {}).get('validation', {})
        compatible_models = result.get('rule_analysis', {}).get('selection_analysis', {}).get('analysis', {}).get('compatible_models', [])
        selected_model = result.get('rule_analysis', {}).get('model_selection', {})
        recommendations = result.get('knowledge_base_recommendations', [])
        
        return AnalysisResponse(
            status="success",
            validation=validation,
            compatible_models=compatible_models,
            selected_model=selected_model,
            recommendations=recommendations[:5]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/forecast", response_model=ForecastResponse)
async def forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(verify_token)
):
    """Create forecast job"""
    check_rate_limit(user_id, limit=50)
    
    try:
        session = get_session(request.session_id)
        
        job_id = str(uuid.uuid4())
        forecast_jobs[job_id] = {
            'status': 'pending',
            'progress': 0,
            'result': None,
            'error': None,
            'created_at': datetime.now()
        }
        
        background_tasks.add_task(
            run_forecast_job,
            job_id,
            request.session_id,
            request.horizon
        )
        
        return ForecastResponse(
            job_id=job_id,
            status="pending",
            message=f"Forecast job created: {job_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Forecast creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/forecast/status/{job_id}", response_model=JobStatusResponse)
async def get_forecast_status(
    job_id: str,
    user_id: str = Depends(verify_token)
):
    """Get forecast status"""
    if job_id not in forecast_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job = forecast_jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress'),
        result=job.get('result'),
        error=job.get('error')
    )

@app.delete("/api/v1/session/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(verify_token)
):
    """Delete session"""
    try:
        session = get_session(session_id)
        
        try:
            session['agent'].close()
        except:
            pass
        
        del sessions[session_id]
        
        logger.info(f"✅ Session deleted: {session_id}")
        return {"message": "Session deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ============================================================================
# Background Tasks
# ============================================================================

async def run_forecast_job(job_id: str, session_id: str, horizon: int):
    """Run forecast in background"""
    try:
        forecast_jobs[job_id]['status'] = 'processing'
        forecast_jobs[job_id]['progress'] = 10
        
        session = sessions[session_id]
        agent = session['agent']
        
        forecast_jobs[job_id]['progress'] = 30
        
        response = agent._run_forecast(horizon)
        
        forecast_jobs[job_id]['progress'] = 80
        
        forecast_data = agent.get_current_forecast_data()
        
        forecast_jobs[job_id]['status'] = 'completed'
        forecast_jobs[job_id]['progress'] = 100
        forecast_jobs[job_id]['result'] = {
            'response': response,
            'forecast_data': forecast_data,
            'interpretation': agent.current_result.get('interpretation') if agent.current_result else None
        }
        
        logger.info(f"✅ Forecast job completed: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Forecast job failed: {job_id} - {e}")
        forecast_jobs[job_id]['status'] = 'failed'
        forecast_jobs[job_id]['error'] = str(e)

# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting AI Forecasting API...")
    
    if not os.getenv("NVIDIA_API_KEY"):
        logger.warning("⚠️ NVIDIA_API_KEY not set!")
    
    if not os.path.exists('supply_chain.db'):
        logger.warning("⚠️ Database not found!")
    
    logger.info("✅ API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down...")
    
    for session_id, session in sessions.items():
        try:
            session['agent'].close()
        except:
            pass
    
    logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)