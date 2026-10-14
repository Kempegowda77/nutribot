"""
NutriChat AI — Entry Point
Run with: python main.py
Or:        uvicorn api.app:app --reload --port 8000
first commit
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
