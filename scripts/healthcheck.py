"""Health check script for verifying service availability."""

import httpx
import redis
import psycopg2
from typing import Dict, List
import sys

# Import configuration
try:
    from app.config import REDIS_URL, POSTGRES_URL
except ImportError:
    # Fallback for standalone execution
    import os
    from dotenv import load_dotenv
    load_dotenv()
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/app")


def check_api_health(api_url: str = "http://localhost:8000") -> Dict:
    """Check API health endpoint."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        if response.status_code == 200:
            return {"status": "healthy", "details": response.json()}
        else:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_redis_health() -> Dict:
    """Check Redis connectivity."""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        return {"status": "healthy", "details": {"connected": True}}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_postgres_health() -> Dict:
    """Check PostgreSQL connectivity."""
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return {"status": "healthy", "details": {"connected": True}}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def run_healthcheck() -> bool:
    """Run all health checks and return overall status."""
    print("Running health checks...")
    print("-" * 50)
    
    checks = {
        "API": check_api_health(),
        "Redis": check_redis_health(),
        "PostgreSQL": check_postgres_health(),
    }
    
    all_healthy = True
    for service, result in checks.items():
        status = result["status"]
        symbol = "✓" if status == "healthy" else "✗"
        print(f"{symbol} {service}: {status}")
        if "error" in result:
            print(f"  Error: {result['error']}")
        if status != "healthy":
            all_healthy = False
    
    print("-" * 50)
    overall = "All systems operational" if all_healthy else "Some services are unhealthy"
    print(f"Overall: {overall}")
    
    return all_healthy


if __name__ == "__main__":
    success = run_healthcheck()
    sys.exit(0 if success else 1)
