from src.common.config import settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.common.repo import get_customer_bundle
import os

DB = os.environ.get("BLS_DUCKDB", str(settings.project.db_path))

app = FastAPI(title="Lead Scoring API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/api/customers/{cust_id}/profile")
def customer_profile(cust_id: int):
    bundle = get_customer_bundle(DB, cust_id)
    if not bundle["profile"]:
        raise HTTPException(status_code=404, detail="Customer not found")
    return bundle