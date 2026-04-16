from fastapi import APIRouter, HTTPException

from api import JOBS, _jobs_lock

router = APIRouter()


@router.get("/api/jobs/{job_id}")
@router.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    with _jobs_lock:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"job_id": job_id, **job}
