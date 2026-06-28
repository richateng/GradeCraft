import os
import asyncio
import time
from typing import Dict, Any
from groq import AsyncGroq, GroqError, RateLimitError
from src.core.logger import get_logger

logger = get_logger(__name__)

# Strict Resource Allocation Matrix derived from rate limit specifications
TIER_PROFILES = {
    "vision_transcriber": {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tpm_ceiling": 30000,
        "rpm_ceiling": 30,
        "base_token_estimate": 3500
    },
    "math_verifier": {
        "model": "groq/compound-mini",
        "tpm_ceiling": 70000,
        "rpm_ceiling": 30,
        "base_token_estimate": 2500
    },
    "rubric_evaluator": {
        "model": "llama-3.3-70b-versatile",
        "tpm_ceiling": 12000,
        "rpm_ceiling": 30,
        "base_token_estimate": 1500
    }
}

class LeakyBucketThrottler:
    """Manages transactional state pacing to enforce multi-tiered API constraints."""
    def __init__(self, rpm: int, tpm: int):
        self.rpm_gate = asyncio.Semaphore(rpm)
        self.tpm_limit = tpm
        self.available_tokens = tpm
        self.checkpoint_time = time.time()

    async def secure_quota(self, expected_tokens: int):
        while True:
            current_time = time.time()
            elapsed = current_time - self.checkpoint_time
            # Replenish capacity linearly based on time elapsed
            self.available_tokens = min(self.tpm_limit, self.available_tokens + elapsed * (self.tpm_limit / 60.0))
            self.checkpoint_time = current_time

            if self.available_tokens >= expected_tokens:
                self.available_tokens -= expected_tokens
                break
            await asyncio.sleep(0.5)

# Initialize global rate limit counters
throttlers = {
    "vision": LeakyBucketThrottler(rpm=TIER_PROFILES["vision_transcriber"]["rpm_ceiling"], tpm=TIER_PROFILES["vision_transcriber"]["tpm_ceiling"]),
    "math": LeakyBucketThrottler(rpm=TIER_PROFILES["math_verifier"]["rpm_ceiling"], tpm=TIER_PROFILES["math_verifier"]["tpm_ceiling"]),
    "eval": LeakyBucketThrottler(rpm=TIER_PROFILES["rubric_evaluator"]["rpm_ceiling"], tpm=TIER_PROFILES["rubric_evaluator"]["tpm_ceiling"])
}

async def execute_throttled_inference(client: AsyncGroq, tier_key: str, payload_args: Dict[str, Any]) -> str:
    """Executes resilient inference routines wrapped in proactive retry loops."""
    if tier_key == "vision_transcriber":
        throttler_key = "vision"
    elif tier_key == "math_verifier":
        throttler_key = "math"
    else:
        throttler_key = "eval"
        
    profile = TIER_PROFILES[tier_key]
    throttler = throttlers[throttler_key]
    
    await throttler.secure_quota(profile["base_token_estimate"])
    async with throttler.rpm_gate:
        retry_delay = 2.0
        for execution_attempt in range(5):
            try:
                inference_job = await client.chat.completions.create(
                    model=profile["model"],
                    temperature=0.0,  # Enforce high determinism
                    **payload_args
                )
                return inference_job.choices[0].message.content
            except RateLimitError as api_anomaly:
                logger.warning(f"[ALERT] Rate Limit triggered for model {profile['model']}. Exponential cooling step: {retry_delay}s.")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            except GroqError as api_anomaly:
                raise api_anomaly
        raise RuntimeError(f"Pipeline failed to recover within allocated retry limits for {tier_key}.")
