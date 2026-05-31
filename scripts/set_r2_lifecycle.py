"""Set the R2/S3 lifecycle rule that expires orphaned async-transcription objects.

The worker deletes the audio object on a *successful* transcription; this rule
sweeps the ones left behind by terminal failures (BullMQ/arq retries exhausted)
so the `transcribe-jobs/` prefix doesn't accumulate. One-time + idempotent —
re-running overwrites the rule. 1 day is well clear of any retry window.

Usage:
    # S3_* must be in the environment (or Media-Service/.env)
    python scripts/set_r2_lifecycle.py
"""
import os
import sys

import boto3
from botocore.config import Config

# Allow running as `python scripts/set_r2_lifecycle.py` (put the repo root on path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Settings  # noqa: E402

PREFIX = "transcribe-jobs/"
EXPIRE_DAYS = 1
RULE_ID = "expire-transcribe-jobs"


def main() -> None:
    s = Settings()
    if not s.s3_configured:
        print(
            "S3 not configured — set S3_ENDPOINT_URL / S3_BUCKET / "
            "S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY."
        )
        sys.exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=s.S3_ENDPOINT_URL,
        aws_access_key_id=s.S3_ACCESS_KEY_ID,
        aws_secret_access_key=s.S3_SECRET_ACCESS_KEY,
        region_name=s.S3_REGION or "auto",
        config=Config(signature_version="s3v4"),
    )

    from botocore.exceptions import ClientError

    try:
        client.put_bucket_lifecycle_configuration(
            Bucket=s.S3_BUCKET,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": RULE_ID,
                        "Filter": {"Prefix": PREFIX},
                        "Status": "Enabled",
                        "Expiration": {"Days": EXPIRE_DAYS},
                    }
                ]
            },
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "Forbidden"):
            print(
                "AccessDenied: this token can't set bucket lifecycle (it's "
                "object-scoped). Set the rule in the Cloudflare dashboard instead:\n"
                f"  R2 → bucket '{s.S3_BUCKET}' → Settings → Object lifecycle rules →\n"
                f"  Add rule: prefix '{PREFIX}', delete objects {EXPIRE_DAYS} day after creation.\n"
                "Or re-run this script with a bucket-admin R2 token."
            )
            sys.exit(2)
        raise

    print(f"Lifecycle set on {s.S3_BUCKET}: expire '{PREFIX}' after {EXPIRE_DAYS} day(s).")
    resp = client.get_bucket_lifecycle_configuration(Bucket=s.S3_BUCKET)
    print("Current rules:", resp.get("Rules"))


if __name__ == "__main__":
    main()
