"""Read status metadata for one Langfuse trace without printing prompt or SQL I/O."""
import argparse
from datetime import datetime, timedelta, timezone

from app.core.tracing import get_langfuse_client, init_tracing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id")
    trace_id = parser.parse_args().trace_id
    init_tracing()
    response = get_langfuse_client().api.observations.get_many(
        trace_id=trace_id,
        from_start_time=datetime.now(timezone.utc) - timedelta(hours=1),
        to_start_time=datetime.now(timezone.utc) + timedelta(minutes=1),
        fields="core,basic",
        limit=100,
    )
    for observation in response.data or []:
        status = getattr(observation, "status_message", None)
        if status:
            print(f"{observation.name}: {str(status)[:500]}")


if __name__ == "__main__":
    main()
