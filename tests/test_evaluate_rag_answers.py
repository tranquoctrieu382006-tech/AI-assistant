import httpx
import pytest

from scripts.evaluate_rag_answers import _post_with_rate_limit_retry


def test_answer_evaluator_retries_429_and_honors_retry_after():
    responses = iter([
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"answer": "ok"}),
    ])
    sleeps = []

    def handler(request):
        return next(responses)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = _post_with_rate_limit_retry(
            client,
            "http://test/rag",
            json_body={"question": "test"},
            headers={},
            sleep=sleeps.append,
        )

    assert response.status_code == 200
    assert len(sleeps) == 1
    assert 0 <= sleeps[0] <= 0.25


def test_answer_evaluator_stops_after_bounded_rate_limit_retries():
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(429))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            _post_with_rate_limit_retry(
                client,
                "http://test/rag",
                json_body={},
                headers={},
                max_retries=1,
                sleep=lambda _: None,
            )
