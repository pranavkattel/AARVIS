import os
import time

import httpx
import pytest


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TEST_TIMEOUT", "20"))


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    with httpx.Client(timeout=TIMEOUT, follow_redirects=False) as client:
        return client.request(method, f"{BASE_URL}{path}", **kwargs)


@pytest.fixture(scope="session", autouse=True)
def ensure_server_up() -> None:
    last_error = None
    for _ in range(20):
        try:
            resp = _request("GET", "/login")
            if resp.status_code in (200, 302):
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)

    pytest.fail(
        "API server is not reachable at "
        f"{BASE_URL}. Start it first with: D:/langgraph/venv/Scripts/python.exe \"d:/langgraph/final fixed fyp/run.py\". "
        f"Last error: {last_error}"
    )


@pytest.mark.parametrize(
    "method,path,expected_status",
    [
        ("GET", "/api/admin/users", 200),
        ("GET", "/api/admin/face-list", 200),
        ("POST", "/api/register", 410),
        ("POST", "/api/login", 410),
        ("POST", "/api/logout", 200),
        ("GET", "/api/pair-status/not-a-real-token", 200),
        ("GET", "/api/local-url", 200),
        ("POST", "/api/pair-trigger/not-a-real-token", 404),
        ("GET", "/api/user", 401),
        ("POST", "/api/user/context", 401),
        ("GET", "/api/voice/readiness", 401),
        ("GET", "/api/calendar/upcoming", 401),
        ("GET", "/api/calendar/range?start=2026-04-19T00:00:00%2B05:45&end=2026-04-20T00:00:00%2B05:45", 401),
        ("GET", "/api/calendar/events/not-a-real-event", 401),
        ("POST", "/api/calendar/events", 401),
        ("PUT", "/api/calendar/events/not-a-real-event", 401),
        ("DELETE", "/api/calendar/events/not-a-real-event", 401),
        ("GET", "/api/mail/inbox", 401),
        ("GET", "/api/mail/message/not-a-real-message", 401),
        ("POST", "/api/mail/send", 401),
        ("POST", "/api/briefing/trigger", 401),
        ("POST", "/api/face/enroll", 200),
    ],
)
def test_api_status_codes(method: str, path: str, expected_status: int) -> None:
    payload = None
    if method == "POST" and path == "/api/user/context":
        payload = {"location": "Kathmandu", "interests": "technology"}
    elif method == "POST" and path == "/api/calendar/events":
        payload = {
            "summary": "Smoke Test Event",
            "start_time": "2026-04-19T10:00:00+05:45",
            "end_time": "2026-04-19T11:00:00+05:45",
            "description": "Calendar smoke test",
            "location": "Kathmandu",
        }
    elif method == "PUT" and path == "/api/calendar/events/not-a-real-event":
        payload = {"summary": "Updated Event"}
    elif method == "POST" and path == "/api/mail/send":
        payload = {"to": "person@example.com", "subject": "Test", "body": "Hello"}
    elif method == "POST" and path == "/api/face/enroll":
        payload = {"username": "unit-user", "images": []}

    response = _request(method, path, json=payload)
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "path,payload,expected_key",
    [
        ("/api/face/verify", {}, "detected"),
        ("/api/face/process", {}, "error"),
        ("/api/face/login", {}, "success"),
    ],
)
def test_face_endpoints_bad_input_shape(path: str, payload: dict, expected_key: str) -> None:
    response = _request("POST", path, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert expected_key in data


def test_news_endpoint_shape() -> None:
    response = _request("GET", "/api/news")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert isinstance(data[0], dict)
    assert "title" in data[0]


def test_weather_endpoint_shape() -> None:
    response = _request("GET", "/api/weather")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    for key in ("temp", "condition", "location", "temp_min", "temp_max"):
        assert key in data


def test_calendar_status_or_shape() -> None:
    response = _request("GET", "/api/calendar")

    # Depends on CALENDAR_AVAILABLE and auth state.
    assert response.status_code in (200, 401)

    data = response.json()
    assert isinstance(data, dict)
    if response.status_code == 200:
        assert "events" in data
    else:
        assert data.get("reauth_required") is True


def test_admin_update_unknown_user() -> None:
    response = _request(
        "PUT",
        "/api/admin/users/9999999",
        json={
            "full_name": "Test User",
            "email": "test@example.com",
            "location": "Kathmandu",
            "interests": "technology",
        },
    )
    assert response.status_code == 404


def test_admin_delete_unknown_user() -> None:
    response = _request("DELETE", "/api/admin/users/9999999")
    assert response.status_code == 404
