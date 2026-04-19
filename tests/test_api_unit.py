from __future__ import annotations

from datetime import datetime
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from app import main
import app.services.gmail_service as gmail_service


def _auth_override() -> main.AuthContext:
    return main.AuthContext(token="unit-token", username="unit-user")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    main.app.dependency_overrides[main.require_auth_context] = _auth_override
    main.app.dependency_overrides[main.get_optional_auth_context] = _auth_override
    monkeypatch.setattr(
        main,
        "get_user_by_username",
        lambda username: {
            "username": username,
            "full_name": "Unit Test User",
            "location": "Kathmandu",
            "interests": "technology",
            "news_interests": "",
            "news_country": "",
            "google_id": "google-unit-id",
        },
    )

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()


def test_calendar_today_endpoint_returns_formatted_events(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_todays_events", lambda raise_on_auth_error=False: [
        {
            "id": "evt-today-1",
            "summary": "Project Standup",
            "description": "Daily sync",
            "location": "Mirror Lab",
            "start": {"dateTime": "2026-04-19T09:00:00+05:45"},
            "end": {"dateTime": "2026-04-19T09:30:00+05:45"},
        }
    ])
    monkeypatch.setattr(main, "set_calendar_current_user", lambda username: None)

    response = client.get("/api/calendar")
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert data["events"][0]["id"] == "evt-today-1"
    assert data["events"][0]["title"] == "Project Standup"


def test_get_authenticated_user_profile(client: TestClient) -> None:
    response = client.get("/api/user")
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "unit-user"
    assert data["session_token"] == "unit-token"


def test_calendar_upcoming_endpoint_returns_count(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_upcoming_events", lambda max_results=10, raise_on_auth_error=False: [
        {
            "id": "evt-upcoming-1",
            "summary": "Client Review",
            "start": {"dateTime": "2099-04-19T14:00:00+05:45"},
            "end": {"dateTime": "2099-04-19T15:00:00+05:45"},
        }
    ])

    response = client.get("/api/calendar/upcoming", params={"max_results": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["events"][0]["id"] == "evt-upcoming-1"


def test_calendar_range_endpoint_passes_datetime_range(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_events_in_range(start_time, end_time, max_results=50, raise_on_auth_error=False):
        captured["start"] = start_time
        captured["end"] = end_time
        captured["max_results"] = max_results
        return [{
            "id": "evt-range-1",
            "summary": "Range Event",
            "start": {"dateTime": "2026-04-19T11:00:00+05:45"},
            "end": {"dateTime": "2026-04-19T12:00:00+05:45"},
        }]

    monkeypatch.setattr(main, "get_events_in_range", fake_get_events_in_range)

    response = client.get(
        "/api/calendar/range",
        params={
            "start": "2026-04-19T00:00:00+05:45",
            "end": "2026-04-20T00:00:00+05:45",
            "max_results": 12,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["events"][0]["title"] == "Range Event"
    assert captured["max_results"] == 12
    assert isinstance(captured["start"], datetime)
    assert isinstance(captured["end"], datetime)


def test_calendar_create_event_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_calendar_event(summary, start_time, end_time, description="", location="", timezone_name="Asia/Kathmandu"):
        captured.update(
            summary=summary,
            start_time=start_time,
            end_time=end_time,
            description=description,
            location=location,
            timezone_name=timezone_name,
        )
        return {
            "id": "evt-created-1",
            "summary": summary,
            "description": description,
            "location": location,
            "start": {"dateTime": start_time.isoformat()},
            "end": {"dateTime": end_time.isoformat()},
        }

    monkeypatch.setattr(main, "create_calendar_event", fake_create_calendar_event)

    response = client.post(
        "/api/calendar/events",
        json={
            "summary": "Design Review",
            "start_time": "2026-04-19T13:00:00+05:45",
            "end_time": "2026-04-19T14:00:00+05:45",
            "description": "Discuss smart mirror API",
            "location": "Innovation Lab",
            "timezone": "Asia/Kathmandu",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "evt-created-1"
    assert captured["summary"] == "Design Review"
    assert captured["location"] == "Innovation Lab"


def test_calendar_update_event_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "update_calendar_event",
        lambda event_id, summary=None, start_time=None, end_time=None, description=None, location=None, timezone_name="Asia/Kathmandu": {
            "id": event_id,
            "summary": summary or "Updated Event",
            "description": description or "",
            "location": location or "",
            "start": {"dateTime": (start_time or datetime.fromisoformat("2026-04-19T15:00:00+05:45")).isoformat()},
            "end": {"dateTime": (end_time or datetime.fromisoformat("2026-04-19T16:00:00+05:45")).isoformat()},
        },
    )

    response = client.put(
        "/api/calendar/events/evt-update-1",
        json={
            "summary": "Updated Review",
            "description": "Updated details",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "evt-update-1"
    assert data["summary"] == "Updated Review"


def test_calendar_delete_event_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "delete_calendar_event", lambda event_id, raise_on_auth_error=False: True)

    response = client.delete("/api/calendar/events/evt-delete-1")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "event_id": "evt-delete-1"}


class _FakeGmailRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeGmailMessagesAPI:
    def list(self, userId="me", labelIds=None, maxResults=10):
        return _FakeGmailRequest({"messages": [{"id": "msg-1"}]})

    def get(self, userId="me", id="", format="full"):
        return _FakeGmailRequest({
            "id": id or "msg-1",
            "threadId": "thread-1",
            "labelIds": ["INBOX", "UNREAD"],
            "snippet": "Hello from Gmail",
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "unit-user@example.com"},
                    {"name": "Subject", "value": "Unit Test Mail"},
                    {"name": "Date", "value": "Sun, 19 Apr 2026 10:00:00 +0545"},
                ],
                "mimeType": "text/plain",
                "body": {"data": "SGVsbG8gZnJvbSBHbWFpbA=="},
            },
        })

    def send(self, userId="me", body=None):
        return _FakeGmailRequest({"id": "msg-sent-1", "threadId": "thread-sent-1"})


class _FakeGmailUsersAPI:
    def messages(self):
        return _FakeGmailMessagesAPI()


class _FakeGmailService:
    def users(self):
        return _FakeGmailUsersAPI()


def test_mail_inbox_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmail_service, "get_gmail_service", lambda username=None, require_send_scope=False: _FakeGmailService())

    response = client.get("/api/mail/inbox")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["messages"][0]["subject"] == "Unit Test Mail"


def test_mail_message_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmail_service, "get_gmail_service", lambda username=None, require_send_scope=False: _FakeGmailService())

    response = client.get("/api/mail/message/msg-1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "msg-1"
    assert "body" in data


def test_mail_send_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gmail_service, "get_gmail_service", lambda username=None, require_send_scope=False: _FakeGmailService())

    response = client.post(
        "/api/mail/send",
        json={
            "to": "recipient@example.com",
            "subject": "Mirror Test",
            "body": "Hello from the smart mirror",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["to"] == "recipient@example.com"


def test_weather_endpoint_returns_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "location": {"name": "Kathmandu", "country": "Nepal"},
                "current": {"temp_c": 22, "condition": {"text": "Sunny"}},
                "forecast": {"forecastday": [{"day": {"mintemp_c": 15, "maxtemp_c": 28}}]},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/api/weather")
    assert response.status_code == 200
    data = response.json()
    assert data["condition"] == "Sunny"
    assert data["location"] == "Kathmandu, Nepal"


def test_news_endpoint_returns_titles(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["interests"] = interests
        captured["location"] = location
        captured["country"] = country
        captured["mode"] = mode
        return [{"title": "Smart mirror assistant reaches new milestone"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["title"] == "Smart mirror assistant reaches new milestone"
    assert captured["mode"] == "world"


def test_news_endpoint_invalid_mode_returns_400(client: TestClient) -> None:
    response = client.get("/api/news?mode=regional")
    assert response.status_code == 400
    assert "mode must be 'world' or 'personalized'" in response.json()["detail"]


def test_news_endpoint_world_pads_to_five_items(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        return [{"title": "Only one world headline"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=world")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert data[0]["title"] == "Only one world headline"
    assert data[1]["title"] == "More world headlines will appear shortly."


def test_news_endpoint_world_empty_returns_default_messages(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        return []

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=world")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert data[0]["title"] == "No world headlines available right now."


def test_news_endpoint_personalized_without_preferences_falls_back_to_world(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "get_user_news_preferences",
        lambda username: {
            "news_interests": "",
            "news_country": "",
            "legacy_interests": "",
            "location": "",
        },
    )

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["mode"] = mode
        return [{"title": "Fallback world headline"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=personalized")
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Fallback world headline"
    assert captured["mode"] == "world"


def test_news_endpoint_personalized_empty_returns_single_placeholder(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["mode"] = mode
        return []

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=personalized&interests=science")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "No news available right now" in data[0]["title"]
    assert captured["mode"] == "personalized"


def test_news_endpoint_personalized_uses_saved_preferences(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        main,
        "get_user_news_preferences",
        lambda username: {
            "news_interests": "science",
            "news_country": "in",
            "legacy_interests": "technology",
            "location": "Kathmandu",
        },
    )

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["interests"] = interests
        captured["location"] = location
        captured["country"] = country
        captured["mode"] = mode
        return [{"title": "Personalized science headline"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=personalized")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["title"] == "Personalized science headline"
    assert captured["mode"] == "personalized"
    assert captured["interests"] == "science"
    assert captured["country"] == "in"


def test_news_endpoint_personalized_country_query_not_fallback_world(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["interests"] = interests
        captured["location"] = location
        captured["country"] = country
        captured["mode"] = mode
        return [{"title": "Nepal focused headline"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=personalized&country=Nepal")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["title"] == "Nepal focused headline"
    assert captured["mode"] == "personalized"
    assert str(captured["country"]).lower() == "nepal"


def test_news_endpoint_personalized_country_with_interest_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_fetch_news_articles(interests=None, location=None, country=None, mode="personalized", limit=5):
        captured["interests"] = interests
        captured["location"] = location
        captured["country"] = country
        captured["mode"] = mode
        return [{"title": "Nepal technology headline"}]

    monkeypatch.setattr(main, "_fetch_news_articles", fake_fetch_news_articles)

    response = client.get("/api/news?mode=personalized&country=Nepal&interests=technology")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["title"] == "Nepal technology headline"
    assert captured["mode"] == "personalized"
    assert str(captured["country"]).lower() == "nepal"
    assert str(captured["interests"]).lower() == "technology"


def test_build_news_url_nepal_uses_everything_query() -> None:
    url = main._build_news_url(
        "unit-test-key",
        interests=None,
        location=None,
        country="Nepal",
        mode="personalized",
    )
    assert "/v2/top-headlines" in url or "/v2/everything" in url
    if "/v2/top-headlines" in url:
        assert "country=np" in url
    else:
        assert "nepal" in urllib.parse.unquote_plus(url).lower()


def test_build_news_url_india_uses_top_headlines_country() -> None:
    url = main._build_news_url(
        "unit-test-key",
        interests=None,
        location=None,
        country="India",
        mode="personalized",
    )
    assert "/v2/top-headlines" in url
    assert "country=in" in url


def test_get_news_preferences_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main,
        "get_user_news_preferences",
        lambda username: {
            "news_interests": "business",
            "news_country": "us",
            "legacy_interests": "technology",
            "location": "Boston, United States",
        },
    )

    response = client.get("/api/news/preferences")
    assert response.status_code == 200
    data = response.json()
    assert data["interests"] == "business"
    assert data["country"] == "us"
    assert data["effective_country"] == "us"


def test_update_news_preferences_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_update_user_news_preferences(username, news_interests=None, news_country=None):
        calls["username"] = username
        calls["news_interests"] = news_interests
        calls["news_country"] = news_country
        return True

    monkeypatch.setattr(main, "update_user_news_preferences", fake_update_user_news_preferences)
    monkeypatch.setattr(
        main,
        "get_user_news_preferences",
        lambda username: {
            "news_interests": "finance",
            "news_country": "gb",
            "legacy_interests": "",
            "location": "London, United Kingdom",
        },
    )

    response = client.put(
        "/api/news/preferences",
        json={"interests": "finance", "country": "United Kingdom"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert calls["username"] == "unit-user"
    assert calls["news_interests"] == "finance"
    assert calls["news_country"] == "gb"


def test_update_news_preferences_requires_payload_values(client: TestClient) -> None:
    response = client.put("/api/news/preferences", json={})
    assert response.status_code == 400
    assert "Provide interests and/or country" in response.json()["detail"]


def test_update_news_preferences_rejects_invalid_country(client: TestClient) -> None:
    response = client.put(
        "/api/news/preferences",
        json={"country": "Atlantis"},
    )
    assert response.status_code == 400
    assert "country must be a valid country name or a 2-letter country code" in response.json()["detail"]


def test_face_verify_upload_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_read_upload_frame(image):
        return b"fake-frame"

    monkeypatch.setattr(main, "FACE_RECOGNITION_AVAILABLE", True)
    monkeypatch.setattr(main, "_read_upload_frame", fake_read_upload_frame)
    monkeypatch.setattr(main, "_verify_face_frame", lambda frame: {"detected": True, "username": "unit-user", "confidence": 88.4})

    response = client.post(
        "/api/face/verify-upload",
        files={"image": ("face.jpg", b"test-image", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["detected"] is True


def test_face_enroll_upload_endpoint_without_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_read_upload_frame(image):
        return b"fake-frame"

    monkeypatch.setattr(main, "FACE_RECOGNITION_AVAILABLE", True)
    monkeypatch.setattr(main, "_read_upload_frame", fake_read_upload_frame)
    monkeypatch.setattr(main, "_prepare_enrollment_embeddings", lambda embeddings: embeddings)
    monkeypatch.setattr(main, "save_face_database", lambda db: None)
    monkeypatch.setattr(main, "face_users_db", {})

    class FakeFaceApp:
        @staticmethod
        def get(frame):
            return [object()]

    monkeypatch.setattr(main, "face_app", FakeFaceApp())
    monkeypatch.setattr(main, "get_face_embedding", lambda frame, face: [0.1, 0.2, 0.3])

    response = client.post(
        "/api/face/enroll-upload",
        data={"username": "unit-user"},
        files=[("images", ("face1.jpg", b"test-image-1", "image/jpeg")), ("images", ("face2.jpg", b"test-image-2", "image/jpeg"))],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["username"] == "unit-user"
    assert data["embeddings_saved"] == 2


def test_face_login_upload_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_read_upload_frame(image):
        return b"fake-frame"

    monkeypatch.setattr(main, "FACE_RECOGNITION_AVAILABLE", True)
    monkeypatch.setattr(main, "_read_upload_frame", fake_read_upload_frame)
    monkeypatch.setattr(
        main,
        "_face_login_from_frame",
        lambda frame, response: {
            "success": True,
            "token": "unit-token",
            "redirect_url": "/?token=unit-token",
            "username": "unit-user",
            "full_name": "Unit Test User",
            "confidence": 91.2,
            "message": "Welcome back, Unit!",
        },
    )

    response = client.post(
        "/api/face/login-upload",
        files={"image": ("face.jpg", b"test-image", "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["username"] == "unit-user"
