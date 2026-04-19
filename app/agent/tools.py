from pathlib import Path
from langchain_core.tools import tool
from app.calendar_service import get_todays_events, get_upcoming_events, add_event_simple, authenticate_google_calendar
from app.database import get_user_news_preferences, update_user_news_preferences
from app.services.google_oauth import GoogleReauthRequiredError
from googleapiclient.errors import HttpError
import httpx
import csv
import os
import urllib.parse
import xml.etree.ElementTree as ET

# ── Contacts CSV helper ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTACTS_CSV = PROJECT_ROOT / "data" / "contacts.csv"
_current_username: str | None = None


def set_current_user(username: str | None) -> None:
    """Set active username for tools that need per-user preferences."""
    global _current_username
    _current_username = username


def get_current_user() -> str | None:
    """Return active username for the current agent session."""
    return _current_username

def lookup_contact(name: str) -> str | None:
    """Look up email address by name from contacts.csv. Case-insensitive."""
    if not os.path.exists(CONTACTS_CSV):
        return None
    try:
        with open(CONTACTS_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('name', '').strip().lower() == name.strip().lower():
                    return row.get('email', '').strip()
    except Exception:
        pass
    return None


def _infer_news_country_code(location: str | None) -> str | None:
    if not location:
        return None
    lowered = location.lower()
    mapping = {
        "united states": "us",
        "usa": "us",
        "america": "us",
        "united kingdom": "gb",
        "uk": "gb",
        "england": "gb",
        "india": "in",
        "nepal": "np",
        "australia": "au",
        "canada": "ca",
        "france": "fr",
        "germany": "de",
        "japan": "jp",
        "singapore": "sg",
        "uae": "ae",
        "united arab emirates": "ae",
    }
    for token, code in mapping.items():
        if token in lowered:
            return code
    return None


NEWSAPI_TOP_HEADLINES_COUNTRIES = {
    "ar", "au", "at", "be", "br", "bg", "ca", "cn", "co", "cu", "cz", "eg", "fr", "de", "gr",
    "hk", "hu", "in", "id", "ie", "il", "it", "jp", "lv", "lt", "my", "mx", "ma", "nl", "nz",
    "ng", "no", "np", "ph", "pl", "pt", "ro", "ru", "sa", "rs", "sg", "sk", "si", "za", "kr", "se",
    "ch", "tw", "th", "tr", "ae", "ua", "gb", "us", "ve"
}

NEWS_COUNTRY_QUERY_KEYWORDS = {
    "np": "nepal",
}


def _resolve_country_query_text(country: str | None, country_code: str | None) -> str | None:
    raw = (country or "").strip()
    if raw and len(raw) > 2:
        return raw
    if country_code:
        return NEWS_COUNTRY_QUERY_KEYWORDS.get(country_code, raw or country_code)
    return raw or None


def _normalize_news_country(country: str | None, location: str | None = None) -> str | None:
    cleaned = (country or "").strip().lower()
    if cleaned:
        if len(cleaned) == 2 and cleaned.isalpha():
            return cleaned
        inferred = _infer_news_country_code(cleaned)
        if inferred:
            return inferred
    return _infer_news_country_code(location)
# ───────────────────────────────────────────────────────────────


@tool
def get_calendar_today() -> str:
    """Get all of today's calendar events. Call this when the user asks what's on their schedule today, asks about today's meetings, or says 'what do I have today'."""
    try:
        events = get_todays_events()
        if not events:
            return "No events scheduled for today."
        lines = []
        for e in events:
            summary = e.get('summary', 'Untitled')
            start = e['start'].get('dateTime', e['start'].get('date'))
            event_id = e.get('id', 'unknown')
            lines.append(f"- {summary} at {start} (event_id: {event_id})")
        return "\n".join(lines)
    except Exception as ex:
        return f"Could not fetch today's events: {ex}"


@tool
def get_upcoming_calendar(max_results: int = 5) -> str:
    """Get upcoming calendar events. Call this when the user asks about future events, upcoming meetings, or what's coming up on their calendar."""
    try:
        events = get_upcoming_events(max_results)
        if not events:
            return "No upcoming events found."
        lines = []
        for e in events:
            summary = e.get('summary', 'Untitled')
            start = e['start'].get('dateTime', e['start'].get('date'))
            event_id = e.get('id', 'unknown')
            lines.append(f"- {summary} at {start} (event_id: {event_id})")
        return "\n".join(lines)
    except Exception as ex:
        return f"Could not fetch upcoming events: {ex}"


@tool
def create_calendar_event(title: str, date: str, time: str, duration_minutes: int = 60, description: str = "") -> str:
    """
    Create a new calendar event. ONLY call this when the user explicitly asks to create/schedule/book an event
    AND you have confirmed the details with them.
    date must be YYYY-MM-DD. time must be HH:MM in 24-hour format (e.g. 14:00, NOT 2:00 PM).
    """
    # Sanitize empty strings from LLM
    if not title or not title.strip():
        return "Error: event title is required. Ask the user what to name the event."
    if not date or not date.strip():
        return "Error: date is required in YYYY-MM-DD format."
    if not time or not time.strip():
        return "Error: time is required in HH:MM 24-hour format."

    # Normalize time — handle "08:00 PM", "2:00 PM", "14:00" etc.
    time = time.strip()
    try:
        from datetime import datetime as dt
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
            try:
                parsed = dt.strptime(time, fmt)
                time = parsed.strftime("%H:%M")  # always convert to 24h
                break
            except ValueError:
                continue
        else:
            return f"Error: could not parse time '{time}'. Use HH:MM 24-hour format like 14:00."
    except Exception:
        pass

    if isinstance(duration_minutes, str):
        duration_minutes = int(duration_minutes) if duration_minutes.strip().isdigit() else 60

    try:
        result = add_event_simple(title.strip(), date.strip(), time, duration_minutes, description or "")
        if result:
            event_id = result.get('id', 'unknown')
            return f"Event '{title}' created on {date} at {time} for {duration_minutes} minutes. (event_id: {event_id})"
        return "Failed to create event. Check Google Calendar credentials."
    except Exception as ex:
        return f"Failed to create event: {ex}"


@tool
def delete_calendar_event(event_id: str) -> str:
    """Delete a calendar event by its event ID. Always confirm with the user before calling this tool."""
    try:
        service = authenticate_google_calendar()
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        return f"Event {event_id} deleted successfully."
    except Exception as e:
        return f"Failed to delete event: {e}"


@tool
def update_calendar_event(event_id: str, new_title: str = None, new_start: str = None, new_end: str = None) -> str:
    """
    Update an existing calendar event. Use this when the user wants to reschedule or rename a meeting.
    Always confirm with the user before calling this tool.
    new_start and new_end must be ISO 8601 datetime strings (e.g. 2026-02-18T23:00:00).
    If only new_start is given, the end time will be auto-calculated to keep the same duration.
    """
    try:
        service = authenticate_google_calendar()
        event = service.events().get(calendarId='primary', eventId=event_id).execute()

        if new_title:
            event['summary'] = new_title

        if new_start:
            # Get original duration to preserve it
            from datetime import datetime as dt
            orig_start_str = event['start'].get('dateTime', '')
            orig_end_str = event['end'].get('dateTime', '')
            orig_tz = event['start'].get('timeZone', 'Asia/Kathmandu')

            # Calculate original duration
            try:
                # Strip timezone offset for parsing
                os_clean = orig_start_str[:19]
                oe_clean = orig_end_str[:19]
                orig_start = dt.fromisoformat(os_clean)
                orig_end = dt.fromisoformat(oe_clean)
                duration = orig_end - orig_start
            except Exception:
                from datetime import timedelta as td
                duration = td(hours=1)

            # Set new start with timezone
            new_start_clean = new_start[:19]  # strip any tz suffix the model adds
            event['start']['dateTime'] = new_start_clean
            event['start']['timeZone'] = orig_tz

            if new_end:
                new_end_clean = new_end[:19]
                event['end']['dateTime'] = new_end_clean
                event['end']['timeZone'] = orig_tz
            else:
                # Auto-calculate end from original duration
                new_start_dt = dt.fromisoformat(new_start_clean)
                new_end_dt = new_start_dt + duration
                event['end']['dateTime'] = new_end_dt.isoformat()
                event['end']['timeZone'] = orig_tz

        updated = service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        final_time = updated['start'].get('dateTime', 'unknown')
        return f"Event updated: {updated.get('summary')} now at {final_time}"
    except Exception as e:
        return f"Failed to update event: {e}"


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location. Call this when the user asks about the weather, temperature, or forecast."""
    API_KEY = "10428bba45b34ba8b4543622252612"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&days=1"
    try:
        response = httpx.get(url, timeout=10.0)
        data = response.json()
        current = data['current']
        forecast = data['forecast']['forecastday'][0]['day']
        return (
            f"Weather in {location}: {current['temp_c']}°C, {current['condition']['text']}. "
            f"High: {forecast['maxtemp_c']}°C, Low: {forecast['mintemp_c']}°C."
        )
    except Exception as e:
        return f"Could not fetch weather: {e}"


@tool
def set_news_preferences(interests: str | None = None, country: str | None = None) -> str:
    """Save personalized news interests and/or country for the current signed-in user."""
    username = get_current_user()
    if not username:
        return "I could not determine the signed-in user to save news preferences."

    clean_interests = interests.strip() if interests is not None else None
    clean_country = country.strip() if country is not None else None

    if clean_interests is None and clean_country is None:
        return "Please provide at least one value: interests or country."

    normalized_country = clean_country
    if clean_country is not None and clean_country != "":
        normalized_country = _normalize_news_country(clean_country)
        if not normalized_country:
            return "I could not recognize that country. Please provide a country name or 2-letter code (for example: Nepal or NP)."

    try:
        updated = update_user_news_preferences(
            username,
            news_interests=clean_interests,
            news_country=normalized_country,
        )
        if not updated:
            return "I could not save your news preferences right now."

        saved = get_user_news_preferences(username) or {}
        saved_interests = (saved.get("news_interests") or "").strip()
        saved_country = (saved.get("news_country") or "").strip()
        return (
            "Saved your news preferences. "
            f"Interests: {saved_interests or 'none'}. Country: {saved_country or 'none'}."
        )
    except Exception as ex:
        return f"Could not save news preferences: {ex}"


@tool
def get_news(
    personalized: bool = False,
    interests: str | None = None,
    location: str | None = None,
    country: str | None = None,
) -> str:
    """Get latest news with details (source, time, summary, link). Defaults to world news unless user-specific filters are provided."""
    api_key = os.getenv("NEWS_API_KEY", "").strip() or "b47750eb5d3a45cda2f4542d117a42e8"

    selected_interests = (interests or "").strip()
    selected_location = (location or "").strip()
    selected_country = (country or "").strip()
    has_explicit_filters = bool(selected_interests or selected_location or selected_country)
    use_personalized = personalized or has_explicit_filters

    if use_personalized and not has_explicit_filters:
        username = get_current_user()
        if username:
            saved = get_user_news_preferences(username) or {}
            selected_interests = (
                (saved.get("news_interests") or "").strip()
                or (saved.get("legacy_interests") or "").strip()
            )
            selected_country = (saved.get("news_country") or "").strip()
            selected_location = (saved.get("location") or "").strip()
        has_explicit_filters = bool(selected_interests or selected_location or selected_country)

    normalized_country = _normalize_news_country(selected_country, selected_location)
    if not normalized_country and selected_interests:
        normalized_country = _normalize_news_country(selected_interests)
    country_query_text = _resolve_country_query_text(selected_country, normalized_country)
    country_supported = bool(normalized_country and normalized_country in NEWSAPI_TOP_HEADLINES_COUNTRIES)

    if use_personalized and not (selected_interests or country_query_text or selected_location):
        return (
            "I can personalize your news. Tell me at least one interest or country, "
            "or ask me to save your news preferences first."
        )

    diagnostics = []
    detail_items = []

    def _fmt_article(index: int, title: str, source: str, published: str, summary: str, link: str) -> str:
        lines = [f"{index}. {title}"]
        meta = []
        if source:
            meta.append(f"Source: {source}")
        if published:
            meta.append(f"Published: {published}")
        if meta:
            lines.append(" | ".join(meta))
        if summary:
            lines.append(f"Summary: {summary}")
        if link:
            lines.append(f"Link: {link}")
        return "\n".join(lines)

    def _normalize_time(raw_time: str | None) -> str:
        if not raw_time:
            return ""
        try:
            return raw_time.replace("T", " ").replace("Z", " UTC")
        except Exception:
            return raw_time

    if not use_personalized:
        world_query = urllib.parse.quote_plus("world OR global")
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={world_query}&language=en&sortBy=publishedAt&pageSize=20&apiKey={api_key}"
        )
        rss_query_text = "world headlines"
    else:
        interest_list = [i.strip().lower() for i in selected_interests.split(',') if i.strip()]
        valid_categories = ['business', 'entertainment', 'health', 'science', 'sports', 'technology']
        primary_interest = interest_list[0] if interest_list else ""
        category = primary_interest if primary_interest in valid_categories else None

        if category and country_supported:
            url = f"https://newsapi.org/v2/top-headlines?country={normalized_country}&category={category}&pageSize=20&apiKey={api_key}"
        elif category and country_query_text:
            query = urllib.parse.quote_plus(f"{category} {country_query_text} {selected_location}".strip() or category)
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={api_key}"
        elif category:
            query = urllib.parse.quote_plus(f"{category} {selected_location}".strip() or category)
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={api_key}"
        elif country_supported and not primary_interest:
            url = f"https://newsapi.org/v2/top-headlines?country={normalized_country}&pageSize=20&apiKey={api_key}"
        elif country_query_text and not primary_interest:
            query = urllib.parse.quote_plus(country_query_text)
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={api_key}"
        else:
            query_text = " ".join([p for p in [primary_interest, country_query_text, selected_location] if p]) or "world"
            query = urllib.parse.quote_plus(query_text)
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={api_key}"

        rss_query_text = " ".join([p for p in [selected_interests, country_query_text, selected_location] if p]).strip() or "world"

    # Primary source: NewsAPI
    try:
        response = httpx.get(url, timeout=10.0)
        data = response.json()
        if data.get("status") == "ok" and data.get("articles"):
            for idx, article in enumerate(data["articles"][:5], start=1):
                title = (article.get("title") or "").strip()
                if not title:
                    continue
                source = ((article.get("source") or {}).get("name") or "").strip()
                published = _normalize_time(article.get("publishedAt"))
                summary = (article.get("description") or article.get("content") or "").strip()
                link = (article.get("url") or "").strip()
                detail_items.append(_fmt_article(idx, title, source, published, summary, link))

            if detail_items:
                return "Top news with details:\n\n" + "\n\n".join(detail_items)
        diagnostics.append(f"newsapi:{data.get('code', 'no_data')}")
    except Exception as e:
        diagnostics.append(f"newsapi_error:{e}")

    # Fallback source: Google News RSS (no key required)
    rss_query = urllib.parse.quote_plus(rss_query_text)
    rss_url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        rss_response = httpx.get(rss_url, timeout=10.0, follow_redirects=True)
        if rss_response.status_code == 200 and rss_response.text:
            root = ET.fromstring(rss_response.text)
            details = []
            for item in root.iter():
                if not item.tag.endswith("item"):
                    continue
                title = ""
                link = ""
                published = ""
                for child in item:
                    if child.tag.endswith("title") and child.text:
                        title = child.text.strip()
                    elif child.tag.endswith("link") and child.text:
                        link = child.text.strip()
                    elif child.tag.endswith("pubDate") and child.text:
                        published = child.text.strip()

                if title:
                    details.append(_fmt_article(len(details) + 1, title, "Google News RSS", published, "", link))
                if len(details) >= 5:
                    break

            if details:
                return "Top news with details:\n\n" + "\n\n".join(details)

        diagnostics.append(f"rss_http:{rss_response.status_code}")
    except Exception as e:
        diagnostics.append(f"rss_error:{e}")

    if diagnostics:
        return f"Could not fetch news at this time. ({diagnostics[0]})"
    return "Could not fetch news at this time."


@tool
def get_emails(max_results: int = 5) -> str:
    """
    Get recent unread emails from Gmail with subject, sender, and preview.
    Call this when the user asks to check their email, read emails, see new messages,
    or wants to know what emails they have.
    """
    try:
        from app.services.gmail_service import get_gmail_service
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            return "No unread emails."
        summaries = []
        for msg in messages:
            m = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = {h['name']: h['value'] for h in m['payload']['headers']}
            sender = headers.get('From', '?')
            subject = headers.get('Subject', '?')
            snippet = m.get('snippet', '')[:200]
            summaries.append(f"- From: {sender} | Subject: {subject}\n  Preview: {snippet}")
        return "\n".join(summaries)
    except Exception as e:
        return f"Could not fetch emails: {e}"


@tool
def send_email(to: str, topic: str, subject: str = "", body: str = "", additional_context: str = "") -> str:
    """
    Send an email. Handles two modes automatically:

    MODE 1 — Auto-compose (preferred): User gives a short description and wants the email written for them.
      Use when the user says things like:
      - 'write an email to John saying he is fired'
      - 'send an email to Sarah about the meeting cancellation'
      - 'generate a professional email to HR about the new policy'
      Provide 'to' and 'topic'. Leave 'subject' and 'body' empty — the email is composed automatically.

    MODE 2 — Direct send: User provides the full email content themselves.
      Provide 'to', 'subject', and 'body' explicitly. 'topic' can be a short label.

    'to' can be a name (looked up in contacts.csv) or a direct email address.
    'additional_context' adds extra detail when auto-composing (optional).
    """
    import re as _re
    import base64
    from email.mime.text import MIMEText

    def _looks_like_google_auth_issue(err_text: str) -> bool:
        lowered = (err_text or "").lower()
        markers = [
            "refresh token",
            "missing refresh token",
            "missing gmail send permission",
            "insufficient authentication scopes",
            "insufficientpermissions",
            "invalid_grant",
            "token has been expired or revoked",
            "invalid credentials",
            "unauthorized",
            "401",
            "reauth",
            "re-auth",
            "credentials do not contain the necessary fields",
        ]
        return any(marker in lowered for marker in markers)

    # Resolve recipient
    recipient_email = to
    recipient_label = to
    if '@' not in to:
        found = lookup_contact(to)
        if found:
            recipient_email = found
            recipient_label = f"{to} ({found})"
        else:
            return f"Could not find an email address for '{to}' in contacts. Please provide a direct email address."

    def _build_fallback_email(_topic: str, _subject: str, _context: str) -> tuple[str, str]:
        fallback_subject = (_subject or _topic or "Message").strip()
        fallback_subject = fallback_subject[:120] if fallback_subject else "Message"

        context_block = _context.strip()
        body_lines = [
            f"Hi {recipient_label.split('(')[0].strip()},",
            "",
            f"I wanted to reach out regarding: {_topic.strip() or 'the requested topic'}.",
        ]
        if context_block:
            body_lines.extend(["", f"Additional details: {context_block}"])
        body_lines.extend([
            "",
            "Please let me know your thoughts.",
            "",
            "Best regards,",
            "AARVIS",
        ])
        return fallback_subject, "\n".join(body_lines)

    # MODE 2: body already provided — send directly
    if body.strip():
        final_subject = subject.strip() or topic.strip().capitalize()
        final_body = body.strip()
    else:
        # MODE 1: auto-compose from topic
        compose_error = None
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage, SystemMessage as SM

            composer = ChatOllama(model="qwen3:4b", temperature=0.7)
            context_line = f"\nAdditional context: {additional_context}" if additional_context.strip() else ""
            compose_prompt = (
                f"Write a complete, professional email on the following topic.\n"
                f"Topic: {topic}{context_line}\n\n"
                f"Output ONLY the email — include a clear Subject line at the top (format: 'Subject: ...'),"
                f" a proper greeting, detailed body paragraphs, and a professional sign-off.\n"
                f"Do NOT include any explanation outside the email itself."
            )
            response = composer.invoke([SM(content="/no_think"), HumanMessage(content=compose_prompt)])
            composed = _re.sub(r"<think>.*?</think>", "", response.content, flags=_re.DOTALL).strip()

            # Extract Subject line from composed text
            final_subject = subject.strip() or topic.strip().capitalize()
            final_body = composed
            for line in composed.splitlines():
                if line.lower().startswith("subject:"):
                    final_subject = line[len("subject:"):].strip()
                    final_body = composed[composed.index(line) + len(line):].strip()
                    break
        except Exception as e:
            compose_error = e
            final_subject, final_body = _build_fallback_email(topic, subject, additional_context)
            print(f"[Email] Auto-compose unavailable; using fallback template. Reason: {e}")

        if not final_body.strip():
            final_subject, final_body = _build_fallback_email(topic, subject, additional_context)
            if compose_error:
                print("[Email] Empty compose output; fallback template applied.")

    # Send
    try:
        from app.services.gmail_service import get_gmail_service

        service = get_gmail_service(require_send_scope=True)
        message = MIMEText(final_body)
        message['to'] = recipient_email
        message['subject'] = final_subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return (
            f"Email sent to {recipient_label}.\n"
            f"Subject: {final_subject}\n\n"
            f"--- Email Body ---\n{final_body}"
        )
    except GoogleReauthRequiredError as e:
        return f"[REAUTH_REQUIRED_GOOGLE] {e}"
    except HttpError as e:
        raw_error = str(e)
        lowered = raw_error.lower()
        if "insufficient authentication scopes" in lowered or "insufficientpermissions" in lowered:
            return (
                "[REAUTH_REQUIRED_GOOGLE] "
                "Google account is connected without Gmail send permission. "
                "Please re-authenticate to allow sending email."
            )
        if "invalid_grant" in lowered or "token has been expired or revoked" in lowered:
            return (
                "[REAUTH_REQUIRED_GOOGLE] "
                "Google token is expired or revoked. Please re-authenticate."
            )
        return f"Email failed to send: {e}\n\nComposed email:\nSubject: {final_subject}\n\n{final_body}"
    except Exception as e:
        if _looks_like_google_auth_issue(str(e)):
            return (
                "[REAUTH_REQUIRED_GOOGLE] "
                "Google authentication is incomplete or expired. "
                "Please re-authenticate your Google account."
            )
        return f"Email failed to send: {e}\n\nComposed email:\nSubject: {final_subject}\n\n{final_body}"


@tool
def summarize_email_by_sender(sender_name: str) -> str:
    """
    Summarize the latest email from a specific sender. Use this when the user says
    'summarize my email from John', 'what did Alex send me?', or 'read the email from Sarah'.
    """
    try:
        from app.services.gmail_service import get_gmail_service
        import base64

        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me', q=f"from:{sender_name}", maxResults=1
        ).execute()
        messages = results.get('messages', [])
        if not messages:
            return f"No emails found from {sender_name}."
        msg = service.users().messages().get(userId='me', id=messages[0]['id'], format='full').execute()
        # Extract subject
        headers = {h['name']: h['value'] for h in msg['payload']['headers']}
        subject = headers.get('Subject', 'No subject')
        # Extract body text
        parts = msg['payload'].get('parts', [])
        body = ""
        for part in parts:
            if part['mimeType'] == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
        if not body:
            body = msg.get('snippet', 'Could not extract email body.')
        return f"Email from {sender_name} — Subject: {subject}\n{body[:500]}..."
    except Exception as e:
        return f"Could not retrieve email: {e}"


# All tools list — bind to the model
tools = [
    get_calendar_today,
    get_upcoming_calendar,
    create_calendar_event,
    delete_calendar_event,
    update_calendar_event,
    get_weather,
    set_news_preferences,
    get_news,
    get_emails,
    send_email,
    summarize_email_by_sender,
]
