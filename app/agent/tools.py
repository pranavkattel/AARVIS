from pathlib import Path
from langchain_core.tools import tool
from app.calendar_service import get_todays_events, get_upcoming_events, add_event_simple, authenticate_google_calendar
import httpx
import csv
import os

# ── Contacts CSV helper ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTACTS_CSV = PROJECT_ROOT / "data" / "contacts.csv"

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
def get_news(interests: str) -> str:
    """Get latest news headlines. Call this when the user asks about news, current events, headlines, or what's happening in the world."""
    API_KEY = "b47750eb5d3a45cda2f4542d117a42e8"
    interest_list = [i.strip().lower() for i in interests.split(',')]
    valid_categories = ['business', 'entertainment', 'health', 'science', 'sports', 'technology']
    category = interest_list[0] if interest_list[0] in valid_categories else None

    if category:
        url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY}"
    else:
        url = f"https://newsapi.org/v2/everything?q={interest_list[0]}&sortBy=publishedAt&apiKey={API_KEY}"

    try:
        response = httpx.get(url, timeout=10.0)
        data = response.json()
        if data.get("status") == "ok" and data.get("articles"):
            headlines = [a["title"] for a in data["articles"][:5]]
            return "Top headlines:\n" + "\n".join(f"- {h}" for h in headlines)
        return "Could not fetch news at this time."
    except Exception as e:
        return f"News fetch failed: {e}"


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

    # MODE 2: body already provided — send directly
    if body.strip():
        final_subject = subject.strip() or topic.strip().capitalize()
        final_body = body.strip()
    else:
        # MODE 1: auto-compose from topic
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
            return f"Failed to compose email: {e}"

    # Send
    try:
        from app.services.gmail_service import get_gmail_service

        service = get_gmail_service()
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
    except Exception as e:
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
    get_news,
    get_emails,
    send_email,
    summarize_email_by_sender,
]
