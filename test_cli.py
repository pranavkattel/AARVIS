"""
AARVIS CLI Test - Test the agent pipeline from the command line.
Run: python test_cli.py

This lets you chat with AARVIS in your terminal without needing
the FastAPI server, browser, or WebSocket connection.
TTS is optional - it will speak responses if Kokoro is installed.
Voice mode: type 'voice on' to use microphone input (STT) + spoken output (TTS).
"""

import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agent.graph import agent
from app.database import init_db, get_user_by_username, save_conversation, get_recent_context, DB_PATH


def clean_messages_for_history(messages):
    """Collapse tool interactions into the final AI response."""
    cleaned = []
    pending_tool_results = []

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, 'name', 'tool')
            pending_tool_results.append(f"[{tool_name} result: {message.content}]")
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            continue
        if isinstance(message, AIMessage):
            if pending_tool_results:
                context = "\n".join(pending_tool_results)
                cleaned.append(AIMessage(content=f"{context}\n\n{message.content}"))
                pending_tool_results = []
            else:
                cleaned.append(message)
        elif isinstance(message, HumanMessage):
            cleaned.append(message)

    return cleaned


TEST_USERNAME = None
ENABLE_TTS = False
ENABLE_VOICE = False
LOAD_HISTORY = True
SAVE_HISTORY = True


_whisper_model = None


def _get_whisper_model():
    """Lazy-load faster_whisper model (base, CPU, int8)."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper model (base, CPU, int8)...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper model loaded.")
    return _whisper_model



def listen_from_mic() -> str | None:
    """Record from microphone until silence, then transcribe with faster_whisper."""
    import math
    import pyaudio
    import struct
    import tempfile
    import wave

    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1
    rate = 16000
    silence_threshold = 500
    silence_duration = 1.5
    max_record_seconds = 15

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(
            format=fmt,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )
    except Exception as exc:
        print(f"  [!] Microphone error: {exc}")
        audio.terminate()
        return None

    print("\nListening... (speak now, silence to stop)", flush=True)
    frames = []
    silent_chunks = 0
    chunks_per_second = rate // chunk
    silence_limit = int(silence_duration * chunks_per_second)
    max_chunks = int(max_record_seconds * chunks_per_second)
    started_speaking = False

    try:
        for _ in range(max_chunks):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            samples = struct.unpack(f"<{chunk}h", data)
            rms = math.sqrt(sum(sample * sample for sample in samples) / chunk)

            if rms > silence_threshold:
                started_speaking = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            if started_speaking and silent_chunks >= silence_limit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()

    if not started_speaking or len(frames) < chunks_per_second * 0.3:
        print("\r[!] No speech detected.")
        audio.terminate()
        return None

    print("Transcribing...", end="", flush=True)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            wf = wave.open(tmp_path, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(fmt))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
            wf.close()

        model = _get_whisper_model()
        segments, _ = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(segment.text.strip() for segment in segments).strip()
        os.unlink(tmp_path)
    except Exception as exc:
        print(f"\r[!] Transcription error: {exc}")
        audio.terminate()
        return None
    finally:
        audio.terminate()

    if not text:
        print("\r[!] Could not understand audio.")
        return None

    print(f"\rYou (voice): {text}")
    return text



def get_test_user():
    """Get a user for testing - either from config or by prompting."""
    global TEST_USERNAME

    if TEST_USERNAME:
        user = get_user_by_username(TEST_USERNAME)
        if user:
            return user
        print(f"[!] User '{TEST_USERNAME}' not found in database.")

    import sqlite3

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT username, full_name, location, interests FROM users")
        users = cursor.fetchall()
        conn.close()

        if users:
            print("\nAvailable users in database:")
            for i, (username, full_name, location, interests) in enumerate(users, 1):
                print(f"  {i}. {username} ({full_name}) - {location}, interests: {interests or 'none'}")
            print()

            choice = input("Enter username (or number): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(users):
                chosen = users[int(choice) - 1][0]
            else:
                chosen = choice

            user = get_user_by_username(chosen)
            if user:
                return user
            print(f"[!] User '{chosen}' not found.")
        else:
            print("[!] No users in database. Register one via the web UI first.")
    except Exception as exc:
        print(f"[!] Could not query users: {exc}")

    return None



def speak(text: str):
    """Optional TTS output."""
    if not ENABLE_TTS:
        return
    try:
        from app.services.tts_service import speak as tts_speak
        tts_speak(text)
    except Exception as exc:
        print(f"  [TTS error: {exc}]")



def run_cli():
    global ENABLE_TTS, ENABLE_VOICE

    print("=" * 60)
    print("   AARVIS CLI Test Mode")
    print("=" * 60)
    print()

    init_db()

    user = get_test_user()
    if not user:
        print("\n[ERROR] No valid user. Exiting.")
        sys.exit(1)

    first_name = user['full_name'].split()[0]
    session_id = str(uuid.uuid4())

    try:
        import app.calendar_service as calendar_module
        from app.services.gmail_service import set_current_user as set_gmail_user
        calendar_module.set_current_user(user['username'])
        set_gmail_user(user['username'])
    except Exception:
        pass

    print(f"\nLogged in as: {user['full_name']} ({user['username']})")
    print(f"   Location: {user.get('location', 'N/A')}")
    print(f"   Interests: {user.get('interests', 'N/A')}")
    print(f"   Session: {session_id[:8]}...")
    print()

    messages = []
    if LOAD_HISTORY:
        recent = get_recent_context(user['id'], limit=10)
        for message in recent:
            if message['role'] == 'user':
                messages.append(HumanMessage(content=message['content']))
            else:
                messages.append(AIMessage(content=message['content']))
        if recent:
            print(f"Loaded {len(recent)} messages from conversation history\n")

    print(f"Hello {first_name}! Type your message below.")
    print("Commands: 'quit', 'clear', 'tts on/off', 'voice on/off', 'history'\n")
    print("-" * 60)

    while True:
        try:
            if ENABLE_VOICE:
                user_text = listen_from_mic()
                if user_text is None:
                    continue
            else:
                user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in ('quit', 'exit', 'q'):
            print("\nGoodbye!")
            break
        if lowered == 'clear':
            messages = []
            print("Conversation history cleared.\n")
            continue
        if lowered == 'tts on':
            ENABLE_TTS = True
            print("TTS enabled.\n")
            continue
        if lowered == 'tts off':
            ENABLE_TTS = False
            print("TTS disabled.\n")
            continue
        if lowered == 'voice on':
            ENABLE_VOICE = True
            ENABLE_TTS = True
            print("Voice mode ON - mic input + spoken output. Say 'voice off' to stop.\n")
            continue
        if lowered == 'voice off':
            ENABLE_VOICE = False
            print("Voice mode OFF - back to text input.\n")
            continue
        if lowered == 'history':
            print("\nCurrent conversation:")
            for message in messages:
                role = 'You' if isinstance(message, HumanMessage) else 'AARVIS'
                preview = message.content[:100]
                suffix = '...' if len(message.content) > 100 else ''
                print(f"  [{role}] {preview}{suffix}")
            print()
            continue

        if SAVE_HISTORY:
            save_conversation(user['id'], session_id, 'user', user_text)

        messages.append(HumanMessage(content=user_text))

        print("\nThinking...", end="", flush=True)

        try:
            result = agent.invoke({
                'messages': messages,
                'current_user': user['username'],
                'user_id': user['id'],
                'session_id': session_id,
                'user_location': user.get('location', 'Kathmandu'),
                'user_interests': user.get('interests', 'technology'),
                'voice_state': 'thinking',
                'pending_confirmation': None,
                'pending_action': None,
                'draft_email': None,
                'final_response': None,
                'error': None,
            })

            import re as _re

            final_messages = result['messages']
            response_text = final_messages[-1].content if final_messages else 'No response.'
            response_text = _re.sub(r'<think>[\s\S]*?</think>', '', response_text)
            response_text = _re.sub(r'<think>[\s\S]*$', '', response_text)
            response_text = response_text.strip() or 'No response.'

            messages = clean_messages_for_history(result['messages'])

            if SAVE_HISTORY:
                save_conversation(
                    user['id'], session_id, 'assistant', response_text,
                    agent_type='AARVIS'
                )

            tool_messages = [message for message in result['messages'] if hasattr(message, 'tool_calls') and message.tool_calls]
            if tool_messages:
                for tool_message in tool_messages:
                    for tool_call in tool_message.tool_calls:
                        print(f"\rTool called: {tool_call['name']}({tool_call.get('args', {})})")

            print(f"\rAARVIS: {response_text}")
            speak(response_text)

        except Exception as exc:
            print(f"\rError: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    run_cli()
