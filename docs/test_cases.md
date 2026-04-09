    # 6 Testing

    ## 6.1 Test Plan

    ### 6.1.1 Project Overview

    The `final fixed fyp` project is an AARVIS smart mirror system built with FastAPI. Based on the current codebase, the system includes:

    - Manual registration and login
    - Google OAuth login and cross-device phone-to-PC pairing
    - Face detection, face enrollment, face verification, and face login
    - A mirror dashboard with time, date, greeting, weather, news, and calendar widgets
    - A WebSocket-based voice assistant with STT, TTS, and conversation history
    - Gmail and Google Calendar integration
    - Morning briefing generation
    - Admin dashboard for user management
    - SQLite-based storage for users, face embeddings, attendance, and conversation history
    - CLI-based assistant testing utilities

    Note: The `Status` column is set to `Planned` so it can be updated after actual execution.

    ### 6.1.2 Manual Authentication and Session Management

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC1 | Register with valid username, email, password, and full name | User account is created successfully and session starts | Planned |
    | TC2 | Register with duplicate username | System shows `Username already exists` error | Planned |
    | TC3 | Register with duplicate email | System shows `Email already exists` error | Planned |
    | TC4 | Register with missing required fields | Registration is blocked and validation error is shown | Planned |
    | TC5 | Register with password and confirm password mismatch | Client-side error is shown and request is not submitted | Planned |
    | TC6 | Register with password shorter than 6 characters | Client-side password length warning is shown | Planned |
    | TC7 | Login with valid username and password | User is authenticated and redirected to the mirror dashboard | Planned |
    | TC8 | Login with incorrect password | System shows invalid credentials error | Planned |
    | TC9 | Login with non-existent username | System shows invalid credentials error | Planned |
    | TC10 | Login with empty username or password fields | Form shows required-field error and login does not proceed | Planned |
    | TC11 | Open `/` without valid session | User is redirected to `/login` | Planned |
    | TC12 | Call `/api/user` without valid session | API returns `401 Not authenticated` | Planned |
    | TC13 | Logout from an active session | Session is cleared and user must log in again | Planned |
    | TC14 | Session cookie creation after login/register | Cookie is set with `HttpOnly`, `max_age`, and expected session token | Planned |

    ### 6.1.3 Google OAuth and Cross-Device Pairing

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC15 | Click Google sign-in from register page | User enters the Google OAuth start flow successfully | Planned |
    | TC16 | Click Google sign-in from login page | User enters the Google OAuth start flow successfully | Planned |
    | TC17 | Start Google OAuth with valid server configuration | Browser is redirected to Google consent screen | Planned |
    | TC18 | Google callback with valid code for a new user | Local Google user is created and redirected to face setup if needed | Planned |
    | TC19 | Google callback with valid code for an existing Google-linked user | Existing account is reused and session is created successfully | Planned |
    | TC20 | Existing manual user signs in with Google using the same email | Existing account is linked to Google instead of creating a duplicate user | Planned |
    | TC21 | Google callback with `access_denied` | User is redirected back with cancellation error message | Planned |
    | TC22 | Google callback with missing authorization code | User is redirected back with `no_code` error | Planned |
    | TC23 | Google callback with token exchange failure | User is redirected back with `token_exchange_failed` error | Planned |
    | TC24 | Register page QR code generation | QR code and mobile URL are displayed correctly | Planned |
    | TC25 | Login page QR code generation | QR code and mobile URL are displayed correctly | Planned |
    | TC26 | Open `/mobile-connect` with a valid pair token | Mobile pairing page loads with correct intent | Planned |
    | TC27 | Open `/mobile-connect` with an invalid or expired pair token | Expired QR message is shown | Planned |
    | TC28 | Poll pair status before phone completes OAuth | Pair status remains `pending` until completed | Planned |
    | TC29 | Phone triggers PC fallback using `/api/pair-trigger/{pair_token}` | PC pairing state changes to `triggered` | Planned |
    | TC30 | Phone completes OAuth and user still needs face setup | PC receives session token and redirects to `/setup-face` | Planned |
    | TC31 | Phone completes OAuth and user already has face enrolled | PC receives session token and redirects to dashboard | Planned |
    | TC32 | Google OAuth started from phone with private IP callback blocked | Mobile page shows fallback guidance and PC fallback option | Planned |
    | TC33 | `/api/local-url` chooses correct local/public base URL for QR flow | Returned URL is reachable for the intended device flow | Planned |

    ### 6.1.4 Face Detection, Enrollment, and Face Login

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC34 | Open face setup page after authenticated redirect | Setup page loads and is ready for camera-based enrollment | Planned |
    | TC35 | Call `/api/face/process` with `detect_only=true` and a valid face image | API returns `detected=true` with bbox, center, frame size, and face ratio | Planned |
    | TC36 | Call `/api/face/process` with `detect_only=true` and no visible face | API returns `detected=false` and `No face detected` message | Planned |
    | TC37 | Call `/api/face/process` without image data | API returns `No image provided` error | Planned |
    | TC38 | Face endpoints called while face recognition runtime is unavailable | API returns graceful unavailable message instead of crashing | Planned |
    | TC39 | Start camera with permission granted on setup page | Live camera preview starts successfully | Planned |
    | TC40 | Start camera with permission denied on setup page | User sees camera access error message | Planned |
    | TC41 | Face becomes visible during motion-check stage | `Face detected` checklist item is marked complete | Planned |
    | TC42 | User performs `look up` motion | `Look up` checklist item is marked complete | Planned |
    | TC43 | User performs `look down` motion | `Look down` checklist item is marked complete | Planned |
    | TC44 | User performs `look left` motion | `Look left` checklist item is marked complete | Planned |
    | TC45 | User performs `look right` motion | `Look right` checklist item is marked complete | Planned |
    | TC46 | All motion checks completed successfully | `Register Face Now` button becomes enabled | Planned |
    | TC47 | Face capture remains stable through enrollment sampling | Enough valid frames are captured and embeddings are saved | Planned |
    | TC48 | Face capture is unstable or too few valid frames are collected | User is asked to retry and enrollment is not saved | Planned |
    | TC49 | Enroll face using valid session cookie | Face embeddings are stored for the authenticated user | Planned |
    | TC50 | Enroll face using token query fallback when cookie is stale/missing | Enrollment still succeeds and cookie is reset | Planned |
    | TC51 | Call `/api/face/enroll` without any images | API returns `No images provided` message | Planned |
    | TC52 | Call `/api/face/enroll` with images that contain no detectable face | API returns `No face detected in provided images` | Planned |
    | TC53 | Face login with a registered enrolled user | User is logged in, session token is issued, and welcome message is returned | Planned |
    | TC54 | Face login with an unknown user | Login is rejected with `Face not recognized` style response | Planned |
    | TC55 | Face login request with no face in frame | API returns `No face detected` | Planned |
    | TC56 | Face login request without image data | API returns `No image provided` | Planned |
    | TC57 | Face verification for a recognized dashboard user | API returns username, confidence, and cache duration | Planned |
    | TC58 | Face verification for an unknown face | API returns `detected=false` with confidence score | Planned |
    | TC59 | Check face cache within 4-minute validity window | API returns `cached=true` with remaining seconds | Planned |
    | TC60 | Check face cache after expiry or without authentication | API returns `cached=false` and asks for new verification | Planned |

    ### 6.1.5 Dashboard Bootstrapping and Personalization

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC61 | Open `/` with a valid session | Dashboard page loads successfully | Planned |
    | TC62 | Open `/` without a valid session | User is redirected to login page | Planned |
    | TC63 | Call `/api/user` with valid session | Authenticated user profile is returned correctly | Planned |
    | TC64 | Dashboard opened with `?token=` after OAuth | Token is stored in local storage and removed from URL | Planned |
    | TC65 | Greeting display in morning, afternoon, evening, and night | Greeting updates to the correct time-based message | Planned |
    | TC66 | Time display update on dashboard | Current time refreshes correctly every second | Planned |
    | TC67 | Date display formatting on dashboard | Date shows correct weekday, month, day, and ordinal suffix | Planned |
    | TC68 | Personalized widgets before face verification | Weather, news, and schedule remain hidden or in waiting state | Planned |
    | TC69 | Personalized widgets after successful face verification | Weather, news, and schedule become visible and refresh normally | Planned |
    | TC70 | Dashboard initialization after auth succeeds | Weather, news, calendar, WebSocket, and voice modules start correctly | Planned |
    | TC71 | WebSocket disconnect during dashboard use | Frontend retries connection automatically | Planned |
    | TC72 | Temporary face verification error during periodic checks | System schedules retry without crashing the dashboard | Planned |

    ### 6.1.6 Voice Assistant and WebSocket Interaction

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC73 | Open WebSocket with valid session token | Connection is accepted and assistant session starts | Planned |
    | TC74 | Open WebSocket without valid session token | Socket returns authentication error and closes | Planned |
    | TC75 | Initial WebSocket connection for authenticated user | Welcome status message is sent to frontend | Planned |
    | TC76 | Send a text message through WebSocket | Assistant returns streamed response chunks successfully | Planned |
    | TC77 | End of streamed assistant response | Final response message is delivered and UI resets to ready state | Planned |
    | TC78 | Send browser-recorded audio through WebSocket | Audio is transcribed and transcript event is returned | Planned |
    | TC79 | Send empty or low-quality audio through WebSocket | System reports it could not understand audio without crashing | Planned |
    | TC80 | Voice state lifecycle during a request | Frontend cycles through listening, thinking, speaking, and idle states correctly | Planned |
    | TC81 | TTS chunks returned from server | Browser queues and plays audio sequentially without overlap | Planned |
    | TC82 | User sends a simple greeting like `hello` | Assistant responds naturally without unnecessary workflow failure | Planned |
    | TC83 | User asks about today's schedule through voice/text | Assistant answers using calendar context | Planned |
    | TC84 | User asks for weather through voice/text | Assistant answers using weather tool output | Planned |
    | TC85 | User asks for news through voice/text | Assistant answers using news tool output | Planned |
    | TC86 | User explicitly asks to send an email | Assistant uses email flow and returns appropriate result | Planned |
    | TC87 | Conversation history exists for the user before socket connect | Recent context is loaded into the new assistant session | Planned |
    | TC88 | User message and assistant response after each turn | Both are saved to `conversation_history` | Planned |
    | TC89 | User says `bye`, `logout`, or `sign out` | Goodbye audio is played and logout event is sent to frontend | Planned |
    | TC90 | LLM or tool failure during a live turn | Assistant returns safe fallback message without crashing the WebSocket session | Planned |

    ### 6.1.7 Weather and News Modules

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC91 | Weather request for user with saved location | Weather API uses saved location and returns correct data | Planned |
    | TC92 | Weather request for user without saved location | System falls back to default location | Planned |
    | TC93 | Weather API timeout or failure | Widget shows fallback weather data or friendly error text | Planned |
    | TC94 | Weather widget display values | Temperature, condition, min, max, and location render correctly in C | Planned |
    | TC95 | Periodic weather refresh | Weather data refreshes at configured interval | Planned |
    | TC96 | News request where first interest is a supported category | Category-based headlines are returned | Planned |
    | TC97 | News request where interest is not a supported category | System falls back to query-based news search or general feed | Planned |
    | TC98 | News request for user without saved interests | Default top headlines are returned | Planned |
    | TC99 | News API failure or empty article response | Widget shows fallback message instead of breaking | Planned |
    | TC100 | Periodic news refresh | News panel refreshes at configured interval | Planned |
    | TC101 | Very long headline rendering in news widget | Headlines are truncated cleanly and layout remains stable | Planned |

    ### 6.1.8 Calendar and Scheduling Module

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC102 | Call `/api/calendar` when calendar integration is unavailable | API returns fallback sample events instead of failing | Planned |
    | TC103 | Call `/api/calendar` with valid Google-authenticated calendar data | Events are sorted and formatted correctly for display | Planned |
    | TC104 | Call `/api/calendar` when no events exist today | Empty event list is returned gracefully | Planned |
    | TC105 | Calendar contains an all-day event | Event is rendered as `All Day` | Planned |
    | TC106 | Calendar contains malformed or partially invalid datetime fields | Event is returned with safe fallback formatting instead of crashing | Planned |
    | TC107 | `get_calendar_today` tool for a user with events | Tool returns today's events with event IDs | Planned |
    | TC108 | `get_upcoming_calendar` tool for a user with future events | Tool returns upcoming events with event IDs | Planned |
    | TC109 | Create calendar event with valid title, date, and 24-hour time | Event is created successfully in Google Calendar | Planned |
    | TC110 | Create calendar event with 12-hour input like `2:00 PM` | Time is normalized to 24-hour format before creation | Planned |
    | TC111 | Create calendar event with missing title | Tool returns validation error and does not create event | Planned |
    | TC112 | Create calendar event with invalid time format | Tool returns parse error and does not create event | Planned |
    | TC113 | Update event title only | Event title is updated successfully | Planned |
    | TC114 | Update event start time only | Event start is updated and original duration is preserved | Planned |
    | TC115 | Update event with explicit new start and end time | Event is updated to the requested new time window | Planned |
    | TC116 | Delete event with a valid `event_id` | Event is removed successfully from Google Calendar | Planned |
    | TC117 | Update or delete using invalid `event_id` | System returns graceful failure message | Planned |
    | TC118 | Expired Google Calendar token during request | Token refresh occurs and request completes successfully | Planned |
    | TC119 | Calendar operation for user without Google credentials | System returns safe failure or empty result without crashing | Planned |

    ### 6.1.9 Gmail and Intelligent Email Assistant

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC120 | Fetch unread emails for a Google-authenticated user | Unread emails are returned with sender, subject, and preview | Planned |
    | TC121 | Fetch unread emails when inbox has no unread messages | System returns `No unread emails.` | Planned |
    | TC122 | Summarize latest email by sender name | Latest matching email subject and content summary are returned | Planned |
    | TC123 | Summarize latest email for sender with no matching messages | System returns `No emails found` style response | Planned |
    | TC124 | Send direct email using explicit recipient email and body | Email is sent successfully through Gmail API | Planned |
    | TC125 | Send email using contact name stored in `contacts.csv` | Contact is resolved and email is sent to matched address | Planned |
    | TC126 | Send email using unknown contact name | System asks for a direct email address | Planned |
    | TC127 | Send email with topic only and no body | Assistant auto-composes subject and body, then sends email | Planned |
    | TC128 | Send email with topic plus additional context | Auto-composed email reflects the additional context | Planned |
    | TC129 | Gmail send failure after email composition | System returns failure message along with composed draft content | Planned |
    | TC130 | Expired Gmail token during read/send action | Token refresh occurs and request completes successfully | Planned |
    | TC131 | Gmail read/send request for manual user without Google credentials | System fails gracefully without crashing | Planned |
    | TC132 | Email response formatting after successful send | Result includes recipient and final subject/body information | Planned |

    ### 6.1.10 Morning Briefing Module

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC133 | Trigger morning briefing for authenticated user | Personalized briefing text is generated successfully | Planned |
    | TC134 | Briefing with calendar, weather, and news all available | Briefing includes all available context in final summary | Planned |
    | TC135 | Briefing when one or more upstream data sources fail | Briefing still generates using remaining available data | Planned |
    | TC136 | Trigger briefing without active session | API returns `401 Not authenticated` | Planned |
    | TC137 | TTS playback error during briefing | Briefing text is still returned and server does not crash | Planned |

    ### 6.1.11 Admin Panel and User Management

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC138 | Open admin dashboard page | Admin page loads successfully | Planned |
    | TC139 | Call `/api/admin/users` | All users are returned for admin listing | Planned |
    | TC140 | Call `/api/admin/face-list` | All face-enrolled usernames are returned correctly | Planned |
    | TC141 | Admin stats cards on page load | Total, Google, manual, and face-enrolled counts are accurate | Planned |
    | TC142 | Search users by username | Table filters matching usernames correctly | Planned |
    | TC143 | Search users by full name | Table filters matching full names correctly | Planned |
    | TC144 | Search users by email | Table filters matching emails correctly | Planned |
    | TC145 | Open edit modal for a selected user | Modal is populated with current user details | Planned |
    | TC146 | Update user full name, email, location, or interests | Changes are saved and visible after reload | Planned |
    | TC147 | Update a non-existent user ID | API returns `404 User not found` | Planned |
    | TC148 | Delete an existing user | User is removed from the database successfully | Planned |
    | TC149 | Delete a user who has enrolled face data | User record and face data are both removed | Planned |
    | TC150 | Search returns no matching users | Admin table shows empty-state message instead of broken layout | Planned |

    ### 6.1.12 Database, Security, and CLI Utilities

    | TC No. | Test Case Description | Expected Result | Status |
    |---|---|---|---|
    | TC151 | First application startup with empty database | Required tables and indexes are created successfully | Planned |
    | TC152 | Start application with legacy database schema | Google OAuth columns and required migrations are added correctly | Planned |
    | TC153 | Insert duplicate non-null `google_id` values | Unique index prevents duplicate Google account linkage | Planned |
    | TC154 | Password storage after manual registration | Password is stored as a hash, not plaintext | Planned |
    | TC155 | `verify_user` with correct and incorrect password | Correct password succeeds and wrong password fails | Planned |
    | TC156 | Resolve a valid signed session token | Session is accepted for authenticated routes | Planned |
    | TC157 | Resolve an expired or tampered signed session token | Session is rejected and protected routes deny access | Planned |
    | TC158 | Two users with different Google tokens access Gmail/Calendar | Each user only receives their own Google data | Planned |
    | TC159 | Save multiple conversation records and fetch recent context | Recent context is returned in chronological order | Planned |
    | TC160 | Clear old conversations beyond configured age | Older records are deleted successfully | Planned |
    | TC161 | Get conversation statistics for a user | Total messages, sessions, and intent breakdown are returned correctly | Planned |
    | TC162 | Save, retrieve, update, and delete face embedding in DB helpers | Face embedding lifecycle works correctly | Planned |
    | TC163 | Mark attendance for a user | Attendance record is inserted successfully | Planned |
    | TC164 | Fetch attendance records for today | Correct attendance rows are returned for the current day | Planned |
    | TC165 | Start CLI test mode with existing users in database | CLI lists users and allows user selection | Planned |
    | TC166 | CLI starts with conversation history enabled | Recent history is loaded into the session correctly | Planned |
    | TC167 | Send text message in CLI mode | Assistant responds and conversation is saved | Planned |
    | TC168 | Use `clear` command in CLI mode | In-memory conversation history is reset | Planned |
    | TC169 | Use `history` command in CLI mode | Current conversation preview is displayed | Planned |
    | TC170 | Use `tts on` and `tts off` in CLI mode | Spoken output mode toggles correctly | Planned |
    | TC171 | Use `voice on` and `voice off` in CLI mode | Microphone interaction mode toggles correctly | Planned |
    | TC172 | CLI microphone capture with no speech detected | CLI handles the case gracefully without crashing | Planned |
    | TC173 | Browser compatibility in Chrome | Core features work correctly | Planned |
    | TC174 | Browser compatibility in Edge | Core features work correctly | Planned |
    | TC175 | Browser compatibility in Firefox | Core features work correctly | Planned |
    | TC176 | Temporary network, API, or WebSocket failure during runtime | System retries gracefully and remains usable | Planned |

    ## 6.2 Testing Summary

    This test plan covers the implemented AARVIS smart mirror modules in the `final fixed fyp` project: manual authentication, Google OAuth and QR pairing, face enrollment and recognition, dashboard behavior, voice assistant, weather, news, calendar scheduling, Gmail integration, morning briefing, admin management, database behavior, security checks, CLI utilities, and compatibility testing.

    The test cases are written in report style so they can be copied directly into the project documentation and later updated with real execution statuses such as `Successful` or `Failed`.
