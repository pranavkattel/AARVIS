# 6 Testing

## 6.1 Test Plan

### 6.1.1 Project Overview
The testing phase for the AARVIS smart mirror evaluates the core integrations: Google OAuth, PyTorch-based Facial Recognition, WebSocket Voice Assistant, and Google Workspace APIs (Calendar & Gmail).

The test cases focus on functional user journeys, intelligent voice interactions, and system-level integrations.

## 6.2 Detailed Test Cases

### 6.2.1 Google OAuth Login
**Table 11: Test of Google OAuth Login**

**Test No:** 1

**Objective:** Verify that a user can authenticate using Google OAuth.

**Action:** 
➔ Navigate to the login page
➔ Click on "Sign in with Google"
➔ Select a Google account and grant permissions

**Expected Result:** User is successfully authenticated and redirected to the dashboard or face setup.

**Actual Result:** User was successfully authenticated and redirected to the dashboard or face setup.

**Conclusion:** Test was successful

---

### 6.2.2 Cross-Device QR Pairing Initialization
**Table 12: Test of Cross-Device QR Pairing Initialization**

**Test No:** 2

**Objective:** Verify that the QR code successfully initiates the mobile pairing flow.

**Action:** 
➔ Scan the QR code displayed on the mirror with a mobile device
➔ Open the generated link on the mobile browser

**Expected Result:** The mobile browser opens the Google OAuth flow tied to the mirror's session.

**Actual Result:** The mobile browser opened the Google OAuth flow tied to the mirror's session.

**Conclusion:** Test was successful

---

### 6.2.3 Cross-Device QR Expiry
**Table 13: Test of Cross-Device QR Expiry**

**Test No:** 3

**Objective:** Verify that expired QR codes are rejected securely.

**Action:** 
➔ Wait for the pair token to expire
➔ Attempt to scan and open the expired QR code link

**Expected Result:** The mobile browser displays a "Token Expired" error message.

**Actual Result:** The mobile browser displayed a "Token Expired" error message.

**Conclusion:** Test was successful

---

### 6.2.4 Mobile-to-PC Auto-Redirect
**Table 14: Test of Mobile-to-PC Auto-Redirect**

**Test No:** 4

**Objective:** Verify the mirror UI updates automatically when mobile pairing completes.

**Action:** 
➔ Complete Google authentication on the mobile device
➔ Observe the smart mirror monitor

**Expected Result:** The mirror automatically redirects to the dashboard without user interaction on the PC.

**Actual Result:** The mirror automatically redirected to the dashboard.

**Conclusion:** Test was successful

---

### 6.2.5 PC Fallback Trigger via Mobile
**Table 15: Test of PC Fallback Trigger via Mobile**

**Test No:** 5

**Objective:** Verify that mobile UI can trigger fallback to local PC login.

**Action:** 
➔ Scan the QR code but tap "Trigger on PC" instead of completing OAuth on mobile

**Expected Result:** The mirror monitor switches to the local PC login fallback flow.

**Actual Result:** The mirror monitor switched to the local PC login fallback flow.

**Conclusion:** Test was successful

---

### 6.2.6 Unauthorized Dashboard Access
**Table 16: Test of Unauthorized Dashboard Access**

**Test No:** 6

**Objective:** Prevent unauthenticated users from accessing the dashboard.

**Action:** 
➔ Clear browser cookies
➔ Navigate directly to the `/` root dashboard URL

**Expected Result:** The system redirects the user to `/login` immediately.

**Actual Result:** The system redirected the user to `/login` immediately.

**Conclusion:** Test was successful

---

### 6.2.7 User Logout
**Table 17: Test of User Logout**

**Test No:** 7

**Objective:** Verify session termination upon logout.

**Action:** 
➔ Say "logout" or click the logout button on the dashboard
➔ Attempt to navigate back to the dashboard

**Expected Result:** Session cookies are cleared, and the user is redirected to the login page.

**Actual Result:** Session cookies were cleared, and the user was redirected to the login page.

**Conclusion:** Test was successful

---

### 6.2.8 Face Detection - Multiple Faces
**Table 18: Test of Face Detection - Multiple Faces**

**Test No:** 8

**Objective:** Verify system behavior when multiple faces are visible during login.

**Action:** 
➔ Position two people in front of the camera
➔ Trigger face detection

**Expected Result:** The system targets the largest/closest bounding box for authentication.

**Actual Result:** The system targeted the largest bounding box for authentication.

**Conclusion:** Test was successful

---

### 6.2.9 Face Liveness Check - Look Left
**Table 19: Test of Face Liveness Check - Look Left**

**Test No:** 9

**Objective:** Verify anti-spoofing motion tracking (Left).

**Action:** 
➔ Start Face Enrollment
➔ Turn head significantly to the left when prompted

**Expected Result:** The "Look Left" checklist item is marked complete.

**Actual Result:** The "Look Left" checklist item was marked complete.

**Conclusion:** Test was successful

---

### 6.2.10 Face Liveness Check - Look Right
**Table 20: Test of Face Liveness Check - Look Right**

**Test No:** 10

**Objective:** Verify anti-spoofing motion tracking (Right).

**Action:** 
➔ Start Face Enrollment
➔ Turn head significantly to the right when prompted

**Expected Result:** The "Look Right" checklist item is marked complete.

**Actual Result:** The "Look Right" checklist item was marked complete.

**Conclusion:** Test was successful

---

### 6.2.11 Face Liveness Check - Look Up
**Table 21: Test of Face Liveness Check - Look Up**

**Test No:** 11

**Objective:** Verify anti-spoofing motion tracking (Up).

**Action:** 
➔ Start Face Enrollment
➔ Tilt head upwards when prompted

**Expected Result:** The "Look Up" checklist item is marked complete.

**Actual Result:** The "Look Up" checklist item was marked complete.

**Conclusion:** Test was successful

---

### 6.2.12 Face Liveness Check - Look Down
**Table 22: Test of Face Liveness Check - Look Down**

**Test No:** 12

**Objective:** Verify anti-spoofing motion tracking (Down).

**Action:** 
➔ Start Face Enrollment
➔ Tilt head downwards when prompted

**Expected Result:** The "Look Down" checklist item is marked complete.

**Actual Result:** The "Look Down" checklist item was marked complete.

**Conclusion:** Test was successful

---

### 6.2.13 Face Enrollment & Embedding Extraction
**Table 23: Test of Face Enrollment & Embedding Extraction**

**Test No:** 13

**Objective:** Verify PyTorch extracts and saves 512-D face embeddings.

**Action:** 
➔ Complete all liveness checks
➔ Click "Register Face Now"

**Expected Result:** The PyTorch MobileNetV2 model generates embeddings and saves them to the database.

**Actual Result:** Embeddings were successfully generated and saved to the database.

**Conclusion:** Test was successful

---

### 6.2.14 Face Login - Authorized User
**Table 24: Test of Face Login - Authorized User**

**Test No:** 14

**Objective:** Verify standard face authentication works for registered users.

**Action:** 
➔ Stand in front of the mirror
➔ Choose "Face Login"

**Expected Result:** Cosine similarity evaluates > 0.40, and the user is logged in.

**Actual Result:** Cosine similarity evaluated > 0.40, and the user was logged in.

**Conclusion:** Test was successful

---

### 6.2.15 Face Login - Unauthorized User
**Table 25: Test of Face Login - Unauthorized User**

**Test No:** 15

**Objective:** Ensure unknown faces cannot access the system.

**Action:** 
➔ Have an unregistered person stand in front of the mirror
➔ Trigger face login

**Expected Result:** Cosine similarity evaluates < 0.40, and access is denied with a failure message.

**Actual Result:** Cosine similarity evaluated < 0.40, and access was denied.

**Conclusion:** Test was successful

---

### 6.2.16 Face Login - No Face in Frame
**Table 26: Test of Face Login - No Face in Frame**

**Test No:** 16

**Objective:** Prevent system crashes when trying to authenticate an empty room.

**Action:** 
➔ Step out of the camera frame
➔ Trigger face login

**Expected Result:** The system returns a gracefully handled "No face detected" message without crashing.

**Actual Result:** The system returned "No face detected" and did not crash.

**Conclusion:** Test was successful

---

### 6.2.17 Face Login - Face Occlusion
**Table 27: Test of Face Login - Face Occlusion**

**Test No:** 17

**Objective:** Reject login attempts when the face is heavily occluded.

**Action:** 
➔ Cover the lower half of the face with a hand or mask
➔ Trigger face login

**Expected Result:** Authentication fails due to insufficient facial landmarks.

**Actual Result:** Authentication failed gracefully as expected.

**Conclusion:** Test was successful

---

### 6.2.18 Face Presence Cache
**Table 28: Test of Face Presence Cache**

**Test No:** 18

**Objective:** Prevent redundant verification queries while the user is actively using the mirror.

**Action:** 
➔ Log in via Face Recognition
➔ Remain in front of the mirror for consecutive checks

**Expected Result:** The system uses cached presence verification to reduce processing overhead.

**Actual Result:** Presence cache prevented redundant API verifications.

**Conclusion:** Test was successful

---

### 6.2.19 Widget Initialization
**Table 29: Test of Widget Initialization**

**Test No:** 19

**Objective:** Ensure widgets load seamlessly with user-specific data.

**Action:** 
➔ Log into the dashboard
➔ Observe the Weather, News, and Clock widgets

**Expected Result:** Widgets initialize immediately loaded with proper authenticated user context.

**Actual Result:** Widgets initialized correctly with the user context.

**Conclusion:** Test was successful

---

### 6.2.20 Time-based Greeting
**Table 30: Test of Time-based Greeting**

**Test No:** 20

**Objective:** Verify dynamic greeting changes based on the local clock.

**Action:** 
➔ Log into the dashboard in the morning (e.g., 9:00 AM)
➔ Observe the main greeting text

**Expected Result:** The UI displays "Good Morning" along with the user's name.

**Actual Result:** The UI displayed "Good Morning" correctly.

**Conclusion:** Test was successful

---

### 6.2.21 Empty Calendar State GUI
**Table 31: Test of Empty Calendar State GUI**

**Test No:** 21

**Objective:** Ensure empty schedules display properly without breaking UI layout.

**Action:** 
➔ Log in with a Google account that has zero events today
➔ Check the calendar widget area

**Expected Result:** The UI gracefully presents a "Free schedule" or "No upcoming events" message.

**Actual Result:** The UI displayed the empty schedule message properly.

**Conclusion:** Test was successful

---

### 6.2.22 Periodic Background Refresh
**Table 32: Test of Periodic Background Refresh**

**Test No:** 22

**Objective:** Ensure information stays up to date without page reloads.

**Action:** 
➔ Leave the dashboard open for the configured refresh interval
➔ Observe news or weather widget

**Expected Result:** The widget data updates dynamically in the background.

**Actual Result:** Widget data updated seamlessly in the background.

**Conclusion:** Test was successful

---

### 6.2.23 Morning Briefing Generation
**Table 33: Test of Morning Briefing Generation**

**Test No:** 23

**Objective:** Validate the LLM synthesizes context into a coherent briefing.

**Action:** 
➔ Trigger the morning briefing sequence
➔ Listen to the generated audio

**Expected Result:** The LLM accurately combines time, weather, and schedule into a fluid verbal greeting.

**Actual Result:** The generated briefing accurately included all contextual data.

**Conclusion:** Test was successful

---

### 6.2.24 Voice Assistant - UI State Synchronization
**Table 34: Test of Voice Assistant - UI State Synchronization**

**Test No:** 24

**Objective:** Ensure visual feedback matches backend processing states.

**Action:** 
➔ Speak to the assistant
➔ Watch the assistant visualizer icon on the dashboard

**Expected Result:** Visualizer cycles strictly through "Listening...", "Thinking...", and "Speaking...".

**Actual Result:** Visualizer cycled accurately matching LangGraph transitions.

**Conclusion:** Test was successful

---

### 6.2.25 Voice-to-Text (STT) Transcription Accuracy
**Table 35: Test of Voice-to-Text (STT) Transcription Accuracy**

**Test No:** 25

**Objective:** Validate Whisper STT correctly processes spoken English.

**Action:** 
➔ Speak "Hello Aarvis, what time is it?" clearly into the microphone

**Expected Result:** The transcribed text exactly matches the spoken intent.

**Actual Result:** The transcript accurately matched the spoken words.

**Conclusion:** Test was successful

---

### 6.2.26 Text-to-Voice (TTS) Playback
**Table 36: Test of Text-to-Voice (TTS) Playback**

**Test No:** 26

**Objective:** Ensure Kokoro TTS synthesizes responses smoothly.

**Action:** 
➔ Ask the assistant a question that yields a multi-sentence answer
➔ Listen to the playback

**Expected Result:** Audio chunks play sequentially without overlapping or cutting off.

**Actual Result:** Audio chunks played smoothly without overlap.

**Conclusion:** Test was successful

---

### 6.2.27 Voice Assistant - Unintelligible Speech
**Table 37: Test of Voice Assistant - Unintelligible Speech**

**Test No:** 27

**Objective:** Handle background noise or mumbled speech gracefully.

**Action:** 
➔ Make random indistinct noises into the microphone

**Expected Result:** Assistant responds asking for clarification (e.g., "I didn't catch that").

**Actual Result:** Assistant asked for clarification gracefully.

**Conclusion:** Test was successful

---

### 6.2.28 Voice Assistant - General Knowledge Inquiry
**Table 38: Test of Voice Assistant - General Knowledge Inquiry**

**Test No:** 28

**Objective:** Ensure standard LLM general knowledge requests function.

**Action:** 
➔ Ask "What is the capital of France?"

**Expected Result:** The assistant voices back "The capital of France is Paris".

**Actual Result:** The assistant answered correctly.

**Conclusion:** Test was successful

---

### 6.2.29 Voice Assistant - Weather Inquiry
**Table 39: Test of Voice Assistant - Weather Inquiry**

**Test No:** 29

**Objective:** Validate tool-calling for real-time weather data.

**Action:** 
➔ Ask "What is the weather right now?"

**Expected Result:** The assistant invokes the weather tool and reads the current conditions.

**Actual Result:** The assistant accurately reported the current tracked weather.

**Conclusion:** Test was successful

---

### 6.2.30 Voice Assistant - Contextual Reasoning (Clothing)
**Table 40: Test of Voice Assistant - Contextual Reasoning (Clothing)**

**Test No:** 30

**Objective:** Evaluate LLM logic connecting weather to user advice.

**Action:** 
➔ Ask "What should I wear today?"

**Expected Result:** Assistant fetches weather and suggests clothes suitable for the temperature.

**Actual Result:** Assistant successfully combined weather context with clothing advice.

**Conclusion:** Test was successful

---

### 6.2.31 Voice Assistant - Conversational Memory
**Table 41: Test of Voice Assistant - Conversational Memory**

**Test No:** 31

**Objective:** Verify session history allows pronoun resolution.

**Action:** 
➔ Ask "Who is the president of the US?"
➔ Then ask "How old is he?"

**Expected Result:** The assistant remembers the context of the previous turn and answers correctly.

**Actual Result:** The assistant correctly resolved the pronoun from memory.

**Conclusion:** Test was successful

---

### 6.2.32 Calendar - Read Today's Schedule
**Table 42: Test of Calendar - Read Today's Schedule**

**Test No:** 32

**Objective:** Validate Google Calendar API read permissions for today.

**Action:** 
➔ Ask "What is on my schedule today?"

**Expected Result:** The assistant fetches today's events and reads them aloud.

**Actual Result:** The assistant accurately read the events for the current date.

**Conclusion:** Test was successful

---

### 6.2.33 Calendar - Read Tomorrow's Schedule
**Table 43: Test of Calendar - Read Tomorrow's Schedule**

**Test No:** 33

**Objective:** Validate datetime boundary shifting for calendar lookups.

**Action:** 
➔ Ask "What do I have tomorrow?"

**Expected Result:** The assistant shifts the query parameters and reads tomorrow's events.

**Actual Result:** The assistant successfully fetched and summarized tomorrow's events.

**Conclusion:** Test was successful

---

### 6.2.34 Calendar - Create Event
**Table 44: Test of Calendar - Create Event**

**Test No:** 34

**Objective:** Validate Google Calendar write permissions and time normalization.

**Action:** 
➔ Ask "Schedule a meeting at 3 PM"

**Expected Result:** The assistant normalizes 3 PM to 15:00 and commits the event to Google Calendar.

**Actual Result:** Event was successfully created in the user's Google Calendar.

**Conclusion:** Test was successful

---

### 6.2.35 Calendar - Prevent Invalid Event Creation
**Table 45: Test of Calendar - Prevent Invalid Event Creation**

**Test No:** 35

**Objective:** Catch logical errors in calendar commands.

**Action:** 
➔ Ask "Schedule a meeting for yesterday"

**Expected Result:** The assistant warns the user that creating an event in the past is invalid.

**Actual Result:** The assistant warned the user and did not invoke the calendar tool.

**Conclusion:** Test was successful

---

### 6.2.36 Calendar - Update Event Time
**Table 46: Test of Calendar - Update Event Time**

**Test No:** 36

**Objective:** Validate updating existing calendar entries.

**Action:** 
➔ Ask "Move my 3 PM meeting to 4 PM"

**Expected Result:** The assistant fetches the event ID and updates the start/end time.

**Actual Result:** The event time was successfully updated in Google Calendar.

**Conclusion:** Test was successful

---

### 6.2.37 Calendar - Delete Event
**Table 47: Test of Calendar - Delete Event**

**Test No:** 37

**Objective:** Validate event cancellation via voice.

**Action:** 
➔ Ask "Cancel my 4 PM meeting"

**Expected Result:** The assistant confirms the deletion and executes the API delete request.

**Actual Result:** The event was successfully canceled from the calendar.

**Conclusion:** Test was successful

---

### 6.2.38 Gmail - Check New Emails
**Table 48: Test of Gmail - Check New Emails**

**Test No:** 38

**Objective:** Validate fetching unread messages from Google Mail.

**Action:** 
➔ Ask "Do I have any new emails?"

**Expected Result:** The assistant queries the Gmail API and states the number of unread emails.

**Actual Result:** Assistant accurately queried and reported unread emails.

**Conclusion:** Test was successful

---

### 6.2.39 Gmail - Summarize Emails
**Table 49: Test of Gmail - Summarize Emails**

**Test No:** 39

**Objective:** Validate extraction of email snippets for summarization.

**Action:** 
➔ Ask "Summarize my recent emails"

**Expected Result:** The assistant reads a synthesized summary of the most recent unread message subjects/snippets.

**Actual Result:** Assistant provided a concise summary of the unread emails.

**Conclusion:** Test was successful

---

### 6.2.40 Gmail - Query Emails by Sender
**Table 50: Test of Gmail - Query Emails by Sender**

**Test No:** 40

**Objective:** Validate targeted search inside the Gmail tool.

**Action:** 
➔ Ask "Did I get an email from John?"

**Expected Result:** The assistant filters unread emails specifically looking for the sender "John".

**Actual Result:** Assistant accurately filtered the inbox and responded.

**Conclusion:** Test was successful

---

### 6.2.41 Gmail - Draft & Send Email via Contacts
**Table 51: Test of Gmail - Draft & Send Email via Contacts**

**Test No:** 41

**Objective:** Verify the full automated email composition and sending flow.

**Action:** 
➔ Ask "Send an email to John about the project meeting"

**Expected Result:** Assistant resolves "John" against contacts.csv, drafts the email, and sends it via Gmail API.

**Actual Result:** Email was drafted and successfully sent to the resolved address.

**Conclusion:** Test was successful

---

### 6.2.42 Gmail - Draft Email for Unknown Contact
**Table 52: Test of Gmail - Draft Email for Unknown Contact**

**Test No:** 42

**Objective:** Handle cases where the requested contact address does not exist.

**Action:** 
➔ Ask "Send an email to Unknown Person"

**Expected Result:** Assistant fails to resolve the name and explicitly asks the user for the email address.

**Actual Result:** Assistant correctly asked for clarification on the email address.

**Conclusion:** Test was successful

---

### 6.2.43 Error Handling - OAuth Token Expiry
**Table 53: Test of Error Handling - OAuth Token Expiry**

**Test No:** 43

**Objective:** Validate the middleware handles expired Google OAuth Access Tokens.

**Action:** 
➔ Force the Google Access Token to expire in the DB
➔ Request a calendar query

**Expected Result:** The system silently uses the Refresh Token to get a new Access Token and completes the query.

**Actual Result:** System silently refreshed the token and completed the query.

**Conclusion:** Test was successful

---

### 6.2.44 Error Handling - External API Outages
**Table 54: Test of Error Handling - External API Outages**

**Test No:** 44

**Objective:** Maintain frontend resiliency when upstream APIs fail.

**Action:** 
➔ Block internet access to the Weather API domain
➔ Load dashboard

**Expected Result:** The weather widget displays a safe fallback message without crashing the UI.

**Actual Result:** The widget displayed a safe error boundary message.

**Conclusion:** Test was successful

---

### 6.2.45 CLI Mode Interaction
**Table 55: Test of CLI Mode Interaction**

**Test No:** 45

**Objective:** Verify text-based CLI testing mode functions detached from frontend.

**Action:** 
➔ Run the CLI test script
➔ Type a query into the console

**Expected Result:** The assistant processes the query and returns text output cleanly to the console.

**Actual Result:** The assistant responded correctly in the CLI environment.

**Conclusion:** Test was successful

---

