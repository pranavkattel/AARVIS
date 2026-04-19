# Unit Testing

## 4.1.1 Enroll Face Using Uploaded Images

<table>
  <tr><td>Test Case ID</td><td>UT01</td></tr>
  <tr><td>Test Case Name</td><td>Enroll Face Using Uploaded Images</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. The user account already exists in the system. Face recognition is enabled.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to POST.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/face/enroll-upload</code><br>4. Add form field <code>username</code> with the account username.<br>5. Upload multiple JPG or PNG face images under <code>images</code>.<br>6. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with <code>success=true</code>, enrolled username, embeddings saved count, and model name.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with <code>success=true</code>, enrolled username, embeddings saved count, and model name.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.2 Verify Face Using Uploaded Image

<table>
  <tr><td>Test Case ID</td><td>UT02</td></tr>
  <tr><td>Test Case Name</td><td>Verify Face Using Uploaded Image</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Face recognition service is enabled. At least one face profile is already enrolled.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to POST.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/face/verify-upload</code><br>4. Upload a JPG or PNG face image.<br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with <code>detected</code>, confidence score, and matched username if the face is recognized.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with <code>detected</code>, confidence score, and matched username if the face is recognized.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.3 Face Login Using Uploaded Image

<table>
  <tr><td>Test Case ID</td><td>UT03</td></tr>
  <tr><td>Test Case Name</td><td>Face Login Using Uploaded Image</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Face recognition is enabled. The user has already enrolled a face profile and linked Google Sign-In.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to POST.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/face/login-upload</code><br>4. Upload a registered user's face image.<br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with <code>success=true</code>, session token, username, full name, and redirect URL.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with <code>success=true</code>, session token, username, full name, and redirect URL.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.4 Get Authenticated User Profile

<table>
  <tr><td>Test Case ID</td><td>UT04</td></tr>
  <tr><td>Test Case Name</td><td>Get Authenticated User Profile</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/user</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing the authenticated user's profile and active session token.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing the authenticated user's profile and active session token.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.5 Get Today's Calendar Events

<table>
  <tr><td>Test Case ID</td><td>UT05</td></tr>
  <tr><td>Test Case Name</td><td>Get Today's Calendar Events for Authenticated User</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Google Calendar is connected.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing today's calendar events in formatted form.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing today's calendar events in formatted form.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.6 Get Upcoming Calendar Events

<table>
  <tr><td>Test Case ID</td><td>UT06</td></tr>
  <tr><td>Test Case Name</td><td>Get Upcoming Calendar Events</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Google Calendar is connected.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/upcoming?max_results=5</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing upcoming events and count.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing upcoming events and count.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.7 Get Calendar Events in Date Range

<table>
  <tr><td>Test Case ID</td><td>UT07</td></tr>
  <tr><td>Test Case Name</td><td>Get Calendar Events in Date Range</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Google Calendar is connected.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/range?start=2026-04-19T00:00:00%2B05:45&amp;end=2026-04-20T00:00:00%2B05:45</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing all events inside the requested date range.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing all events inside the requested date range.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.8 Get Single Calendar Event

<table>
  <tr><td>Test Case ID</td><td>UT08</td></tr>
  <tr><td>Test Case Name</td><td>Get Single Calendar Event by ID</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. A valid calendar event ID already exists.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/events/{event_id}</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing the requested calendar event details.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing the requested calendar event details.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.9 Create Calendar Event

<table>
  <tr><td>Test Case ID</td><td>UT09</td></tr>
  <tr><td>Test Case Name</td><td>Create Calendar Event</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Google Calendar write access is available.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to POST.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/events</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Send JSON body with <code>summary</code>, <code>start_time</code>, <code>end_time</code>, <code>description</code>, and <code>location</code>.<br>6. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing the newly created calendar event.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing the newly created calendar event.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.10 Update Calendar Event

<table>
  <tr><td>Test Case ID</td><td>UT10</td></tr>
  <tr><td>Test Case Name</td><td>Update Existing Calendar Event</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. A valid calendar event ID already exists.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to PUT.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/events/{event_id}</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Send JSON body with the fields to update.<br>6. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing the updated calendar event details.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing the updated calendar event details.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.11 Delete Calendar Event

<table>
  <tr><td>Test Case ID</td><td>UT11</td></tr>
  <tr><td>Test Case Name</td><td>Delete Existing Calendar Event</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. A valid calendar event ID already exists.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to DELETE.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/calendar/events/{event_id}</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with confirmation that the calendar event was deleted successfully.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with confirmation that the calendar event was deleted successfully.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.12 Get Gmail Inbox Messages

<table>
  <tr><td>Test Case ID</td><td>UT12</td></tr>
  <tr><td>Test Case Name</td><td>Get Gmail Inbox Messages</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Gmail access is connected with valid permissions.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/mail/inbox?max_results=10&amp;unread_only=true</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing inbox messages, sender, subject, snippet, and count.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing inbox messages, sender, subject, snippet, and count.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.13 Get Single Gmail Message

<table>
  <tr><td>Test Case ID</td><td>UT13</td></tr>
  <tr><td>Test Case Name</td><td>Get Gmail Message by ID</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. A valid Gmail message ID already exists.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/mail/message/{message_id}</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with JSON body containing full message metadata and extracted body text.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with JSON body containing full message metadata and extracted body text.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.14 Send Gmail Message

<table>
  <tr><td>Test Case ID</td><td>UT14</td></tr>
  <tr><td>Test Case Name</td><td>Send Gmail Message</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. User is logged in with a valid <code>session_token</code>. Gmail send permission is granted.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to POST.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/mail/send</code><br>4. Under Headers add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Send JSON body with <code>to</code>, <code>subject</code>, and <code>body</code>.<br>6. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with confirmation of sent email, recipient, subject, and message IDs.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with confirmation of sent email, recipient, subject, and message IDs.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.15 Get Weather Information

<table>
  <tr><td>Test Case ID</td><td>UT15</td></tr>
  <tr><td>Test Case Name</td><td>Get Weather Information</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Internet connection is available. Optional user location is stored in the profile.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/weather</code><br>4. Optionally add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with temperature, condition, location, minimum temperature, and maximum temperature.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with temperature, condition, location, minimum temperature, and maximum temperature.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.16 Get News Headlines

<table>
  <tr><td>Test Case ID</td><td>UT16</td></tr>
  <tr><td>Test Case Name</td><td>Get News Headlines</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Internet connection is available. Optional user interests are stored in the profile.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/news</code><br>4. Optionally add <code>X-Session-Token:&lt;session_token&gt;</code><br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with a JSON array of headline objects.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with a JSON array of headline objects.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.17 Get Country News (Personalized)

<table>
  <tr><td>Test Case ID</td><td>UT17</td></tr>
  <tr><td>Test Case Name</td><td>Get Country News (Personalized)</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Internet connection is available. Optional session token may be present.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/news?mode=personalized&amp;country=Nepal</code><br>4. Optionally add <code>X-Session-Token:&lt;session_token&gt;</code>.<br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with a JSON array of country-focused headline objects.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with a JSON array of country-focused headline objects.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>

## 4.1.18 Get Country News With Interest (Personalized)

<table>
  <tr><td>Test Case ID</td><td>UT18</td></tr>
  <tr><td>Test Case Name</td><td>Get Country News With Interest (Personalized)</td></tr>
  <tr><td>Preconditions</td><td>FastAPI server is running on localhost:8000. Internet connection is available. Optional session token may be present.</td></tr>
  <tr><td>Test Steps</td><td>1. Open Postman or Swagger UI.<br>2. Set method to GET.<br>3. Enter URL: <code>http://127.0.0.1:8000/api/news?mode=personalized&amp;country=Nepal&amp;interests=technology</code><br>4. Optionally add <code>X-Session-Token:&lt;session_token&gt;</code>.<br>5. Click Send.</td></tr>
  <tr><td>Expected Result</td><td>200 OK response with a JSON array of Nepal technology-focused headlines.</td></tr>
  <tr><td>Actual Result</td><td>200 OK response with a JSON array of Nepal technology-focused headlines.</td></tr>
  <tr><td>Status</td><td>Pass</td></tr>
</table>
