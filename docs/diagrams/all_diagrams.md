# AARVIS Mermaid Diagram Set

This file combines all Mermaid diagrams for the `final fixed fyp` smart mirror project.

## 1. Gantt Chart

```mermaid
gantt
    title AARVIS Smart Mirror FYP Gantt Chart
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d
    excludes    weekends

    section Phase 1 - Initiation
    Requirements and user analysis     :done, p1, 2024-11-02, 2024-11-18
    M1 Requirements complete           :milestone, m1, 2024-11-18, 0d

    section Phase 2 - Design
    Architecture and prototyping       :done, p2, 2024-11-18, 2024-12-06
    M2 Design complete                 :milestone, m2, 2024-12-06, 0d

    section Sprint 1
    Facial recognition authentication  :done, s1, 2024-12-07, 2024-12-18
    M3 Face recognition complete       :milestone, m3, 2024-12-18, 0d

    section Sprint 2
    Morning briefing and news          :done, s2, 2024-12-18, 2024-12-31
    M4 Briefings complete              :milestone, m4, 2024-12-31, 0d

    section Sprint 3
    Natural AI communication           :done, s3, 2025-01-01, 2025-01-14
    M5 AI communication ready          :milestone, m5, 2025-01-14, 0d

    section Sprint 4
    Hardware setup and validation      :done, s4, 2025-01-14, 2025-01-27
    M6 Hardware ready                  :milestone, m6, 2025-01-27, 0d

    section Sprint 5
    Voice controlled scheduling        :done, s5, 2025-01-27, 2025-02-08
    M7 Scheduling ready                :milestone, m7, 2025-02-08, 0d

    section Sprint 6
    Intelligent email assistant        :done, s6, 2025-02-08, 2025-02-20
    M8 Email assistant complete        :milestone, m8, 2025-02-20, 0d

    section Phase 4 - Testing
    Unit and integration testing       :done, t1, 2025-02-20, 2025-03-07
    M9 Testing complete                :milestone, m9, 2025-03-07, 0d

    section Phase 5 - Documentation
    Report, manual, handover           :done, d1, 2025-03-07, 2025-03-22
    M10 Documentation complete         :milestone, m10, 2025-03-22, 0d
    M11 Project complete               :milestone, m11, 2025-03-22, 0d
```

## 2. Milestone Timeline

```mermaid
timeline
    title AARVIS Smart Mirror Milestone Timeline
    November 2024 : M1 Requirements Complete
    December 2024 : M2 Design Complete
                  : M3 Facial Recognition Complete
                  : M4 Briefings Complete
    January 2025  : M5 AI Communication Ready
                  : M6 Hardware Ready
    February 2025 : M7 Scheduling Ready
                  : M8 Email Assistant Complete
    March 2025    : M9 Testing Complete
                  : M10 Documentation Complete
                  : M11 Project Complete
```

## 3. WBS

```mermaid
flowchart TD
    A["1.0 AARVIS Smart Mirror FYP"]
    A --> B["1.1 Project Initiation and Requirements"]
    B --> B1["1.1.1 Requirement identification"]
    B --> B2["1.1.2 User needs analysis"]
    B --> B3["1.1.3 Product backlog and sprint planning"]
    A --> C["1.2 Design and Prototyping"]
    C --> C1["1.2.1 Core architecture design"]
    C --> C2["1.2.2 Prototype subsystems"]
    C --> C3["1.2.3 Technical diagrams"]
    C --> C4["1.2.4 Feedback based design refinement"]
    A --> D["1.3 Sprint Based Development"]
    D --> D1["1.3.1 Sprint 1 - Face authentication"]
    D1 --> D11["Detection, embedding, face login"]
    D --> D2["1.3.2 Sprint 2 - Briefing and live widgets"]
    D2 --> D21["Weather, news, calendar, morning briefing"]
    D --> D3["1.3.3 Sprint 3 - Natural AI communication"]
    D3 --> D31["LangGraph, STT, TTS, WebSocket streaming"]
    D --> D4["1.3.4 Sprint 4 - Hardware setup"]
    D4 --> D41["Raspberry Pi, monitor, camera, network"]
    D --> D5["1.3.5 Sprint 5 - Voice scheduling"]
    D5 --> D51["Read, create, update, delete calendar events"]
    D --> D6["1.3.6 Sprint 6 - Email assistant"]
    D6 --> D61["Read, summarize, draft, send emails"]
    A --> E["1.4 Testing and Quality Assurance"]
    E --> E1["1.4.1 Unit testing"]
    E --> E2["1.4.2 Integration testing"]
    E --> E3["1.4.3 User feedback testing"]
    E --> E4["1.4.4 Bug fixing and refinement"]
    A --> F["1.5 Documentation and Handover"]
    F --> F1["1.5.1 Project report"]
    F --> F2["1.5.2 User manual"]
    F --> F3["1.5.3 FAQ and feature guide"]
    F --> F4["1.5.4 Final submission"]
```

## 4. Mind Map

```mermaid
mindmap
  root((AARVIS Smart Mirror))
    Users
      Manual login
      Google sign in
      Face login
      Admin management
    Frontend
      Mirror dashboard
      Login and register pages
      Face setup page
      Mobile pairing page
      Admin panel
    AI Assistant
      LangGraph workflow
      Gemini model
      Ollama support
      Conversation memory
      Tool calling
    Voice Interface
      Browser microphone capture
      Faster Whisper STT
      Kokoro TTS
      WebSocket streaming
      Voice activity detection
    Vision
      InsightFace detection
      Custom face backbone
      Face enrollment
      Similarity matching
      Presence cache
    Integrations
      Google OAuth
      Gmail API
      Google Calendar API
      Weather API
      News API
    Data Layer
      SQLite database
      Face database pickle
      Contacts CSV
      OAuth tokens
      Conversation history
    Deployment
      FastAPI server
      Browser client
      Raspberry Pi setup
      Camera microphone speaker
```

## 5. System Architecture

```mermaid
flowchart LR
    subgraph Clients["Client Layer"]
        U1["Mirror Browser UI"]
        U2["Mobile Browser for QR/OAuth"]
        U3["CLI Test Client"]
    end
    subgraph App["Application Layer - FastAPI"]
        R1["Jinja Pages and REST APIs"]
        R2["WebSocket Voice Channel"]
        R3["Session and Auth Manager"]
        R4["Admin and Pairing Routes"]
    end
    subgraph AI["AI and Processing Layer"]
        A1["LangGraph Agent"]
        A2["Tool Layer"]
        A3["Faster Whisper STT"]
        A4["Kokoro TTS"]
        A5["Face Recognition Service"]
    end
    subgraph Models["Models and Runtime"]
        M1["Gemini Chat Model"]
        M2["Ollama Local Models"]
        M3["InsightFace FaceAnalysis"]
        M4["Custom FaceEmbeddingModel"]
    end
    subgraph Data["Data Layer"]
        D1[("SQLite smart_mirror.db")]
        D2[("face_database.pkl")]
        D3[("contacts.csv")]
        D4[("Google credential files")]
    end
    subgraph External["External Services"]
        E1["Google OAuth"]
        E2["Google Calendar API"]
        E3["Gmail API"]
        E4["Weather API"]
        E5["News API"]
    end

    U1 --> R1
    U1 --> R2
    U2 --> R4
    U3 --> A1
    R1 --> R3
    R1 --> R4
    R2 --> A3
    R2 --> A1
    A1 --> A2
    A1 --> A4
    R1 --> A5
    A1 --> M1
    A1 --> M2
    A5 --> M3
    A5 --> M4
    R3 --> D1
    R4 --> D1
    A1 --> D1
    A2 --> D1
    A5 --> D2
    A2 --> D3
    R3 --> D4
    A2 --> E2
    A2 --> E3
    R3 --> E1
    R1 --> E4
    R1 --> E5
```

## 6. Use Case Diagram

```mermaid
flowchart LR
    Guest(("Guest User"))
    User(("Registered User"))
    Admin(("Admin"))
    Google(("Google"))
    subgraph System["AARVIS Smart Mirror System"]
        UC1(["Register account"])
        UC2(["Sign in with username and password"])
        UC3(["Sign in with Google"])
        UC4(["Enroll face"])
        UC5(["Face login"])
        UC6(["View personalized dashboard"])
        UC7(["Receive morning briefing"])
        UC8(["Talk to AARVIS"])
        UC9(["Check weather, news, and schedule"])
        UC10(["Create, update, or delete calendar events"])
        UC11(["Read and summarize emails"])
        UC12(["Draft and send emails"])
        UC13(["Manage users and face enrollments"])
    end

    Guest --> UC1
    Guest --> UC2
    Guest --> UC3
    Guest --> UC4
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    User --> UC8
    User --> UC9
    User --> UC10
    User --> UC11
    User --> UC12
    Admin --> UC13
    Google --> UC3
    Google --> UC10
    Google --> UC11
    Google --> UC12
    UC6 --> UC9
    UC8 --> UC10
    UC8 --> UC11
    UC8 --> UC12
```

## 7. Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Mirror Browser UI
    participant WS as FastAPI WebSocket
    participant STT as Faster Whisper
    participant Agent as LangGraph Agent
    participant Tools as Tool Layer
    participant APIs as Google/Weather/News APIs
    participant TTS as Kokoro TTS

    User->>UI: Speak a command
    UI->>WS: Send recorded audio
    WS->>STT: Transcribe audio bytes
    STT-->>WS: Transcript text
    WS-->>UI: Show transcript
    WS->>Agent: Invoke agent with user context and history
    alt Tool is needed
        Agent->>Tools: Call selected tool
        Tools->>APIs: Request data or perform action
        APIs-->>Tools: Response payload
        Tools-->>Agent: Tool result
    else No tool needed
        Agent-->>Agent: Generate direct response
    end
    Agent-->>WS: Stream response tokens
    loop Sentence by sentence
        WS->>TTS: Convert sentence to audio
        TTS-->>WS: WAV audio bytes
        WS-->>UI: Response chunk + TTS audio
    end
    UI-->>User: Display text and play spoken reply
```

## 8. Flow Chart

```mermaid
flowchart TD
    A([Start]) --> B["Open Smart Mirror Application"]
    B --> C{"Valid session token?"}
    C -- No --> D["Show login/register page"]
    D --> E{"Auth method selected?"}
    E -- Manual --> F["Submit username and password"]
    E -- Google --> G["Start Google OAuth and optional mobile pairing"]
    F --> H{"Authentication successful?"}
    G --> H
    H -- No --> D
    H -- Yes --> I{"Face enrolled?"}
    C -- Yes --> J["Load mirror dashboard"]
    I -- No --> K["Open face setup and enroll samples"]
    I -- Yes --> J
    K --> J
    J --> L{"Face cache valid?"}
    L -- No --> M["Capture frame and verify face"]
    M --> N{"Recognized user?"}
    N -- No --> O["Hide personalized content and retry"]
    O --> M
    N -- Yes --> P["Show personalized dashboard"]
    L -- Yes --> P
    P --> Q["Capture user text or audio input"]
    Q --> R{"Audio input?"}
    R -- Yes --> S["Transcribe with Faster Whisper"]
    R -- No --> T["Use typed text"]
    S --> U["Invoke LangGraph agent"]
    T --> U
    U --> V{"Tool required?"}
    V -- Yes --> W["Call calendar, email, weather, or news tool"]
    V -- No --> X["Generate direct response"]
    W --> X
    X --> Y["Stream response to UI"]
    Y --> Z["Convert sentences to speech with Kokoro"]
    Z --> AA["Play reply and update widgets"]
    AA --> AB{"Continue session?"}
    AB -- Yes --> Q
    AB -- No --> AC["Logout and clear session"]
    AC --> AD([End])
```

## 9. Entity Relationship Diagram

```mermaid
erDiagram
    USERS {
        int id PK
        string username UK
        string email UK
        string password_hash
        string full_name
        string location
        string interests
        string google_id UK
        string google_access_token
        string google_refresh_token
        string google_token_uri
        string google_client_id
        string google_client_secret
        string google_scopes
        string google_token_expiry
        datetime created_at
    }
    FACE_EMBEDDINGS {
        int id PK
        int user_id FK
        blob embedding
        string embedding_version
        string face_photo_path
        datetime enrolled_at
    }
    ATTENDANCE {
        int id PK
        int user_id FK
        datetime check_in_time
        float verification_score
        string method
    }
    CONVERSATION_HISTORY {
        int id PK
        int user_id FK
        string session_id
        string role
        string content
        string intent
        string agent_type
        string metadata
        datetime created_at
    }
    USERS ||--o{ FACE_EMBEDDINGS : has
    USERS ||--o{ ATTENDANCE : records
    USERS ||--o{ CONVERSATION_HISTORY : generates
```

## 10. Data Flow Diagram

```mermaid
flowchart LR
    U[User]
    A[Admin]
    G[Google Services]
    W[Weather API]
    N[News API]
    P1["1.0 Authentication and Session Management"]
    P2["2.0 Face Recognition and Enrollment"]
    P3["3.0 Voice Assistant Orchestration"]
    P4["4.0 Information and Action Services"]
    P5["5.0 Admin Management"]
    D1[("D1 Users and OAuth Tokens")]
    D2[("D2 Face Database")]
    D3[("D3 Conversation History")]
    D4[("D4 Contacts and Config Files")]

    U --> P1
    U --> P2
    U --> P3
    A --> P5
    P1 <--> D1
    P2 <--> D2
    P3 <--> D3
    P4 <--> D1
    P4 <--> D4
    P5 <--> D1
    P5 <--> D2
    P3 --> P4
    P4 --> G
    P4 --> W
    P4 --> N
    G --> P4
    W --> P4
    N --> P4
    P1 --> U
    P2 --> U
    P3 --> U
    P5 --> A
```

## 11. Wireframe

```mermaid
flowchart TB
    subgraph Screen["Smart Mirror Dashboard Wireframe"]
        subgraph Header["Header"]
            H1["Clock and Date"]
            H2["Weather Widget"]
        end
        subgraph Main["Main Interaction Area"]
            M1["Greeting Banner"]
            M2["Response Display"]
            M3["Voice Ring / Mic Button"]
            M4["Voice Status"]
        end
        subgraph Footer["Footer"]
            F1["News Panel"]
            F2["Spacer / Visual Balance"]
            F3["Schedule Panel"]
        end
    end
    B1["Background Face Verification"] -.-> Main
    B2["WebSocket Streaming"] -.-> M2
    B3["Auto Listening VAD"] -.-> M3
    B4["Periodic Weather, News, Calendar Refresh"] -.-> Header
    B4 -.-> Footer
```
