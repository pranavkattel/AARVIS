console.log('[DEBUG] Mirror.js loaded!');

let ws = null;
let isConnected = false;
let currentUser = null;
let faceVerified = false;
let faceCheckInterval = null;
let videoStream = null;
let videoElement = null;

// Face verification state
const FACE_CHECK_INTERVAL = 240000; // 4 minutes in milliseconds
const INITIAL_CHECK_DELAY = 5000; // 5 seconds initial delay

// Check authentication on load
async function checkAuth() {
    // Pick up token passed via URL (e.g. after Google OAuth redirect)
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');
    if (urlToken) {
        localStorage.setItem('sessionToken', urlToken);
        // Clean the token out of the URL without reloading
        const clean = window.location.pathname;
        window.history.replaceState({}, '', clean);
    }

    const sessionToken = localStorage.getItem('sessionToken');
    
    if (!sessionToken) {
        window.location.href = '/login';
        return false;
    }
    
    try {
        const response = await fetch('/api/user', {
            credentials: 'include'
        });
        
        if (response.ok) {
            currentUser = await response.json();
            console.log('[DEBUG] User loaded:', currentUser);
            // Force update greeting with user's name immediately
            updateGreeting(new Date().getHours());
            return true;
        } else {
            localStorage.removeItem('sessionToken');
            window.location.href = '/login';
            return false;
        }
    } catch (error) {
        console.error('[DEBUG] Auth check failed:', error);
        localStorage.removeItem('sessionToken');
        window.location.href = '/login';
        return false;
    }
}

function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const sessionToken = localStorage.getItem('sessionToken');
    const tokenQuery = sessionToken ? `?token=${encodeURIComponent(sessionToken)}` : '';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws${tokenQuery}`;
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected:', wsUrl);
        isConnected = true;
        updateStatus('Ready', 'ready');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('Connection Error', 'error');
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        isConnected = false;
        updateStatus('Disconnected', 'error');
        setTimeout(connectWebSocket, 3000);
    };
}

// Accumulator for streamed tokens
let _streamingText = '';

// ──── TTS Audio Playback Queue ────
// Queues base64 WAV chunks and plays them sequentially so sentences
// don't overlap and they play in order even if they arrive while another is playing.
const _ttsQueue = [];
let _ttsPlaying = false;

let _ttsIsPlaying = false;
let _systemBusy = false; // true while thinking/speaking — VAD paused

async function _playNextTTS() {
    if (_ttsPlaying || _ttsQueue.length === 0) return;
    _ttsPlaying = true;
    _ttsIsPlaying = true;

    const b64 = _ttsQueue.shift();
    try {
        const raw = atob(b64);
        const buf = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuf = await audioCtx.decodeAudioData(buf.buffer);
        const src = audioCtx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(audioCtx.destination);
        src.onended = () => {
            audioCtx.close();
            _ttsPlaying = false;
            if (_ttsQueue.length > 0) {
                _playNextTTS();
            } else {
                _ttsIsPlaying = false;
                _systemBusy = false;
                updateVoiceStatus(isAutoListening ? 'listening' : 'idle');
            }
        };
        src.start(0);
        updateVoiceStatus('speaking');
    } catch (e) {
        console.error('[TTS] Audio playback error:', e);
        _ttsPlaying = false;
        _ttsIsPlaying = false;
        _systemBusy = false;
        _playNextTTS(); // skip bad chunk
    }
}

function queueTTSAudio(base64Data) {
    _ttsQueue.push(base64Data);
    _playNextTTS();
}

function handleMessage(data) {
    switch(data.type) {
        case 'error':
            updateResponse(data.text || 'Connection/authentication error. Please log in again.');
            updateVoiceStatus('idle');
            break;
        case 'response_chunk':
            // Token-by-token streaming from the LLM
            if (data.first) {
                _streamingText = '';  // reset on first token
            }
            _streamingText += data.token;
            updateResponse(_streamingText);
            break;
        case 'response':
            // Final complete response — overwrite streamed text
            _streamingText = '';
            updateResponse(data.text);
            break;
        case 'tts_audio':
            // Server-generated TTS audio (base64 WAV) — play in browser
            if (data.data) {
                console.log('[TTS] Received audio chunk, queueing playback');
                queueTTSAudio(data.data);
            }
            break;
        case 'transcript':
            // Show what the server transcribed from voice
            console.log('[VOICE] Server transcript:', data.text);
            updateResponse('🎤 ' + data.text);
            break;
        case 'status':
            updateStatus(data.text, data.state);
            break;
        case 'voice_state':
            if (data.state === 'thinking') {
                _systemBusy = true;
                updateVoiceStatus('thinking');
            } else if (data.state === 'speaking') {
                _systemBusy = true;
                updateVoiceStatus('speaking');
            } else if (data.state === 'idle') {
                // Only resume listening if TTS queue is also empty
                if (!_ttsIsPlaying && _ttsQueue.length === 0) {
                    _systemBusy = false;
                    updateVoiceStatus(isAutoListening ? 'listening' : 'idle');
                }
            }
            break;
        case 'animation_start':
            startVoiceAnimation();
            break;
        case 'animation_stop':
            stopVoiceAnimation();
            break;
        case 'weather':
            updateWeather(data.data);
            break;
        case 'news':
            updateNews(data.data);
            break;
        case 'logout':
            // Server says to log out (e.g. user said "bye bye")
            _performLogout();
            break;
    }
}

function sendMessage(text) {
    if (isConnected && text.trim()) {
        ws.send(JSON.stringify({
            type: 'message',
            text: text
        }));
        updateVoiceStatus('thinking');
    }
}

// ──── Logout (triggered by server after "bye bye") ────
async function _performLogout() {
    // Wait for any TTS goodbye to finish playing before redirecting
    const waitForTTS = () => new Promise(resolve => {
        const check = () => {
            if (!_ttsPlaying && _ttsQueue.length === 0) return resolve();
            setTimeout(check, 200);
        };
        check();
    });
    await waitForTTS();

    // Stop mic & VAD
    isAutoListening = false;
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (vadAudioCtx) { vadAudioCtx.close().catch(() => {}); }

    // Call server logout endpoint, clear local storage, redirect
    try { await fetch('/api/logout', { method: 'POST', credentials: 'include' }); } catch (_) {}
    localStorage.removeItem('sessionToken');
    document.cookie = 'session_token=; Max-Age=0; path=/;';
    window.location.href = '/login';
}

function updateTime() {
    const now = new Date();

    // Time: 23:31
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const timeEl = document.getElementById('timeDisplay');
    if (timeEl) timeEl.textContent = `${hours}:${minutes}`;

    // Date: Wednesday February 18th
    const dayStr   = now.toLocaleDateString('en-US', { weekday: 'long' });
    const monthStr = now.toLocaleDateString('en-US', { month: 'long' });
    const day      = now.getDate();
    const dateEl   = document.getElementById('dateDisplay');
    if (dateEl) dateEl.textContent = `${dayStr} ${monthStr} ${day}${getDaySuffix(day)}`;

    updateGreeting(now.getHours());
}

function getDaySuffix(day) {
    if (day >= 11 && day <= 13) return 'th';
    switch (day % 10) {
        case 1: return 'st';
        case 2: return 'nd';
        case 3: return 'rd';
        default: return 'th';
    }
}

function updateGreeting(hour) {
    const greetingEl = document.getElementById('greeting');
    if (!greetingEl) return;

    let greeting = 'Good Morning';
    if (hour >= 12 && hour < 17)      greeting = 'Good Afternoon';
    else if (hour >= 17 && hour < 22) greeting = 'Good Evening';
    else if (hour >= 22 || hour < 5)  greeting = 'Good Night';

    console.log(`[DEBUG] Updating greeting: ${greeting}`);
    greetingEl.textContent = greeting;
}

function updateStatus(text, state) {
    // legacy — kept for WebSocket status messages
    console.log(`[STATUS] ${text} (${state})`);
}

function updateResponse(text) {
    const el = document.getElementById('responseDisplay');
    if (el) {
        el.textContent = text;
        el.classList.add('visible');
        // Auto-hide after 15 seconds
        clearTimeout(window._responseHideTimer);
        window._responseHideTimer = setTimeout(() => el.classList.remove('visible'), 15000);
    }
}

function startVoiceAnimation() {
    _systemBusy = true;
    updateVoiceStatus('thinking');
}

function stopVoiceAnimation() {
    // Don't resume listening here — wait for TTS to finish
    if (!_ttsIsPlaying && _ttsQueue.length === 0) {
        _systemBusy = false;
        updateVoiceStatus(isAutoListening ? 'listening' : 'idle');
    }
}

// ──── Voice Status Manager ────
function updateVoiceStatus(state) {
    const wrapper = document.getElementById('micWrapper');
    const statusEl = document.getElementById('voiceStatus');
    if (!wrapper || !statusEl) return;

    // Clear all state classes
    wrapper.classList.remove('auto-listening', 'speech-detected', 'listening', 'thinking-state');
    statusEl.classList.remove('listening', 'thinking');
    statusEl.textContent = '';

    switch (state) {
        case 'listening':
            wrapper.classList.add('auto-listening');
            statusEl.classList.add('listening');
            statusEl.textContent = 'Listening…';
            break;
        case 'speech-detected':
            wrapper.classList.add('auto-listening', 'speech-detected');
            statusEl.classList.add('listening');
            statusEl.textContent = 'Listening…';
            break;
        case 'thinking':
            wrapper.classList.add('thinking-state');
            statusEl.classList.add('thinking');
            statusEl.textContent = 'Thinking…';
            break;
        case 'speaking':
            statusEl.textContent = 'Speaking…';
            break;
        case 'needs-permission':
            statusEl.textContent = 'Click anywhere to enable mic';
            break;
        default:
            // idle — only go back to listening if system is not busy
            if (isAutoListening && !_systemBusy) {
                wrapper.classList.add('auto-listening');
                statusEl.classList.add('listening');
                statusEl.textContent = 'Listening…';
            } else {
                statusEl.textContent = '';
            }
    }
}

// ──── Auto-Listening with Voice Activity Detection (VAD) ────
let micStream = null;
let vadAudioCtx = null;
let vadAnalyser = null;
let vadMicSource = null;
let isAutoListening = false;
let speechDetected = false;
let vadSilenceTimer = null;
let speechStartTime = 0;
let autoRecorder = null;
let autoChunks = [];
let _vadAnimFrame = null;

const VAD_THRESHOLD = 18;        // frequency-bin average to count as speech
const SILENCE_TIMEOUT_MS = 1800; // ms of silence before ending capture
const MIN_SPEECH_MS = 500;       // minimum speech duration to send
const MAX_SPEECH_MS = 20000;     // safety cap on recording length

// Preferred MIME type for MediaRecorder
function _pickMime() {
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus';
    if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus'))  return 'audio/ogg;codecs=opus';
    if (MediaRecorder.isTypeSupported('audio/mp4'))              return 'audio/mp4';
    return 'audio/webm';
}

// ── Initialise auto-listening (called once from init) ──
async function initAutoListening() {
    const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
    if (!window.isSecureContext && !isLocalhost) {
        console.warn('[VOICE] Not a secure context – mic unavailable');
        updateResponse('Microphone requires HTTPS or localhost.');
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('[VOICE] getUserMedia not available');
        return;
    }

    try {
        await _startAutoListening();
    } catch (err) {
        if (err.name === 'NotAllowedError') {
            // Browser blocked auto-play mic – wait for a click anywhere
            console.warn('[VOICE] Mic permission needs user gesture, waiting for click…');
            updateVoiceStatus('needs-permission');
            document.addEventListener('click', async function _resume() {
                document.removeEventListener('click', _resume);
                try { await _startAutoListening(); } catch (e) {
                    console.error('[VOICE] Still cannot access mic:', e);
                    updateResponse('Microphone access denied.');
                }
            }, { once: true });
        } else {
            console.error('[VOICE] Mic init error:', err);
            updateResponse('Mic error: ' + err.message);
        }
    }
}

async function _startAutoListening() {
    // Acquire mic stream (once)
    if (!micStream) {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 }
        });
        console.log('[VOICE] Mic stream acquired');
    }

    // Create persistent AudioContext + Analyser for VAD
    vadAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    vadAnalyser = vadAudioCtx.createAnalyser();
    vadAnalyser.fftSize = 256;
    vadAnalyser.smoothingTimeConstant = 0.85;
    vadMicSource = vadAudioCtx.createMediaStreamSource(micStream);
    vadMicSource.connect(vadAnalyser);

    isAutoListening = true;
    updateVoiceStatus('listening');
    console.log('[VOICE] Auto-listening active (VAD threshold=' + VAD_THRESHOLD + ')');

    // Kick off the VAD render loop
    _vadLoop();
}

// ── Core VAD render loop — runs every animation frame ──
function _vadLoop() {
    if (!isAutoListening) return;

    const data = new Uint8Array(vadAnalyser.frequencyBinCount);
    vadAnalyser.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const level = sum / data.length;

    // Update audio-bar visualisation
    _updateAudioBars(data);

    // Skip VAD logic while system is busy (thinking, speaking, processing)
    if (_systemBusy) {
        _vadAnimFrame = requestAnimationFrame(_vadLoop);
        return;
    }

    if (level > VAD_THRESHOLD) {
        if (!speechDetected) {
            // Speech just started
            speechDetected = true;
            speechStartTime = Date.now();
            _startSpeechCapture();
        }
        // Reset the silence countdown
        clearTimeout(vadSilenceTimer);
        vadSilenceTimer = setTimeout(() => {
            if (!speechDetected) return;
            const dur = Date.now() - speechStartTime;
            if (dur >= MIN_SPEECH_MS) {
                _finishSpeechCapture();
            } else {
                _cancelSpeechCapture();
            }
        }, SILENCE_TIMEOUT_MS);
    }

    // Safety cap: force-stop very long recordings
    if (speechDetected && (Date.now() - speechStartTime) > MAX_SPEECH_MS) {
        console.log('[VOICE] Max speech duration reached, finishing capture');
        clearTimeout(vadSilenceTimer);
        _finishSpeechCapture();
    }

    _vadAnimFrame = requestAnimationFrame(_vadLoop);
}

// ── Update the 5 audio-bar elements based on frequency data ──
function _updateAudioBars(freqData) {
    const bars = document.querySelectorAll('#audioBars span');
    if (bars.length === 0) return;

    // If not in a listening state visually, fallback to gentle idle sine
    const wrapper = document.getElementById('micWrapper');
    const isVisuallyListening = wrapper && wrapper.classList.contains('auto-listening');

    if (!isVisuallyListening) {
        bars.forEach(b => { b.style.height = '4px'; });
        return;
    }

    if (speechDetected) {
        // Use real frequency data – sample 5 bands
        const bandSize = Math.floor(freqData.length / bars.length);
        bars.forEach((bar, i) => {
            let bandSum = 0;
            for (let j = i * bandSize; j < (i + 1) * bandSize; j++) bandSum += freqData[j];
            const avg = bandSum / bandSize;
            const h = Math.max(4, (avg / 255) * 32);
            bar.style.height = h + 'px';
        });
    } else {
        // Gentle idle sine animation to show "I'm alive"
        const t = Date.now() / 1000;
        bars.forEach((bar, i) => {
            const h = 4 + Math.sin(t * 1.8 + i * 1.1) * 5;
            bar.style.height = Math.max(3, h) + 'px';
        });
    }
}

// ── Start capturing speech into a MediaRecorder ──
function _startSpeechCapture() {
    autoChunks = [];
    const mime = _pickMime();
    autoRecorder = new MediaRecorder(micStream, { mimeType: mime });
    autoRecorder.ondataavailable = (e) => { if (e.data.size > 0) autoChunks.push(e.data); };
    autoRecorder.start(250);
    updateVoiceStatus('speech-detected');
    console.log('[VOICE] Speech capture started');
}

// ── Finish capture & send audio to server ──
function _finishSpeechCapture() {
    speechDetected = false;
    clearTimeout(vadSilenceTimer);

    if (!autoRecorder || autoRecorder.state !== 'recording') {
        updateVoiceStatus('listening');
        return;
    }

    autoRecorder.onstop = async () => {
        if (autoChunks.length === 0) { updateVoiceStatus('listening'); return; }

        const blob = new Blob(autoChunks, { type: autoRecorder.mimeType });
        autoChunks = [];
        if (blob.size < 1000) { updateVoiceStatus('listening'); return; }

        _systemBusy = true;
        updateVoiceStatus('thinking');

        // Base64-encode and send over WebSocket
        const reader = new FileReader();
        reader.onloadend = () => {
            if (reader.readyState !== FileReader.DONE) return;
            const base64 = reader.result.split(',')[1];
            if (base64 && base64.length > 100 && isConnected && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'audio', data: base64 }));
                console.log('[VOICE] Audio sent to server (' + base64.length + ' chars)');
            } else {
                updateVoiceStatus('listening');
            }
        };
        reader.onerror = () => { updateVoiceStatus('listening'); };
        reader.readAsDataURL(blob);
    };
    autoRecorder.stop();
    console.log('[VOICE] Speech capture finished, sending…');
}

// ── Discard capture (too short) ──
function _cancelSpeechCapture() {
    speechDetected = false;
    clearTimeout(vadSilenceTimer);
    if (autoRecorder && autoRecorder.state === 'recording') {
        autoRecorder.onstop = () => {};
        autoRecorder.stop();
    }
    autoChunks = [];
    updateVoiceStatus('listening');
}

function updateWeather(data) {
    console.log('[DEBUG] Updating weather display:', data);
    console.log('[DEBUG] temp_min:', data.temp_min, 'temp_max:', data.temp_max);
    
    const weatherTemp = document.getElementById('weatherTemp');
    const weatherCondition = document.getElementById('weatherCondition');
    const weatherLocation = document.getElementById('weatherLocation');
    const weatherLow = document.getElementById('weatherLow');
    const weatherHigh = document.getElementById('weatherHigh');
    const weatherIcon = document.getElementById('weatherIcon');
    
    // Add smooth transition class
    const weatherWidget = document.getElementById('weatherWidget');
    if (weatherWidget) {
        weatherWidget.style.opacity = '0';
        setTimeout(() => {
            weatherWidget.style.transition = 'opacity 0.5s ease';
            weatherWidget.style.opacity = '1';
        }, 50);
    }
    
    if (weatherTemp && data.temp !== undefined) {
        weatherTemp.textContent = `${Math.round(data.temp)}°C`;
    }
    
    if (weatherCondition && data.condition) {
        weatherCondition.textContent = data.condition;
    }
    
    if (weatherLocation && data.location) {
        weatherLocation.textContent = data.location;
    }
    
    if (weatherLow) {
        console.log('[DEBUG] Setting weatherLow element');
        if (data.temp_min !== undefined && data.temp_min !== null) {
            weatherLow.textContent = `${Math.round(data.temp_min)}°C`;
            console.log('[DEBUG] Set weatherLow to:', weatherLow.textContent);
        } else {
            console.log('[DEBUG] temp_min is undefined or null');
        }
    } else {
        console.log('[DEBUG] weatherLow element not found!');
    }
    
    if (weatherHigh) {
        console.log('[DEBUG] Setting weatherHigh element');
        if (data.temp_max !== undefined && data.temp_max !== null) {
            weatherHigh.textContent = `${Math.round(data.temp_max)}°C`;
            console.log('[DEBUG] Set weatherHigh to:', weatherHigh.textContent);
        } else {
            console.log('[DEBUG] temp_max is undefined or null');
        }
    } else {
        console.log('[DEBUG] weatherHigh element not found!');
    }
    
    // Update weather icon based on condition with animation
    if (weatherIcon && data.condition) {
        const condition = data.condition.toLowerCase();
        weatherIcon.style.transform = 'scale(0)';
        setTimeout(() => {
            weatherIcon.style.transition = 'transform 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
            weatherIcon.style.transform = 'scale(1)';
            
            if (condition.includes('sun') || condition.includes('clear')) {
                weatherIcon.textContent = '☀️';
            } else if (condition.includes('cloud')) {
                weatherIcon.textContent = '☁️';
            } else if (condition.includes('rain')) {
                weatherIcon.textContent = '🌧️';
            } else if (condition.includes('snow')) {
                weatherIcon.textContent = '❄️';
            } else if (condition.includes('storm') || condition.includes('thunder')) {
                weatherIcon.textContent = '⛈️';
            } else {
                weatherIcon.textContent = '🌤️';
            }
        }, 100);
    }
}

function updateNews(articles) {
    console.log('[DEBUG] updateNews called with:', articles);
    const newsWidget = document.getElementById('newsWidget');
    
    // Fade out before updating
    if (newsWidget) {
        newsWidget.style.transition = 'opacity 0.3s ease';
        newsWidget.style.opacity = '0';
        
        setTimeout(() => {
            const nums = ['01.','02.','03.','04.','05.'];
            if (articles && articles.length > 0) {
                const items = articles.slice(0, 4).map((article, i) => {
                    const title = article.title.length > 160
                        ? article.title.substring(0, 157) + '…'
                        : article.title;
                    return `<li><span class="sm-news-num">${nums[i]}</span>${title}</li>`;
                }).join('');
                newsWidget.innerHTML =
                    `<div class="sm-section-label">Latest News</div><ul class="sm-news-list">${items}</ul>`;
            } else {
                newsWidget.innerHTML =
                    '<div class="sm-section-label">Latest News</div><ul class="sm-news-list"><li>No news available</li></ul>';
            }
            setTimeout(() => { newsWidget.style.opacity = '1'; }, 50);
        }, 300);
    }
}

function fetchWeather() {
    fetch('/api/weather', {
        credentials: 'include'
    })
        .then(response => response.json())
        .then(data => updateWeather(data))
        .catch(error => console.error('Error fetching weather:', error));
}

function fetchNews() {
    console.log('[DEBUG] Fetching news from API...');
    fetch('/api/news', {
        credentials: 'include'
    })
        .then(response => {
            console.log('[DEBUG] News API response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('[DEBUG] News API response data:', data);
            updateNews(data);
        })
        .catch(error => {
            console.error('[DEBUG] Error fetching news:', error);
            const newsWidget = document.getElementById('newsWidget');
            if (newsWidget) {
                newsWidget.innerHTML = '<div class="news-header">Latest News</div><div class="news-item"><div class="news-title">Unable to load news</div></div>';
            }
        });
}

function updateCalendar(data) {
    console.log('[DEBUG] updateCalendar called with data:', data);
    console.log('[DEBUG] Looking for timeline element...');
    const checkElement = document.getElementById('timelineEvents');
    console.log('[DEBUG] Timeline element found:', checkElement);
    
    if (data && data.events && data.events.length > 0) {
        console.log(`[DEBUG] Displaying ${data.events.length} events on timeline`);
        
        // Log events for debugging
        data.events.forEach(event => {
            console.log(`[DEBUG] Event: ${event.title} at ${event.time}, status: ${event.status}`);
        });
        
        // Update timeline with events
        updateTimeline(data.events);
    } else {
        console.log('[DEBUG] No events to display');
        // Clear timeline
        const timelineEvents = document.getElementById('timelineEvents');
        if (timelineEvents) {
            timelineEvents.innerHTML = '';
        }
    }
}

const DEFAULT_TIMELINE_START = 7;
const DEFAULT_TIMELINE_END = 20;

function formatHourLabel(hourValue) {
    const normalizedHour = ((hourValue % 24) + 24) % 24;
    const displayHour = normalizedHour % 12 === 0 ? 12 : normalizedHour % 12;
    return `${String(displayHour).padStart(2, '0')}:00`;
}

function renderTimelineHourMarks(timelineStart, timelineEnd) {
    const timelineHoursEl = document.querySelector('.timeline-hours');
    if (!timelineHoursEl) return;

    const marks = [];
    for (let hour = Math.floor(timelineStart); hour <= Math.ceil(timelineEnd); hour += 1) {
        marks.push(`<div class="hour-mark">${formatHourLabel(hour)}</div>`);
    }
    timelineHoursEl.innerHTML = marks.join('');
}

function buildTimelineWindow(events) {
    const timedRanges = (events || [])
        .filter((event) => event.status !== 'all-day')
        .map((event) => {
            const startHour = Number(event.startHour);
            const endHourRaw = Number(event.endHour);
            if (!Number.isFinite(startHour) || !Number.isFinite(endHourRaw)) return null;

            const endHour = endHourRaw <= startHour ? endHourRaw + 24 : endHourRaw;
            return { startHour, endHour };
        })
        .filter(Boolean);

    if (timedRanges.length === 0) {
        return {
            timelineStart: DEFAULT_TIMELINE_START,
            timelineEnd: DEFAULT_TIMELINE_END
        };
    }

    const minHour = Math.floor(Math.min(...timedRanges.map((item) => item.startHour)));
    const maxHour = Math.ceil(Math.max(...timedRanges.map((item) => item.endHour)));

    const timelineStart = Math.min(DEFAULT_TIMELINE_START, minHour);
    const timelineEnd = Math.max(DEFAULT_TIMELINE_END, maxHour);

    return {
        timelineStart,
        timelineEnd
    };
}

function layoutOverlappingEvents(events) {
    if (!events || events.length === 0) return [];

    const sorted = [...events].sort((a, b) => {
        if (a.clampedStart !== b.clampedStart) return a.clampedStart - b.clampedStart;
        return a.clampedEnd - b.clampedEnd;
    });

    const groups = [];
    let currentGroup = [];
    let currentGroupEnd = -Infinity;

    sorted.forEach((event) => {
        if (currentGroup.length === 0 || event.clampedStart < currentGroupEnd) {
            currentGroup.push(event);
            currentGroupEnd = Math.max(currentGroupEnd, event.clampedEnd);
            return;
        }

        groups.push(currentGroup);
        currentGroup = [event];
        currentGroupEnd = event.clampedEnd;
    });

    if (currentGroup.length > 0) {
        groups.push(currentGroup);
    }

    const laidOut = [];

    groups.forEach((group) => {
        const columnEndTimes = [];
        const itemsWithColumns = group.map((event) => {
            let columnIndex = columnEndTimes.findIndex((endTime) => endTime <= event.clampedStart);
            if (columnIndex === -1) {
                columnIndex = columnEndTimes.length;
                columnEndTimes.push(event.clampedEnd);
            } else {
                columnEndTimes[columnIndex] = event.clampedEnd;
            }

            return {
                ...event,
                columnIndex
            };
        });

        const columnCount = Math.max(columnEndTimes.length, 1);
        itemsWithColumns.forEach((item) => {
            laidOut.push({
                ...item,
                columnCount
            });
        });
    });

    return laidOut;
}

function updateTimeline(events) {
    console.log('[DEBUG] updateTimeline called with events:', events);

    const timelineEvents = document.getElementById('timelineEvents');
    if (!timelineEvents) {
        console.error('[DEBUG] Timeline element not found!');
        return;
    }

    // ── NEW: render as simple schedule list (no positioned timeline) ──
    if (!events || events.length === 0) {
        timelineEvents.innerHTML = '';
        return;
    }

    // Sort by startHour
    const sorted = [...events]
        .filter(e => e.status !== 'all-day')
        .sort((a, b) => Number(a.startHour) - Number(b.startHour));

    const allDay = events.filter(e => e.status === 'all-day');

    const renderItem = (ev) => {
        const timeRange = (ev.time && ev.endTime)
            ? `${ev.time} — ${ev.endTime}`
            : (ev.time || 'All Day');
        return `
            <div class="schedule-item">
                <span class="schedule-time">${timeRange}</span>
                <div class="schedule-card">
                    <span class="schedule-card-title">${ev.title || 'Untitled'}</span>
                </div>
            </div>`;
    };

    timelineEvents.style.opacity = '0';
    timelineEvents.innerHTML = [...sorted, ...allDay].map(renderItem).join('');
    requestAnimationFrame(() => {
        timelineEvents.style.transition = 'opacity 0.4s ease';
        timelineEvents.style.opacity = '1';
    });
}

function fetchCalendar() {
    console.log('[DEBUG] Fetching calendar from API...');
    fetch('/api/calendar', {
        credentials: 'include'
    })
        .then(response => {
            console.log('[DEBUG] Calendar API response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('[DEBUG] Calendar API response data:', data);
            updateCalendar(data);
        })
        .catch(error => {
            console.error('[DEBUG] Error fetching calendar:', error);
            // Clear timeline on error
            const timelineEvents = document.getElementById('timelineEvents');
            if (timelineEvents) {
                timelineEvents.innerHTML = '';
            }
        });
}

// Voice input via SpeechRecognition (replaces old sendBtn/userInput)
// Initialized in init() after WebSocket connects

// Text input fallback — send on Enter
const userInput = document.getElementById('userInput');
if (userInput) {
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            sendMessage(e.target.value.trim());
            e.target.value = '';
        }
    });
}

// Create animated raindrops (no-op in new UI – container removed)
function createRaindrops() {
    const container = document.getElementById('raindropsContainer');
    if (!container) return;
    
    for (let i = 0; i < dropCount; i++) {
        const drop = document.createElement('div');
        drop.style.position = 'absolute';
        drop.style.width = Math.random() * 3 + 1 + 'px';
        drop.style.height = Math.random() * 3 + 1 + 'px';
        drop.style.borderRadius = '50%';
        drop.style.background = `rgba(255, 255, 255, ${Math.random() * 0.3 + 0.1})`;
        drop.style.left = Math.random() * 100 + '%';
        drop.style.top = Math.random() * 100 + '%';
        drop.style.boxShadow = `0 0 ${Math.random() * 8 + 2}px rgba(255, 255, 255, 0.3)`;
        container.appendChild(drop);
    }
}

// Initialize everything
async function init() {
    console.log('[DEBUG] Starting initialization...');
    
    // First check authentication and load user
    const authSuccess = await checkAuth();
    
    if (!authSuccess) {
        console.log('[DEBUG] Auth failed, redirecting to login');
        return; // Will redirect to login
    }
    
    console.log('[DEBUG] Auth successful, user:', currentUser);
    
    // Force immediate greeting update with loaded user data
    const now = new Date();
    updateGreeting(now.getHours());
    console.log('[DEBUG] Initial greeting updated');
    
    // Now that user is loaded, start everything else
    updateTime();
    setInterval(updateTime, 1000);
    
    connectWebSocket();
    createRaindrops();
    initAutoListening();
    
    // Initialize face verification
    initFaceVerification();
    
    console.log('[DEBUG] Initializing API fetches...');
    setTimeout(() => {
        console.log('[DEBUG] Fetching weather...');
        fetchWeather();
    }, 1000);
    setTimeout(() => {
        console.log('[DEBUG] Fetching news...');
        fetchNews();
    }, 1500);
    setTimeout(() => {
        console.log('[DEBUG] Fetching calendar...');
        fetchCalendar();
    }, 2000);
    
    setInterval(fetchWeather, 600000);
    setInterval(fetchNews, 300000);
    setInterval(fetchCalendar, 300000);
    console.log('[DEBUG] Periodic updates scheduled');
}

// Face Verification Functions
async function initFaceVerification() {
    console.log('[DEBUG] Initializing face verification...');
    
    // Check if user was recently detected (cache check)
    try {
        const cacheResponse = await fetch('/api/face/check-cache', {
            credentials: 'include'
        });
        const cacheData = await cacheResponse.json();
        
        if (cacheData.cached) {
            console.log(`[DEBUG] User detected in cache: ${cacheData.username}`);
            faceVerified = true;
            showPersonalizedContent();
            
            // Schedule next check based on remaining time
            const nextCheck = Math.max(cacheData.seconds_remaining * 1000, 30000);
            setTimeout(performFaceCheck, nextCheck);
            return;
        }
    } catch (error) {
        console.log('[DEBUG] Cache check failed:', error);
    }
    
    // No cache, start face verification after initial delay
    setTimeout(performFaceCheck, INITIAL_CHECK_DELAY);
}

async function performFaceCheck() {
    console.log('[DEBUG] Performing face verification check...');
    
    try {
        // Create hidden video element for face capture
        if (!videoElement) {
            videoElement = document.createElement('video');
            videoElement.width = 640;
            videoElement.height = 480;
            videoElement.style.display = 'none';
            document.body.appendChild(videoElement);
        }
        
        // Get camera access
        if (!videoStream) {
            videoStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            videoElement.srcObject = videoStream;
            await videoElement.play();
        }
        
        // Wait a moment for camera to stabilize
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        canvas.getContext('2d').drawImage(videoElement, 0, 0, 640, 480);
        const imageData = canvas.toDataURL('image/jpeg', 0.7);
        
        // Send for verification
        const response = await fetch('/api/face/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        if (data.detected) {
            console.log(`[DEBUG] Face verified: ${data.username} (${data.confidence}% confidence)`);
            faceVerified = true;
            showPersonalizedContent();
            
            // Schedule next check after cache duration
            setTimeout(performFaceCheck, FACE_CHECK_INTERVAL);
        } else {
            console.log('[DEBUG] No face detected or unknown user');
            faceVerified = false;
            hidePersonalizedContent();
            
            // Retry sooner if no face detected
            setTimeout(performFaceCheck, 30000); // Check again in 30 seconds
        }
        
    } catch (error) {
        console.error('[DEBUG] Face verification error:', error);
        // Retry after error
        setTimeout(performFaceCheck, 60000); // Check again in 1 minute
    }
}

function showPersonalizedContent() {
    console.log('[DEBUG] Showing personalized content');
    
    // Trigger data fetches
    fetchWeather();
    fetchNews();
    fetchCalendar();
    
    // Show UI elements (they might be hidden by default)
    const weatherWidget = document.getElementById('weatherWidget');
    const newsWidget = document.getElementById('newsWidget');
    const timelineContainer = document.getElementById('timelineContainer');
    
    if (weatherWidget) weatherWidget.style.opacity = '1';
    if (newsWidget) newsWidget.style.opacity = '1';
    if (timelineContainer) timelineContainer.style.opacity = '1';
}

function hidePersonalizedContent() {
    console.log('[DEBUG] Hiding personalized content');
    
    // Optionally fade out or hide personal content
    const weatherWidget = document.getElementById('weatherWidget');
    const newsWidget = document.getElementById('newsWidget');
    const timelineContainer = document.getElementById('timelineContainer');
    
    if (weatherWidget) {
        document.getElementById('weatherCondition').textContent = 'Waiting for user...';
        document.getElementById('weatherTemp').textContent = '--°C';
    }
    if (newsWidget) {
        newsWidget.innerHTML = '<div class="news-header">Latest News</div><div class="news-item"><div class="news-title">Please stand in front of mirror</div></div>';
    }
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

