(function() {
    'use strict';

    let isRunning = false;
    let currentMode = 'webcam';
    let selectedFile = null;
    let eventSource = null;

    const elements = {
        modeRadios: document.querySelectorAll('input[name="mode"]'),
        videoFeed: document.getElementById('videoFeed'),
        placeholder: document.getElementById('placeholder'),
        btnStart: document.getElementById('btnStart'),
        uploadSection: document.getElementById('uploadSection'),
        btnUpload: document.getElementById('btnUpload'),
        videoInput: document.getElementById('videoInput'),
        fileName: document.getElementById('fileName'),
        faceList: document.getElementById('faceList'),
        totalFaces: document.getElementById('totalFaces'),
        forwardFaces: document.getElementById('forwardFaces'),
        attentionRate: document.getElementById('attentionRate')
    };

    function init() {
        elements.modeRadios.forEach(radio => {
            radio.addEventListener('change', handleModeChange);
        });

        elements.btnStart.addEventListener('click', handleStartStop);
        elements.btnUpload.addEventListener('click', () => elements.videoInput.click());
        elements.videoInput.addEventListener('change', handleFileSelect);
    }

    function handleModeChange(e) {
        currentMode = e.target.value;
        
        if (currentMode === 'upload') {
            elements.uploadSection.style.display = 'flex';
        } else {
            elements.uploadSection.style.display = 'none';
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            elements.fileName.textContent = file.name;
        }
    }

    async function handleStartStop() {
        if (isRunning) {
            await stopProcessing();
        } else {
            await startProcessing();
        }
    }

    async function startProcessing() {
        if (currentMode === 'webcam') {
            try {
                const response = await fetch('/webcam/start', { method: 'POST' });
                if (response.ok) {
                    beginProcessing();
                }
            } catch (err) {
                console.error('Failed to start webcam:', err);
            }
        } else {
            if (!selectedFile) {
                alert('Please select a video file');
                return;
            }

            const formData = new FormData();
            formData.append('video', selectedFile);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                if (response.ok) {
                    beginProcessing();
                }
            } catch (err) {
                console.error('Failed to upload video:', err);
            }
        }
    }

    function beginProcessing() {
        isRunning = true;
        elements.btnStart.textContent = 'Stop';
        elements.btnStart.classList.remove('btn-primary');
        elements.btnStart.classList.add('btn-danger');
        elements.videoFeed.style.display = 'block';
        elements.placeholder.style.display = 'none';

        connectToFeed();
    }

    async function stopProcessing() {
        try {
            await fetch('/webcam/stop', { method: 'POST' });
        } catch (err) {
            console.error('Failed to stop:', err);
        }

        isRunning = false;
        elements.btnStart.textContent = 'Start';
        elements.btnStart.classList.remove('btn-danger');
        elements.btnStart.classList.add('btn-primary');
        elements.videoFeed.style.display = 'none';
        elements.placeholder.style.display = 'block';
        
        elements.faceList.innerHTML = '<div class="empty-state">No faces detected</div>';
        elements.totalFaces.textContent = '0';
        elements.forwardFaces.textContent = '0';
        elements.attentionRate.textContent = '0%';

        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function connectToFeed() {
        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource('/video_feed');

        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                updateUI(data);
            } catch (err) {
                console.error('Failed to parse SSE data:', err);
            }
        };

        eventSource.onerror = function() {
            console.error('SSE connection error');
            stopProcessing();
        };
    }

    function updateUI(data) {
        if (data.frame) {
            elements.videoFeed.src = 'data:image/jpeg;base64,' + data.frame;
        }

        if (data.face_statuses && Array.isArray(data.face_statuses)) {
            updateFaceList(data.face_statuses);
        }

        if (data.stats) {
            elements.totalFaces.textContent = data.stats.total_faces;
            elements.forwardFaces.textContent = data.stats.forward_faces;
            elements.attentionRate.textContent = data.stats.attention_rate + '%';
        }
    }

    function updateFaceList(faces) {
        if (faces.length === 0) {
            elements.faceList.innerHTML = '<div class="empty-state">No faces detected</div>';
            return;
        }

        const html = faces.map(face => {
            const isForward = face.is_forward;
            const statusClass = isForward ? 'forward' : 'not-forward';
            const eyeClass = face.eye_state === 'Eyes Open' ? 'eyes-open' : 'eyes-closed';
            
            return `
                <div class="face-item ${statusClass}">
                    <div class="face-id">Face #${face.id}</div>
                    <div class="face-status">
                        <span class="indicator ${statusClass}"></span>
                        <span class="label ${statusClass}">${face.label}</span>
                    </div>
                    <div class="face-eyes">
                        <span class="${eyeClass}">${face.eye_state}</span>
                    </div>
                    <div class="face-angles">
                        <span>Yaw: ${face.yaw}°</span>
                        <span>Pitch: ${face.pitch}°</span>
                    </div>
                </div>
            `;
        }).join('');

        elements.faceList.innerHTML = html;
    }

    init();
})();
