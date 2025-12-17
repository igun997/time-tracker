// YOLO Time Tracker Frontend Application

class TimeTrackerApp {
    constructor() {
        this.ws = null;
        this.wsEvents = null;
        this.isConnected = false;
        this.isDetectionRunning = false;

        this.elements = {
            connectionStatus: document.getElementById('connection-status'),
            detectionStatus: document.getElementById('detection-status'),
            videoFeed: document.getElementById('video-feed'),
            videoPlaceholder: document.getElementById('video-placeholder'),
            sourceSelect: document.getElementById('source-select'),
            demoMode: document.getElementById('demo-mode'),
            startBtn: document.getElementById('start-detection'),
            stopBtn: document.getElementById('stop-detection'),
            personsList: document.getElementById('persons-list'),
            statFps: document.getElementById('stat-fps'),
            statPersons: document.getElementById('stat-persons'),
            statIdentified: document.getElementById('stat-identified'),
            statUptime: document.getElementById('stat-uptime'),
            employeesTableBody: document.getElementById('employees-table-body'),
            sourcesTableBody: document.getElementById('sources-table-body'),
            reportDate: document.getElementById('report-date'),
            reportTableBody: document.getElementById('report-table-body'),
            reportSummary: document.getElementById('report-summary'),
            eventsLog: document.getElementById('events-log'),
            modal: document.getElementById('modal'),
            modalTitle: document.getElementById('modal-title'),
            modalBody: document.getElementById('modal-body')
        };

        this.init();
    }

    init() {
        // Set default date to today
        this.elements.reportDate.value = new Date().toISOString().split('T')[0];

        // Bind event handlers
        this.bindEvents();

        // Load initial data
        this.loadSources();
        this.loadEmployees();

        // Check API status
        this.checkStatus();
    }

    bindEvents() {
        // Detection controls
        this.elements.startBtn.addEventListener('click', () => this.startDetection());
        this.elements.stopBtn.addEventListener('click', () => this.stopDetection());
        document.getElementById('refresh-sources').addEventListener('click', () => this.loadSources());

        // Tab navigation
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Quick actions
        document.getElementById('btn-add-source').addEventListener('click', () => this.showAddSourceModal());
        document.getElementById('btn-add-employee').addEventListener('click', () => this.showAddEmployeeModal());
        document.getElementById('btn-view-report').addEventListener('click', () => {
            this.switchTab('reports');
            this.loadReport();
        });
        document.getElementById('btn-new-employee').addEventListener('click', () => this.showAddEmployeeModal());
        document.getElementById('btn-new-source').addEventListener('click', () => this.showAddSourceModal());

        // Report
        document.getElementById('btn-load-report').addEventListener('click', () => this.loadReport());

        // Events
        document.getElementById('btn-clear-events').addEventListener('click', () => this.clearEvents());

        // Modal close
        document.querySelector('.modal-close').addEventListener('click', () => this.hideModal());
        this.elements.modal.addEventListener('click', (e) => {
            if (e.target === this.elements.modal) this.hideModal();
        });
    }

    // WebSocket Connection
    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.isConnected = true;
            this.elements.connectionStatus.textContent = 'Connected';
            this.elements.connectionStatus.classList.remove('disconnected');
            this.elements.connectionStatus.classList.add('connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleFrame(data);
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.elements.connectionStatus.textContent = 'Disconnected';
            this.elements.connectionStatus.classList.remove('connected');
            this.elements.connectionStatus.classList.add('disconnected');

            // Reconnect if detection is still running
            if (this.isDetectionRunning) {
                setTimeout(() => this.connectWebSocket(), 2000);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    connectEventsWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/events`;

        this.wsEvents = new WebSocket(wsUrl);

        this.wsEvents.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleEvent(data);
        };
    }

    disconnectWebSocket() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        if (this.wsEvents) {
            this.wsEvents.close();
            this.wsEvents = null;
        }
    }

    // Frame handling
    handleFrame(data) {
        if (data.type === 'frame') {
            // Update video feed
            this.elements.videoFeed.src = `data:image/jpeg;base64,${data.frame}`;
            this.elements.videoPlaceholder.classList.add('hidden');

            // Update stats
            this.elements.statFps.textContent = data.stats.fps.toFixed(1);
            this.elements.statPersons.textContent = data.stats.active_persons;
            this.elements.statIdentified.textContent = data.stats.identified;
            this.elements.statUptime.textContent = this.formatDuration(data.stats.uptime);

            // Update persons list
            this.updatePersonsList(data.persons);
        }
    }

    handleEvent(data) {
        this.addEventLog(data);
    }

    // Detection Control
    async startDetection() {
        const sourceId = this.elements.sourceSelect.value;
        if (!sourceId) {
            alert('Please select a video source');
            return;
        }

        const demoMode = this.elements.demoMode.checked;

        try {
            const response = await fetch('/api/detection/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source_id: parseInt(sourceId),
                    demo_mode: demoMode,
                    frame_skip: 2,
                    confidence_threshold: 0.5
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start detection');
            }

            this.isDetectionRunning = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            this.elements.detectionStatus.textContent = 'Detection Running';
            this.elements.detectionStatus.classList.remove('stopped');
            this.elements.detectionStatus.classList.add('running');

            this.connectWebSocket();
            this.connectEventsWebSocket();

        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    async stopDetection() {
        try {
            const response = await fetch('/api/detection/stop', { method: 'POST' });

            if (!response.ok) {
                throw new Error('Failed to stop detection');
            }

            this.isDetectionRunning = false;
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            this.elements.detectionStatus.textContent = 'Detection Stopped';
            this.elements.detectionStatus.classList.remove('running');
            this.elements.detectionStatus.classList.add('stopped');

            this.disconnectWebSocket();
            this.elements.videoPlaceholder.classList.remove('hidden');
            this.elements.videoFeed.src = '';

        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    // Data Loading
    async loadSources() {
        try {
            const response = await fetch('/api/sources');
            const data = await response.json();

            // Update select
            this.elements.sourceSelect.innerHTML = '<option value="">Select source...</option>';
            data.sources.forEach(source => {
                const option = document.createElement('option');
                option.value = source.id;
                option.textContent = `${source.name} (${source.source_type})`;
                this.elements.sourceSelect.appendChild(option);
            });

            // Update table
            this.updateSourcesTable(data.sources);

        } catch (error) {
            console.error('Error loading sources:', error);
        }
    }

    async loadEmployees() {
        try {
            const response = await fetch('/api/employees');
            const data = await response.json();
            this.updateEmployeesTable(data.employees);
        } catch (error) {
            console.error('Error loading employees:', error);
        }
    }

    async loadReport() {
        const date = this.elements.reportDate.value;
        if (!date) return;

        try {
            const response = await fetch(`/api/reports/daily?report_date=${date}`);
            const data = await response.json();
            this.updateReportTable(data);
        } catch (error) {
            console.error('Error loading report:', error);
        }
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            if (status.detection_running) {
                this.isDetectionRunning = true;
                this.elements.startBtn.disabled = true;
                this.elements.stopBtn.disabled = false;
                this.elements.detectionStatus.textContent = 'Detection Running';
                this.elements.detectionStatus.classList.add('running');
                this.connectWebSocket();
                this.connectEventsWebSocket();
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }

    // UI Updates
    updatePersonsList(persons) {
        if (persons.length === 0) {
            this.elements.personsList.innerHTML = '<p class="empty-message">No persons detected</p>';
            return;
        }

        this.elements.personsList.innerHTML = persons.map(person => `
            <div class="person-card">
                <div class="person-info">
                    <div class="person-name ${person.is_identified ? 'identified' : 'unknown'}">
                        ${person.name}
                    </div>
                    <div class="person-duration">${this.formatDuration(person.duration_seconds)}</div>
                </div>
                ${!person.is_identified ? `
                    <div class="person-actions">
                        <button class="btn btn-small" onclick="app.showIdentifyModal(${person.track_id})">
                            Identify
                        </button>
                    </div>
                ` : ''}
            </div>
        `).join('');
    }

    updateEmployeesTable(employees) {
        if (employees.length === 0) {
            this.elements.employeesTableBody.innerHTML = '<tr><td colspan="5" class="empty-message">No employees registered</td></tr>';
            return;
        }

        this.elements.employeesTableBody.innerHTML = employees.map(emp => `
            <tr>
                <td>${emp.name}</td>
                <td>${emp.employee_id || '-'}</td>
                <td>${emp.face_count}</td>
                <td>${emp.is_active ? 'Active' : 'Inactive'}</td>
                <td>
                    <button class="btn btn-small" onclick="app.showUploadFaceModal(${emp.id}, '${emp.name}')">
                        Add Face
                    </button>
                    <button class="btn btn-small btn-danger" onclick="app.deleteEmployee(${emp.id})">
                        Delete
                    </button>
                </td>
            </tr>
        `).join('');
    }

    updateSourcesTable(sources) {
        if (sources.length === 0) {
            this.elements.sourcesTableBody.innerHTML = '<tr><td colspan="5" class="empty-message">No sources configured</td></tr>';
            return;
        }

        this.elements.sourcesTableBody.innerHTML = sources.map(src => `
            <tr>
                <td>${src.name}</td>
                <td>${src.source_type}</td>
                <td>${src.source_url || (src.device_index !== null ? `Device ${src.device_index}` : '-')}</td>
                <td>${src.is_active ? 'Active' : 'Inactive'}</td>
                <td>
                    <button class="btn btn-small btn-danger" onclick="app.deleteSource(${src.id})">
                        Delete
                    </button>
                </td>
            </tr>
        `).join('');
    }

    updateReportTable(report) {
        if (report.entries.length === 0) {
            this.elements.reportTableBody.innerHTML = '<tr><td colspan="3" class="empty-message">No data for this date</td></tr>';
            this.elements.reportSummary.innerHTML = '';
            return;
        }

        this.elements.reportTableBody.innerHTML = report.entries.map(entry => `
            <tr>
                <td>${entry.employee_name}</td>
                <td>${entry.formatted_duration}</td>
                <td>${(entry.total_seconds / 3600).toFixed(2)}</td>
            </tr>
        `).join('');

        this.elements.reportSummary.innerHTML = `
            <p><strong>Total Employees:</strong> ${report.employee_count}</p>
            <p><strong>Total Time:</strong> ${report.total_formatted}</p>
        `;
    }

    addEventLog(event) {
        const log = this.elements.eventsLog;
        if (log.querySelector('.empty-message')) {
            log.innerHTML = '';
        }

        const eventEl = document.createElement('div');
        eventEl.className = `event-item ${event.event_type}`;
        eventEl.innerHTML = `
            <div class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</div>
            <div class="event-message">
                <strong>${event.employee_name}</strong> ${event.event_type} session at ${event.source_name}
                ${event.event_type === 'ended' ? `(Duration: ${this.formatDuration(event.duration_seconds)})` : ''}
            </div>
        `;

        log.insertBefore(eventEl, log.firstChild);

        // Limit to 50 events
        while (log.children.length > 50) {
            log.removeChild(log.lastChild);
        }
    }

    clearEvents() {
        this.elements.eventsLog.innerHTML = '<p class="empty-message">No events yet</p>';
    }

    // Tab Navigation
    switchTab(tabName) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`tab-${tabName}`).classList.add('active');
    }

    // Modals
    showModal(title, content) {
        this.elements.modalTitle.textContent = title;
        this.elements.modalBody.innerHTML = content;
        this.elements.modal.classList.remove('hidden');
    }

    hideModal() {
        this.elements.modal.classList.add('hidden');
    }

    showAddEmployeeModal() {
        this.showModal('Add Employee', `
            <form id="add-employee-form">
                <div class="form-group">
                    <label for="emp-name">Name *</label>
                    <input type="text" id="emp-name" required>
                </div>
                <div class="form-group">
                    <label for="emp-id">Employee ID</label>
                    <input type="text" id="emp-id">
                </div>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Add Employee</button>
                    <button type="button" class="btn" onclick="app.hideModal()">Cancel</button>
                </div>
            </form>
        `);

        document.getElementById('add-employee-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.addEmployee();
        });
    }

    showAddSourceModal() {
        this.showModal('Add Video Source', `
            <form id="add-source-form">
                <div class="form-group">
                    <label for="src-name">Name *</label>
                    <input type="text" id="src-name" required>
                </div>
                <div class="form-group">
                    <label for="src-type">Type *</label>
                    <select id="src-type" required>
                        <option value="webcam">Webcam</option>
                        <option value="rtsp">RTSP Stream</option>
                        <option value="file">Video File</option>
                    </select>
                </div>
                <div class="form-group" id="src-url-group">
                    <label for="src-url">URL / Path</label>
                    <input type="text" id="src-url" placeholder="rtsp://... or /path/to/video.mp4">
                </div>
                <div class="form-group" id="src-device-group">
                    <label for="src-device">Device Index</label>
                    <input type="number" id="src-device" value="0" min="0">
                </div>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Add Source</button>
                    <button type="button" class="btn" onclick="app.hideModal()">Cancel</button>
                </div>
            </form>
        `);

        document.getElementById('src-type').addEventListener('change', (e) => {
            const isWebcam = e.target.value === 'webcam';
            document.getElementById('src-url-group').style.display = isWebcam ? 'none' : 'block';
            document.getElementById('src-device-group').style.display = isWebcam ? 'block' : 'none';
        });

        document.getElementById('add-source-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.addSource();
        });
    }

    showUploadFaceModal(employeeId, employeeName) {
        this.showModal(`Upload Face - ${employeeName}`, `
            <form id="upload-face-form">
                <div class="form-group">
                    <label for="face-file">Select Photo</label>
                    <input type="file" id="face-file" accept="image/jpeg,image/png" required>
                </div>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
                    Upload a clear photo of the employee's face. The photo should show the face clearly and be well-lit.
                </p>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Upload</button>
                    <button type="button" class="btn" onclick="app.hideModal()">Cancel</button>
                </div>
            </form>
        `);

        document.getElementById('upload-face-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.uploadFace(employeeId);
        });
    }

    showIdentifyModal(trackId) {
        // First load employees
        fetch('/api/employees')
            .then(res => res.json())
            .then(data => {
                const options = data.employees.map(e =>
                    `<option value="${e.id}">${e.name}</option>`
                ).join('');

                this.showModal('Identify Person', `
                    <form id="identify-form">
                        <div class="form-group">
                            <label for="identify-employee">Select Employee</label>
                            <select id="identify-employee" required>
                                <option value="">Choose employee...</option>
                                ${options}
                            </select>
                        </div>
                        <div class="button-group">
                            <button type="submit" class="btn btn-primary">Identify</button>
                            <button type="button" class="btn" onclick="app.hideModal()">Cancel</button>
                        </div>
                    </form>
                `);

                document.getElementById('identify-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    await this.identifyPerson(trackId);
                });
            });
    }

    // API Actions
    async addEmployee() {
        const name = document.getElementById('emp-name').value;
        const employeeId = document.getElementById('emp-id').value || null;

        try {
            const response = await fetch('/api/employees', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, employee_id: employeeId })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to add employee');
            }

            this.hideModal();
            this.loadEmployees();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    async addSource() {
        const name = document.getElementById('src-name').value;
        const sourceType = document.getElementById('src-type').value;
        const sourceUrl = document.getElementById('src-url').value || null;
        const deviceIndex = parseInt(document.getElementById('src-device').value) || 0;

        try {
            const response = await fetch('/api/sources', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    source_type: sourceType,
                    source_url: sourceUrl,
                    device_index: sourceType === 'webcam' ? deviceIndex : null
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to add source');
            }

            this.hideModal();
            this.loadSources();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    async uploadFace(employeeId) {
        const fileInput = document.getElementById('face-file');
        const file = fileInput.files[0];

        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`/api/employees/${employeeId}/faces`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to upload face');
            }

            const result = await response.json();
            alert(result.message);
            this.hideModal();
            this.loadEmployees();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    async identifyPerson(trackId) {
        const employeeId = document.getElementById('identify-employee').value;

        if (!employeeId) return;

        try {
            const response = await fetch(`/api/detection/identify?track_id=${trackId}&employee_id=${employeeId}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to identify person');
            }

            this.hideModal();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    async deleteEmployee(id) {
        if (!confirm('Are you sure you want to delete this employee?')) return;

        try {
            const response = await fetch(`/api/employees/${id}`, { method: 'DELETE' });
            if (response.ok) {
                this.loadEmployees();
            }
        } catch (error) {
            alert('Failed to delete employee');
        }
    }

    async deleteSource(id) {
        if (!confirm('Are you sure you want to delete this source?')) return;

        try {
            const response = await fetch(`/api/sources/${id}`, { method: 'DELETE' });
            if (response.ok) {
                this.loadSources();
            }
        } catch (error) {
            alert('Failed to delete source');
        }
    }

    // Utilities
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize app
const app = new TimeTrackerApp();
