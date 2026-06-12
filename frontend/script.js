const BACKEND_URL = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/ar-stream";

let socket = null;
let currentTab = "login";
let isChatMinimized = false;

// Các biến phục vụ Zoom/Pan bản đồ
let mapScale = 1;
let mapOffsetX = 0;
let mapOffsetY = 0;
let isMapDragging = false;
let startMapDragX = 0;
let startMapDragY = 0;

// Các biến phục vụ Drag (kéo thả) Chatbot
let isDraggingChat = false;
let chatOffsetX = 0;
let chatOffsetY = 0;

// Vị trí hiển thị chuẩn xác trên Canvas (hệ tọa độ 0-100) theo sơ đồ thực tế
const CUSTOM_UI_POSITIONS = {
    "Tòa A": { x: 10, y: 40 },
    "Tòa B": { x: 22, y: 20 },
    "Tòa C": { x: 34, y: 40 },
    "Tòa D": { x: 46, y: 40 },
    "Tòa E": { x: 58, y: 40 },
    "Tòa F": { x: 78, y: 40 },
    "Tòa G": { x: 90, y: 40 },
    "Nhà thể dục": { x: 75, y: 10 },
    "Nhà xe": { x: 40, y: 85 },
    "Cây ATM": { x: 38, y: 65 },
    "Nhà điều hành": { x: 75, y: 85 },
    "Cổng trường": { x: 58, y: 85 },
    "Canteen": { x: 46, y: 15 },
    "Căn tin": { x: 46, y: 15 },
    "ATM": { x: 38, y: 65 },
    "Cây ATM": { x: 38, y: 65 }
};

document.addEventListener("DOMContentLoaded", () => {
    initializeWebcamSimulator();
    verifyAuthTokenState();
    initMapControls();
    initChatDrag();
});

function initChatDrag() {
    // Đã vô hiệu hoá tính năng kéo thả Chatbot theo yêu cầu của user
    return;
}

function initMapControls() {
    const container = document.getElementById('canvas-container');
    const canvasEl = document.getElementById('campus-map-canvas');
    if (!container || !canvasEl) return;

    container.addEventListener('wheel', (e) => {
        if (document.getElementById('canvas-container').classList.contains('hidden')) return;
        e.preventDefault();
        const rect = canvasEl.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomIntensity = 0.1;
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoom = Math.exp(wheel * zoomIntensity);
        
        const newScale = mapScale * zoom;
        if (newScale < 0.2 || newScale > 10) return;
        
        mapOffsetX = mouseX - (mouseX - mapOffsetX) * zoom;
        mapOffsetY = mouseY - (mouseY - mapOffsetY) * zoom;
        mapScale = newScale;
        
        drawMap();
    }, { passive: false });

    container.addEventListener('mousedown', (e) => {
        if (document.getElementById('canvas-container').classList.contains('hidden')) return;
        isMapDragging = true;
        startMapDragX = e.clientX - mapOffsetX;
        startMapDragY = e.clientY - mapOffsetY;
        // Ghi nhận toạ độ bắt đầu click (để phân biệt drag và click)
        window.lastClickDownX = e.clientX;
        window.lastClickDownY = e.clientY;
        container.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (isMapDragging) {
            mapOffsetX = e.clientX - startMapDragX;
            mapOffsetY = e.clientY - startMapDragY;
            drawMap();
        }
    });

    window.addEventListener('mouseup', () => {
        isMapDragging = false;
        container.style.cursor = 'grab';
    });
}

async function initializeWebcamSimulator() {
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("face-overlay");
    const ctx = canvas.getContext("2d");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = stream;
        
        function drawOverlayLoop() {
            if (!video.paused && !video.ended) {
                canvas.width = video.clientWidth;
                canvas.height = video.clientHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                ctx.strokeStyle = "rgba(34, 221, 238, 0.4)";
                ctx.lineWidth = 1;
                ctx.strokeRect(canvas.width/2 - 40, canvas.height/2 - 50, 80, 100);
                
                ctx.beginPath();
                ctx.moveTo(canvas.width/2, canvas.height/2 - 30);
                ctx.lineTo(canvas.width/2 - 30, canvas.height/2);
                ctx.lineTo(canvas.width/2, canvas.height/2 + 30);
                ctx.lineTo(canvas.width/2 + 30, canvas.height/2);
                ctx.closePath();
                ctx.stroke();
            }
            requestAnimationFrame(drawOverlayLoop);
        }
        video.addEventListener("play", drawOverlayLoop);
    } catch (err) {
        showToast("Lỗi: Không thể truy cập Camera. Vui lòng cấp quyền!", "error");
        ctx.fillStyle = "#020617";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.font = "12px monospace";
        ctx.fillStyle = "#ef4444";
        ctx.fillText("CAMERA OFFLINE", 20, 40);
    }
}

function captureFaceBase64() {
    const video = document.getElementById("webcam");
    if (!video.videoWidth) return ""; 

    const hiddenCanvas = document.createElement("canvas");
    hiddenCanvas.width = video.videoWidth;
    hiddenCanvas.height = video.videoHeight;
    const ctx = hiddenCanvas.getContext("2d");
    
    ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    return hiddenCanvas.toDataURL("image/jpeg", 0.8);
}

async function verifyAuthTokenState() {
    const authView = document.getElementById("auth-view");
    const token = localStorage.getItem("token");
    if (token) {
        try {
            const res = await fetch(`${BACKEND_URL}/users/me`, {
                headers: { "Authorization": `Bearer ${token}` }
            });
            if (res.ok) {
                const userData = await res.json();
                activateMainDashboardView(userData);
            } else {
                localStorage.removeItem("token");
                authView.classList.remove("hidden");
            }
        } catch (e) {
            showToast("Mất kết nối gateway backend", "error");
            authView.classList.remove("hidden");
        }
    } else {
        authView.classList.remove("hidden");
    }
}

function switchAuthTab(tab) {
    currentTab = tab;
    const btnLogin = document.getElementById("tab-login");
    const btnRegister = document.getElementById("tab-register");
    const formLogin = document.getElementById("form-login");
    const formRegister = document.getElementById("form-register");

    if (tab === "login") {
        btnLogin.className = "flex-1 py-2 text-sm font-semibold rounded-lg bg-blue-600 text-white transition-all";
        btnRegister.className = "flex-1 py-2 text-sm font-semibold rounded-lg text-slate-400 transition-all";
        formLogin.classList.remove("hidden");
        formRegister.classList.add("hidden");
    } else {
        btnRegister.className = "flex-1 py-2 text-sm font-semibold rounded-lg bg-emerald-600 text-white transition-all";
        btnLogin.className = "flex-1 py-2 text-sm font-semibold rounded-lg text-slate-400 transition-all";
        formRegister.classList.remove("hidden");
        formLogin.classList.add("hidden");
    }
}

async function submitRegister(e) {
    e.preventDefault();
    const name = document.getElementById("reg-name").value.trim();
    const email = document.getElementById("reg-email").value.trim();
    const password = document.getElementById("reg-password").value;
    
    // Ràng buộc 1: Kiểm tra định dạng Email chuẩn bằng biểu thức chính quy (Regex)
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        showToast("Định dạng email không hợp lệ (Thiếu @ hoặc sai tên miền)!", "error");
        return;
    }

    // Ràng buộc 2: Kiểm tra độ dài mật khẩu nghiêm ngặt tối thiểu 6 ký tự
    if (password.length < 6) {
        showToast("Mật khẩu đăng ký bắt buộc phải từ 6 ký tự trở lên!", "error");
        return;
    }
    
    const faceDataStr = captureFaceBase64();
    if (!faceDataStr) {
        showToast("Vui lòng nhìn vào camera để lấy dữ liệu khuôn mặt!", "error");
        return;
    }

    showToast("Đang trích xuất Vector khuôn mặt...", "success");

    try {
        const res = await fetch(`${BACKEND_URL}/users/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ FullName: name, Email: email, Password: password, FaceData: faceDataStr })
        });
        const data = await res.json();

        if (res.ok) {
            showToast("Đăng ký & Lưu khuôn mặt thành công!", "success");
            switchAuthTab("login");
            document.getElementById("login-email").value = email;
            document.getElementById("login-password").value = ""; 
        } else {
            showToast(data.detail || "Đăng ký thất bại", "error");
        }
    } catch (err) {
        showToast("Lỗi mạng, không thể kết nối Server", "error");
    }
}

// Hàm xử lý đăng nhập có tham số type
async function submitLogin(e, loginType) {
    e.preventDefault();
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value;
    
    if (!email) {
        showToast("Vui lòng nhập Email sinh viên trước khi tiếp tục", "error");
        return;
    }

    // Kiểm tra định dạng email trước khi gửi đi
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        showToast("Định dạng email nhập vào không đúng!", "error");
        return;
    }

    let faceDataStr = "";
    let passwordToSend = "";

    if (loginType === 'face') {
        faceDataStr = captureFaceBase64();
        if (!faceDataStr) {
            showToast("Vui lòng nhìn vào camera để quét FaceID", "error");
            return;
        }
        showToast("Đang quét và so khớp FaceID...", "success");
    } else if (loginType === 'password') {
        if (!password) {
            showToast("Vui lòng nhập mật khẩu", "error");
            return;
        }
        // Kiểm tra độ dài mật khẩu ngay tại Frontend
        if (password.length < 6) {
            showToast("Mật khẩu hệ thống không khớp (Tối thiểu phải từ 6 ký tự)!", "error");
            return;
        }
        passwordToSend = password;
        showToast("Đang xác thực mật khẩu...", "success");
    }

    try {
        const res = await fetch(`${BACKEND_URL}/users/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                Email: email, 
                Password: passwordToSend,
                FaceData: faceDataStr 
            })
        });
        const data = await res.json();

        if (res.ok) {
            localStorage.setItem("token", data.access_token);
            showToast("Đăng nhập thành công!", "success");
            
            const userRes = await fetch(`${BACKEND_URL}/users/me`, {
                headers: { "Authorization": `Bearer ${data.access_token}` }
            });
            const userData = await userRes.json();
            activateMainDashboardView(userData);
        } else {
            showToast(data.detail || "Xác thực thất bại", "error");
        }
    } catch (err) {
        showToast("Không thể kết nối đến API server", "error");
    }
}

function activateMainDashboardView(user) {
    document.getElementById("auth-view").classList.add("hidden");
    document.getElementById("main-view").classList.remove("hidden");
    document.getElementById("main-view").classList.add("flex");
    document.getElementById("user-display").innerText = `👤 ${user.FullName}`;
    
    const video = document.getElementById("webcam");
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    establishWebSocketChannel();
}

function executeLogout() {
    localStorage.removeItem("token");
    if (socket) socket.close();
    location.reload();
}

function establishWebSocketChannel() {
    socket = new WebSocket(WS_URL);
    socket.onopen = () => showToast("Đã đồng bộ thời gian thực qua WebSocket", "success");
    socket.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            // Bỏ qua tin nhắn "đang xử lý" từ backend cũ
            if (payload.type === "chat_reply" && !payload.message.includes("đang xử lý")) {
                appendChatBubble(payload.message, "bot");
            }
        } catch(e) {}
    };
    socket.onclose = () => setTimeout(establishWebSocketChannel, 3000);
}

async function transmitChatMessage() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;

    appendChatBubble(message, "user");
    input.value = "";
    
    // Hiệu ứng "Đang gõ..."
    const processingId = "proc-" + Date.now();
    const logs = document.getElementById("chat-logs");
    const procBubble = document.createElement("div");
    procBubble.id = processingId;
    procBubble.className = "bg-slate-800/60 border border-slate-700 text-slate-400 p-2.5 rounded-xl max-w-[85%] self-start shadow-md text-xs italic animate-pulse";
    procBubble.innerText = "RAG AI đang phân tích...";
    logs.appendChild(procBubble);
    logs.scrollTop = logs.scrollHeight;

    try {
        const response = await fetch(`${BACKEND_URL}/chat/query?q=${encodeURIComponent(message)}`);
        const data = await response.json();
        
        const el = document.getElementById(processingId);
        if (el) el.remove();
        
        if (data.status === "success") {
            appendChatBubble(data.answer, "bot");
            
            // Đẩy lịch sử tìm kiếm lên Firebase
            const token = localStorage.getItem("token");
            if (token) {
                fetch(`${BACKEND_URL}/history/search`, {
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${token}`
                    },
                    body: JSON.stringify({ query: message, matched_node: data.answer.substring(0, 50) })
                }).catch(e => console.error("Search history save error", e));
            }
        } else {
            appendChatBubble("Xin lỗi, hệ thống đang bận. Bạn thử lại sau nhé!", "bot");
        }
    } catch (error) {
        const el = document.getElementById(processingId);
        if (el) el.remove();
        appendChatBubble("Không thể kết nối đến máy chủ RAG.", "bot");
    }
}

function transmitWebSocketPayload(obj) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(obj));
    } else {
        showToast("Lỗi: Kênh truyền thông tin thời gian thực đang nghẽn", "error");
    }
}

function appendChatBubble(text, sender) {
    const logs = document.getElementById("chat-logs");
    const bubble = document.createElement("div");
    
    if (sender === "user") {
        bubble.className = "bg-blue-600 text-white p-3.5 rounded-2xl rounded-tr-sm max-w-[85%] self-end ml-auto shadow-md leading-relaxed animate-fade-in";
    } else {
        bubble.className = "bg-slate-800 border border-slate-700 text-slate-200 p-3.5 rounded-2xl rounded-tl-sm max-w-[85%] self-start shadow-md leading-relaxed animate-fade-in";
    }
    
    bubble.innerText = text;
    logs.appendChild(bubble);
    logs.scrollTop = logs.scrollHeight;
}

function checkChatEnter(e) {
    if (e.key === "Enter") transmitChatMessage();
}

let isRecording = false;
let recognition = null;

function toggleVoiceInput() {
    const btn = document.getElementById("btn-voice");
    const input = document.getElementById("chat-input");

    if (!isRecording) {
        // Kiểm tra xem trình duyệt có hỗ trợ Web Speech API không
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            showToast("Trình duyệt của bạn không hỗ trợ nhận diện giọng nói", "error");
            return;
        }

        recognition = new SpeechRecognition();
        recognition.lang = 'vi-VN'; // Ngôn ngữ tiếng Việt
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = function() {
            isRecording = true;
            btn.classList.add("text-rose-500", "animate-pulse"); // Đổi màu nút sang đỏ để báo đang thu
            input.placeholder = "Đang nghe...";
        };

        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            input.value = transcript;
            transmitChatMessage(); // Tự động gửi tin nhắn sau khi nói xong
        };

        recognition.onerror = function(event) {
            showToast("Lỗi nhận diện giọng nói: " + event.error, "error");
            stopVoiceInput();
        };

        recognition.onend = function() {
            stopVoiceInput();
        };

        recognition.start();

    } else {
        stopVoiceInput();
        if (recognition) recognition.stop();
    }
}

function stopVoiceInput() {
    isRecording = false;
    const btn = document.getElementById("btn-voice");
    const input = document.getElementById("chat-input");
    btn.classList.remove("text-rose-500", "animate-pulse");
    input.placeholder = "Hỏi RAG Chatbot...";
}

function minimizeChat(action = 'toggle') {
    const chatWin = document.getElementById("chat-window");
    const toggleIcon = document.getElementById("chat-toggle-icon");

    if (action === 'minimize' && isChatMinimized) return;
    if (action === 'maximize' && !isChatMinimized) return;

    if (!isChatMinimized) {
        // Trạng thái: Thu nhỏ lại
        chatWin.classList.remove("h-[450px]", "md:h-[500px]");
        chatWin.classList.add("h-[56px]");
        isChatMinimized = true;
        // Đổi icon thành dấu Cộng (+)
        toggleIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />';
    } else {
        // Trạng thái: Phóng to ra
        chatWin.classList.remove("h-[56px]");
        chatWin.classList.add("h-[450px]", "md:h-[500px]");
        isChatMinimized = false;
        // Đổi icon về lại dấu Trừ (-)
        toggleIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" d="M19.5 12h-15" />';
    }
}

function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    toast.innerText = message;
    toast.style.transform = "";
    
    if (type === "success") {
        toast.className = "fixed top-5 right-5 z-50 bg-slate-950 border border-emerald-500/40 text-emerald-400 px-4 py-2.5 rounded-xl shadow-2xl text-xs font-semibold translate-x-0 transition-all duration-300";
    } else {
        toast.className = "fixed top-5 right-5 z-50 bg-slate-950 border border-rose-500/40 text-rose-400 px-4 py-2.5 rounded-xl shadow-2xl text-xs font-semibold translate-x-0 transition-all duration-300";
    }

    setTimeout(() => toast.style.transform = "translateX(150%)", 4000);
}

// Hàm ẩn/hiện mật khẩu
function togglePasswordVisibility(inputId, iconId) {
    const passwordInput = document.getElementById(inputId);
    const eyeIcon = document.getElementById(iconId);

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        // Thay đổi cấu trúc SVG thành icon con mắt có đường gạch chéo (Ẩn mật khẩu)
        eyeIcon.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
        `;
    } else {
        passwordInput.type = "password";
        // Khôi phục lại icon con mắt mở bình thường
        eyeIcon.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        `;
    }
}

// =========================================================
// MODULE CANVAS MAP & AI RECOMMENDATION
// =========================================================

let campusNodesLoaded = false;
let currentRouteWaypoints = [];
let graphData = null; // Store nodes, edges, bounds
let currentGPSPos = null;
let gpsSimulationInterval = null;
const canvas = document.getElementById('campus-map-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;

function resizeCanvas() {
    if(!canvas) return;
    const container = document.getElementById('canvas-container');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawMap();
}
window.addEventListener('resize', resizeCanvas);

// Hàm lấy danh sách điểm từ API của Khoa
async function fetchCampusNodes() {
    if (campusNodesLoaded) return; 
    try {
        const nodesRes = await fetch(`${BACKEND_URL}/map/api_get_graph`); 
        
        if (nodesRes.ok) {
            const data = await nodesRes.json();
            graphData = data; // Save graph data
            
            const startSel = document.getElementById("route-start");
            const endSel = document.getElementById("route-end");
            const waySel = document.getElementById("route-waypoint");
            
            startSel.innerHTML = '<option value="">-- Điểm xuất phát --</option>';
            endSel.innerHTML = '<option value="">-- Điểm đến --</option>';
            if(waySel) waySel.innerHTML = '<option value="">-- Điểm dừng (tuỳ chọn) --</option>';
            
            // API của Khoa trả về một mảng các object 'nodes', mỗi object có trường 'id'
            data.nodes.forEach(nodeObj => {
                const nodeId = nodeObj.id; // Lấy ID của node (VD: "Toa_A")
                const text = nodeId.replace(/_/g, " ");
                startSel.innerHTML += `<option value="${nodeId}">${text}</option>`;
                endSel.innerHTML += `<option value="${nodeId}">${text}</option>`;
                if(waySel) waySel.innerHTML += `<option value="${nodeId}">${text}</option>`;
            });
            
            campusNodesLoaded = true;
            setTimeout(resizeCanvas, 100);
            showToast("Đã tải dữ liệu Không gian 2D", "success");
        }
    } catch (e) {
        console.error(e);
        showToast("Lỗi đồng bộ đồ thị GNN", "error");
    }
}

// Vẽ Đồ thị tĩnh
function drawMap() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!graphData || !graphData.nodes) return;

    // Ghi đè tọa độ Node thành Grid 0-100 theo hình sơ đồ
    const nodeDict = {};
    graphData.nodes.forEach(n => {
        if (CUSTOM_UI_POSITIONS[n.id]) {
            n.ui_x = CUSTOM_UI_POSITIONS[n.id].x;
            n.ui_y = CUSTOM_UI_POSITIONS[n.id].y;
        } else {
            n.ui_x = 50; n.ui_y = 50;
        }
        nodeDict[n.id] = n;
    });

    const padding = 50;
    const drawW = canvas.width - padding * 2;
    const drawH = canvas.height - padding * 2;

    const mapX = (x) => {
        const base = padding + (x / 100) * drawW;
        return base * mapScale + mapOffsetX;
    };
    const mapY = (y) => {
        const base = padding + (y / 100) * drawH; 
        return base * mapScale + mapOffsetY;
    };

    // Vẽ các cạnh (Edges)
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(59, 130, 246, 0.4)";
    if (graphData.edges) {
        graphData.edges.forEach(edge => {
            const u = nodeDict[edge.source];
            const v = nodeDict[edge.target];
            if (u && v) {
                ctx.beginPath();
                ctx.moveTo(mapX(u.ui_x), mapY(u.ui_y));
                ctx.lineTo(mapX(v.ui_x), mapY(v.ui_y));
                ctx.stroke();
            }
        });
    }

    // Vẽ lộ trình (Đường dây nối)
    if (currentRouteWaypoints.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#34d399'; 
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#34d399';
        
        for (let i = 0; i < currentRouteWaypoints.length; i++) {
            const nodeId = currentRouteWaypoints[i];
            const node = nodeDict[nodeId];
            if (!node) continue;
            
            const x = mapX(node.ui_x);
            const y = mapY(node.ui_y);
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    // Vẽ các đỉnh (Nodes) bằng Icon (Emoji)
    graphData.nodes.forEach(node => {
        const x = mapX(node.ui_x);
        const y = mapY(node.ui_y);
        
        // Chọn icon dựa trên tên đỉnh
        let icon = "🏢"; // Hình tòa nhà mặc định
        if (node.id.includes("Cổng trường")) icon = "⛩️"; 
        else if (node.id.includes("Nhà xe")) icon = "🚙";
        else if (node.id.includes("Nhà thể dục")) icon = "🏟️";
        else if (node.id.includes("Nhà điều hành")) icon = "🏛️"; 
        else if (node.id.includes("ATM") || node.id.includes("Cây ATM")) icon = "🏧";
        else if (node.id.includes("Canteen") || node.id.includes("Căn tin")) icon = "🍽️";

        // Vẽ Icon (Emoji)
        ctx.font = "18px Arial"; 
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(icon, x, y);
        
        // Vẽ tên đỉnh đầy đủ, không bị cắt
        ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(node.id, x, y + 12); // Dời chữ xuống dưới icon một chút
    });

    // Vẽ điểm báo lộ trình (Các ô vuông xanh lá)
    if (currentRouteWaypoints.length > 0) {
        for (let i = 0; i < currentRouteWaypoints.length; i++) {
            const nodeId = currentRouteWaypoints[i];
            const node = nodeDict[nodeId];
            if (!node) continue;
            
            const x = mapX(node.ui_x);
            const y = mapY(node.ui_y);
            
            ctx.fillStyle = '#10b981';
            ctx.fillRect(x - 6, y - 6, 12, 12);
            
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 11px monospace';
            ctx.textAlign = "left";
            ctx.textBaseline = "bottom";
            ctx.fillText("📍 " + nodeId.replace(/_/g, " "), x + 8, y - 8);
        }
    }

    // Vẽ điểm GPS hiện tại nếu có (Tracking mô phỏng)
    if (typeof currentGPSPos !== 'undefined' && currentGPSPos) {
        const x = mapX(currentGPSPos.ui_x);
        const y = mapY(currentGPSPos.ui_y);
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#ef4444'; // Red dot
        ctx.fill();
        
        // Hiệu ứng chớp (ping)
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Kích hoạt tìm đường
async function calculateRoute() {
    const startNode = document.getElementById("route-start").value;
    const endNode = document.getElementById("route-end").value;
    const waypointNode = document.getElementById("route-waypoint") ? document.getElementById("route-waypoint").value : "";
    const weather = document.getElementById("route-weather") ? document.getElementById("route-weather").value : "normal";
    const preference = document.getElementById("route-preference") ? document.getElementById("route-preference").value : "fastest";
    
    if (!startNode || !endNode) {
        showToast("Vui lòng chọn 2 điểm!", "error");
        return;
    }

    let waypoints = startNode;
    if (waypointNode) waypoints += `,${waypointNode}`;
    waypoints += `,${endNode}`;

    try {
        const res = await fetch(`${BACKEND_URL}/map/api_get_route?waypoints=${encodeURIComponent(waypoints)}&weather=${weather}&preference=${preference}`);
        const data = await res.json();

        document.getElementById("route-result").classList.remove("hidden");

        if (data.status === "success") {
            currentRouteWaypoints = data.path;
            drawMap(); 
            
            // Tự động đóng bảng điều khiển trên mobile để xem bản đồ
            const container = document.getElementById("panels-container");
            if (container && container.classList.contains("flex") && window.innerWidth < 768) {
                toggleMobilePanels();
            } 
            
            document.getElementById("route-path-text").innerHTML = data.path.map(n => n.replace(/_/g, " ")).join(" <br>↓<br> ");
            document.getElementById("route-distance").innerText = `📏 Quãng đường: ${data.total_distance_m}m (${data.estimated_mins} phút)`;
            showToast("Đã vẽ lộ trình!", "success");

            // Đẩy lịch sử lên Firebase
            const token = localStorage.getItem("token");
            if (token) {
                fetch(`${BACKEND_URL}/history/location`, {
                    method: "POST",
                    headers: { 
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${token}`
                    },
                    body: JSON.stringify({ start_node: startNode, end_node: endNode })
                }).catch(e => console.error("History save error", e));
            }
        } else {
            currentRouteWaypoints = [];
            drawMap();
            document.getElementById("route-path-text").innerHTML = `<span class="text-rose-400">Không tìm thấy đường đi.</span>`;
            document.getElementById("route-distance").innerText = "";
        }
    } catch (e) {
        console.error("calculateRoute error:", e);
        currentRouteWaypoints = [];
        drawMap();
        document.getElementById("route-path-text").innerHTML = `<span class="text-rose-400">Lỗi kết nối máy chủ.</span>`;
        document.getElementById("route-distance").innerText = "";
        showToast("Lỗi hệ thống Router", "error");
    }
}

// Chạy mô phỏng GPS
async function startGPSTracking(simulate = false) {
    if (!currentRouteWaypoints || currentRouteWaypoints.length < 2) {
        showToast("Vui lòng tính toán lộ trình trước khi mô phỏng GPS", "error");
        return;
    }
    
    if (gpsSimulationInterval) clearInterval(gpsSimulationInterval);

    if (simulate) {
        const endNode = currentRouteWaypoints[currentRouteWaypoints.length - 1];
        const weather = document.getElementById("route-weather") ? document.getElementById("route-weather").value : "normal";
        const preference = document.getElementById("route-preference") ? document.getElementById("route-preference").value : "fastest";
        const session_id = document.getElementById("login-email") ? document.getElementById("login-email").value : "default";

        let pathIndex = 0;
        
        gpsSimulationInterval = setInterval(async () => {
            if (pathIndex >= currentRouteWaypoints.length) {
                clearInterval(gpsSimulationInterval);
                showToast("Đã đến đích!", "success");
                document.getElementById("gps-dist").innerText = "0m";
                document.getElementById("gps-time").innerText = "0 phút";
                currentGPSPos = null;
                drawMap();
                return;
            }

            const currentNodeId = currentRouteWaypoints[pathIndex];
            const node = graphData.nodes.find(n => n.id === currentNodeId);
            
            if (node) {
                // Giả lập GPS tại node này
                currentGPSPos = { ui_x: node.ui_x, ui_y: node.ui_y, lat: node.gps[0], lon: node.gps[1] };
                drawMap();

                document.getElementById("gps-coords").innerText = `${node.gps[0].toFixed(5)}, ${node.gps[1].toFixed(5)}`;

                // Gọi API backend
                try {
                    const url = `${BACKEND_URL}/map/api_realtime_tracking?current_lat=${node.gps[0]}&current_lon=${node.gps[1]}&end=${endNode}&weather=${weather}&preference=${preference}&session_id=${session_id}`;
                    const res = await fetch(url);
                    const data = await res.json();
                    
                    if (data.status === "tracking") {
                        // Handle the bug "snroutered_node" from backend replacement if it exists
                        const nearestNode = data.snapped_node || data.snroutered_node || "";
                        document.getElementById("gps-nearest").innerText = nearestNode.replace(/_/g, " ");
                        document.getElementById("gps-dist").innerText = `${data.total_remaining_meters}m`;
                        document.getElementById("gps-time").innerText = `${data.estimated_mins} phút`;
                        
                        // Xử lý geofencing alerts
                        if (data.geofence_alerts && data.geofence_alerts.length > 0) {
                            showToast(`⚠️ ${data.geofence_alerts[0].message}`, "error");
                        }
                        
                        // Cập nhật AI Recommend Box
                        if (data.route_suggestions && data.route_suggestions.length > 0) {
                            const listEl = document.getElementById("ai-recs-list");
                            if(listEl) {
                                listEl.innerHTML = "";
                                data.route_suggestions.forEach(rec => {
                                    listEl.innerHTML += `<li><span class="text-purple-300 font-bold">${rec.node.replace(/_/g, " ")}</span> (${rec.score})</li>`;
                                });
                                document.getElementById("ai-recs-box").classList.remove("hidden");
                            }
                        }
                    } else if (data.status === "arrived") {
                        document.getElementById("gps-nearest").innerText = data.message;
                    }
                } catch(e) {
                    console.error("Lỗi gọi GPS API", e);
                }
            }
            pathIndex++;
        }, 3000); // 3s qua 1 node
        
        showToast("Bắt đầu mô phỏng GPS...", "success");
    } else {
        const endNode = currentRouteWaypoints[currentRouteWaypoints.length - 1];
        const weather = document.getElementById("route-weather") ? document.getElementById("route-weather").value : "normal";
        const preference = document.getElementById("route-preference") ? document.getElementById("route-preference").value : "fastest";
        const session_id = document.getElementById("login-email") ? document.getElementById("login-email").value : "default";

        if ("geolocation" in navigator) {
            showToast("Đang kết nối GPS thiết bị...", "success");
            const watchId = navigator.geolocation.watchPosition(async (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                document.getElementById("gps-coords").innerText = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
                
                try {
                    const url = `${BACKEND_URL}/map/api_realtime_tracking?current_lat=${lat}&current_lon=${lon}&end=${endNode}&weather=${weather}&preference=${preference}&session_id=${session_id}`;
                    const res = await fetch(url);
                    const data = await res.json();
                    
                    if (data.status === "tracking") {
                        const nearestNode = data.snapped_node || data.snroutered_node || "";
                        document.getElementById("gps-nearest").innerText = nearestNode.replace(/_/g, " ");
                        document.getElementById("gps-dist").innerText = `${data.total_remaining_meters}m`;
                        document.getElementById("gps-time").innerText = `${data.estimated_mins} phút`;
                        
                        // Cập nhật currentGPSPos dựa trên nearestNode để vẽ lên bản đồ
                        const node = graphData.nodes.find(n => n.id === nearestNode);
                        if (node) {
                            currentGPSPos = { ui_x: node.ui_x, ui_y: node.ui_y, lat: lat, lon: lon };
                            drawMap();
                        }
                    } else if (data.status === "arrived") {
                        document.getElementById("gps-nearest").innerText = data.message;
                        navigator.geolocation.clearWatch(watchId);
                        showToast("Đã đến đích!", "success");
                        document.getElementById("gps-dist").innerText = "0m";
                        document.getElementById("gps-time").innerText = "0 phút";
                        currentGPSPos = null;
                        drawMap();
                    }
                } catch(e) {
                    console.error("Lỗi gọi GPS API", e);
                }
            }, (error) => {
                showToast("Lỗi GPS: " + error.message, "error");
            }, { enableHighAccuracy: true });
            
            // Lưu watchId nếu cần
            window.gpsRealInterval = watchId;
        } else {
            showToast("Trình duyệt không hỗ trợ GPS", "error");
        }
    }
}

// Lấy đề xuất AI
async function fetchAIRecommendations() {
    try {
        const session_id = document.getElementById("login-email") ? document.getElementById("login-email").value : "default"; 
        
        let targetLat = 10.878000;
        let targetLon = 106.798750;
        if (currentGPSPos) {
            targetLat = currentGPSPos.lat;
            targetLon = currentGPSPos.lon;
        }
        
        const res = await fetch(`${BACKEND_URL}/map/api_recommend?current_lat=${targetLat}&current_lon=${targetLon}&session_id=${session_id}`);
        const data = await res.json();
        
        if(data.status === "success") {
            const listEl = document.getElementById("ai-recs-list");
            listEl.innerHTML = "";
            data.recommendations.forEach(rec => {
                listEl.innerHTML += `<li><span class="text-purple-300 font-bold">${rec.node}</span> (Điểm tin cậy: ${rec.score})</li>`;
            });
            document.getElementById("ai-recs-box").classList.remove("hidden");
            
            const tagsEl = document.getElementById("ai-intent-tags");
            tagsEl.innerHTML = "";
            const INTEREST_LABELS = {
                "hoc_tap": "Học tập & Nghiên cứu 📚",
                "an_uong": "Ăn uống & Căn tin 🍽️",
                "nghi_ngoi": "Nghỉ ngơi & Thư giãn 😴",
                "the_thao": "Thể thao & Vận động 🏃",
                "cntt": "CNTT & Lập trình 💻",
                "tien_ich": "Tiện ích campus 🏪",
                "giai_tri": "Sự kiện & Giải trí 🎪",
                "hanh_chinh": "Hành chính & Giao vụ 🏛️"
            };

            if (data.user_profile && data.user_profile.interests && data.user_profile.interests.length > 0) {
                data.user_profile.interests.forEach(interest => {
                    const displayLabel = INTEREST_LABELS[interest] || interest;
                    tagsEl.innerHTML += `<span class="bg-blue-900/50 text-blue-300 px-2 py-1 rounded border border-blue-700/50 mr-1">${displayLabel}</span>`;
                });
            } else {
                tagsEl.innerHTML = `<span class="bg-blue-900/50 text-blue-300 px-2 py-1 rounded border border-blue-700/50">Chưa có dữ liệu học</span>`;
            }

            showToast("Đã phân tích hồ sơ AI thành công", "success");
        }
    } catch(e) {
        showToast("Lỗi phân tích hành vi", "error");
    }
}

// Huấn luyện model
async function trainModels() {
    const btn = document.getElementById("btn-train");
    btn.innerHTML = "⏳ Đang tối ưu hóa PyTorch...";
    btn.classList.replace("bg-emerald-600", "bg-slate-600");
    
    try {
        const res = await fetch(`${BACKEND_URL}/map/api_train_models`, {method: "POST"});
        const data = await res.json();
        
        if (data.status === "success") {
            showToast(`Huấn luyện hoàn tất! MAE: ${data.stats.crowd_mae}`, "success");
        }
    } catch (e) {
        showToast("Lỗi khi huấn luyện mô hình", "error");
    } finally {
        btn.innerHTML = "🔬 Huấn luyện lại Mạng PyTorch";
        btn.classList.replace("bg-slate-600", "bg-emerald-600");
    }
}

function toggleMapMode() {
    const mapLayer = document.getElementById("map-layer");
    const arLayer = document.getElementById("ar-layer");
    const placeholder = document.getElementById("viewport-placeholder");
    const btn = document.getElementById("btn-map");
    const canvasCont = document.getElementById("canvas-container");
    const viewport = document.getElementById("viewport-container");

    arLayer.classList.add("hidden");
    document.getElementById("btn-ar").classList.remove("bg-indigo-600", "border-indigo-400");

    if (mapLayer.classList.contains("hidden")) {
        mapLayer.classList.remove("hidden");
        canvasCont.classList.remove("hidden");
        viewport.classList.replace("bg-black", "bg-slate-950");
        placeholder.classList.add("opacity-0", "pointer-events-none");
        btn.classList.add("bg-blue-600", "border-blue-400");
        
        const gnnPanel = document.getElementById("gnn-panel");
        const aiPanel = document.getElementById("ai-panel");
        const gpsPanel = document.getElementById("gps-panel");
        if(gnnPanel) gnnPanel.classList.remove("hidden");
        if(aiPanel) aiPanel.classList.remove("hidden");
        if(gpsPanel) gpsPanel.classList.remove("hidden");

        fetchCampusNodes(); 
    } else {
        mapLayer.classList.add("hidden");
        if (arLayer.classList.contains("hidden")) {
            canvasCont.classList.add("hidden");
            placeholder.classList.remove("opacity-0", "pointer-events-none");
        }
        btn.classList.remove("bg-blue-600", "border-blue-400");

        const gnnPanel = document.getElementById("gnn-panel");
        const aiPanel = document.getElementById("ai-panel");
        const gpsPanel = document.getElementById("gps-panel");
        if(gnnPanel) gnnPanel.classList.add("hidden");
        if(aiPanel) aiPanel.classList.add("hidden");
        if(gpsPanel) gpsPanel.classList.add("hidden");
    }

    const isHome = document.getElementById("map-layer").classList.contains("hidden") && document.getElementById("ar-layer").classList.contains("hidden");
    if (isHome && window.innerWidth >= 768) {
        minimizeChat('maximize');
    } else {
        minimizeChat('minimize');
    }
}

function toggleARMode() {
    const mapLayer = document.getElementById("map-layer");
    const arLayer = document.getElementById("ar-layer");
    const placeholder = document.getElementById("viewport-placeholder");
    const btn = document.getElementById("btn-ar");
    const canvasCont = document.getElementById("canvas-container");
    const viewport = document.getElementById("viewport-container");

    mapLayer.classList.add("hidden");
    document.getElementById("btn-map").classList.remove("bg-blue-600", "border-blue-400");

    const gnnPanel = document.getElementById("gnn-panel");
    const aiPanel = document.getElementById("ai-panel");
    const gpsPanel = document.getElementById("gps-panel");
    if(gnnPanel) gnnPanel.classList.add("hidden");
    if(aiPanel) aiPanel.classList.add("hidden");
    if(gpsPanel) gpsPanel.classList.add("hidden");

    if (arLayer.classList.contains("hidden")) {
        arLayer.classList.remove("hidden");
        canvasCont.classList.add("hidden"); // Ẩn 3D map của GNN
        viewport.classList.replace("bg-slate-950", "bg-black");
        placeholder.classList.add("opacity-0", "pointer-events-none");
        btn.classList.add("bg-indigo-600", "border-indigo-400");
        
        if (typeof initLocalMap === 'function') {
            initLocalMap();
        }
        
        fetchCampusNodes(); // To ensure graphData is loaded
        // transmitWebSocketPayload({ action: "ar_nav", payload: "start_camera_tracking" }); // Bỏ cái websocket tracking cũ này đi vì AR đã xử lý local
    } else {
        arLayer.classList.add("hidden");
        if (mapLayer.classList.contains("hidden")) {
            canvasCont.classList.add("hidden");
            placeholder.classList.remove("opacity-0", "pointer-events-none");
        }
        viewport.classList.replace("bg-black", "bg-slate-950");
        btn.classList.remove("bg-indigo-600", "border-indigo-400");
    }

    const isHome = document.getElementById("map-layer").classList.contains("hidden") && document.getElementById("ar-layer").classList.contains("hidden");
    if (isHome && window.innerWidth >= 768) {
        minimizeChat('maximize');
    } else {
        minimizeChat('minimize');
    }
}

// Lấy lịch sử ghé thăm của người dùng
async function showUserHistory() {
    try {
        const token = localStorage.getItem("token");
        if (!token) {
            showToast("Vui lòng đăng nhập để xem lịch sử", "error");
            return;
        }

        const res = await fetch(`${BACKEND_URL}/history/user`, {
            headers: { "Authorization": `Bearer ${token}` }
        });
        const data = await res.json();
        
        if (data.status === "success") {
            const listEl = document.getElementById("history-list");
            listEl.innerHTML = "";
            
            if (!data.locations || data.locations.length === 0) {
                listEl.innerHTML = `<p class="text-sm text-slate-400 text-center py-4">Chưa có dữ liệu di chuyển</p>`;
            } else {
                data.locations.forEach(loc => {
                    const start = loc.StartNode.replace(/_/g, " ");
                    const end = loc.EndNode.replace(/_/g, " ");
                    const date = new Date(loc.Timestamp).toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
                    listEl.innerHTML += `
                        <div class="flex flex-col bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
                            <div class="flex justify-between items-center mb-1">
                                <span class="text-xs text-slate-400">${date}</span>
                                <span class="text-[10px] bg-emerald-600/20 text-emerald-400 px-2 py-0.5 rounded border border-emerald-500/20">Đã đi</span>
                            </div>
                            <div class="flex items-center gap-2 text-sm text-slate-200 font-medium font-mono">
                                <span>${start}</span>
                                <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3"></path></svg>
                                <span>${end}</span>
                            </div>
                        </div>
                    `;
                });
            }
            document.getElementById("history-modal").classList.remove("hidden");
        } else {
            showToast("Lỗi: " + data.detail, "error");
        }
    } catch (e) {
        showToast("Lỗi lấy lịch sử", "error");
    }
}

// Xóa lịch sử ghé thăm
async function clearUserHistory() {
    try {
        const emailInput = document.getElementById("login-email");
        const session_id = (emailInput && emailInput.value) ? emailInput.value : "default";
        // Call backend api_user_profile with reset_history=true as POST
        const res = await fetch(`${BACKEND_URL}/map/api_user_profile?session_id=${session_id}&reset_history=true`, { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            showToast("Đã xóa lịch sử!", "success");
            showUserHistory(); // Refresh modal
        }
    } catch (e) {
        showToast("Lỗi xóa lịch sử", "error");
    }
}

// Node Image Modal logic (GNN Map Visualization)
document.getElementById('canvas-container').addEventListener('click', (e) => {
    
    // Bỏ qua nếu người dùng vừa mới kéo bản đồ xong (tránh lỗi click nhầm khi đang kéo)
    if (window.lastClickDownX && window.lastClickDownY) {
        if (Math.abs(e.clientX - window.lastClickDownX) > 5 || Math.abs(e.clientY - window.lastClickDownY) > 5) return;
    }
    
    if (!graphData || !graphData.nodes) return;
    
    const rect = document.getElementById('canvas-container').getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    let closestNode = null;
    let minDist = Infinity;
    
    const padding = 50;
    const drawW = canvas.width - padding * 2;
    const drawH = canvas.height - padding * 2;
    
    graphData.nodes.forEach(n => {
        const nx = padding + (n.ui_x / 100) * drawW;
        const ny = padding + (n.ui_y / 100) * drawH;
        
        const mapX = nx * mapScale + mapOffsetX;
        const mapY = ny * mapScale + mapOffsetY;
        
        const dist = Math.hypot(mapX - x, mapY - y);
        if (dist < minDist) {
            minDist = dist;
            closestNode = n;
        }
    });
    
    // Bán kính click (nhạy hơn một chút để dễ bấm trên điện thoại)
    if (minDist < 40 && closestNode) {
        showNodeImage(closestNode);
    }
});

function showNodeImage(node) {
    console.log("Showing node image for:", node);
    document.getElementById('node-image-title').innerText = node.id.replace(/_/g, " ");
    const iconEl = document.getElementById('node-image-icon');
    const imgEl = document.getElementById('node-image-img');
    
    if (node.image_url) {
        iconEl.classList.add('hidden');
        imgEl.classList.remove('hidden');
        imgEl.src = node.image_url.replace('/static/images/', 'img/');
    } else {
        iconEl.classList.remove('hidden');
        imgEl.classList.add('hidden');
        
        let icon = "🏢";
        if (node.id.includes("Cổng trường")) icon = "⛩️"; 
        else if (node.id.includes("Nhà xe")) icon = "🚙";
        else if (node.id.includes("Nhà thể dục")) icon = "🏟️";
        else if (node.id.includes("Nhà điều hành")) icon = "🏛️";
        else if (node.id.includes("Cây ATM") || node.id.includes("ATM")) icon = "🏧";
        else if (node.id.includes("Canteen") || node.id.includes("Căn tin")) icon = "🍽️";
        
        iconEl.innerText = icon;
    }
    document.getElementById('node-image-modal').classList.remove('hidden');
}

// ==========================================
// TIKTOK SWIPE EXPLORE UI
// ==========================================

let tiktokRecommendations = [];
let currentTiktokIndex = 0;

async function openTiktokExplore() {
    // Hiện loading
    document.getElementById('tiktok-modal').classList.remove('hidden');
    document.getElementById('tiktok-loading').classList.remove('hidden');
    document.getElementById('tiktok-card-container').classList.add('opacity-0');
    
    try {
        if (graphData && graphData.nodes && graphData.nodes.length > 0) {
            // Lấy tất cả các địa điểm và xáo trộn (shuffle)
            let allNodes = [...graphData.nodes];
            for (let i = allNodes.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [allNodes[i], allNodes[j]] = [allNodes[j], allNodes[i]];
            }
            
            tiktokRecommendations = allNodes.map(n => {
                let ams = [];
                if (n.features) {
                    if (n.features.parking) ams.push("Bãi xe");
                    if (n.features.wc) ams.push("WC");
                    if (n.features.water) ams.push("Nước uống");
                    if (n.features.elevator) ams.push("Thang máy");
                }
                
                return {
                    node: n.id,
                    tagline: n.tagline || 'Khu vực trong khuôn viên trường',
                    function_summary: n.function_summary || '',
                    image_url: n.image_url || '',
                    amenities: ams,
                    score: 100
                };
            });
            
            currentTiktokIndex = 0;
            renderTiktokCard();
            
            document.getElementById('tiktok-loading').classList.add('hidden');
            document.getElementById('tiktok-card-container').classList.remove('opacity-0');
            
            const emptyState = document.getElementById('tiktok-empty-state');
            if (emptyState) emptyState.classList.add('hidden');
        } else {
            // Hiển thị giao diện rỗng thay vì đóng ngay lập tức
            document.getElementById('tiktok-loading').classList.add('hidden');
            document.getElementById('tiktok-card-container').classList.remove('opacity-0');
            
            const emptyState = document.getElementById('tiktok-empty-state');
            if (emptyState) emptyState.classList.remove('hidden');
        }
    } catch (error) {
        console.error(error);
        showToast('Lỗi khi tải Tiktok Explore', 'error');
        closeTiktokExplore();
    }
}

function closeTiktokExplore() {
    document.getElementById('tiktok-modal').classList.add('hidden');
    tiktokRecommendations = [];
}

function renderTiktokCard() {
    if (currentTiktokIndex >= tiktokRecommendations.length) {
        showToast('Đã xem hết gợi ý. Đang tải thêm...');
        closeTiktokExplore();
        return;
    }
    
    const rec = tiktokRecommendations[currentTiktokIndex];
    const imgEl = document.getElementById('tiktok-img');
    const titleEl = document.getElementById('tiktok-title');
    const taglineEl = document.getElementById('tiktok-tagline');
    const scoreEl = document.getElementById('tiktok-score');
    const amenitiesEl = document.getElementById('tiktok-amenities');
    
    // Reset animation
    imgEl.classList.add('opacity-0', 'scale-110');
    
    setTimeout(() => {
        imgEl.src = rec.image_url ? rec.image_url.replace('/static/images/', 'img/') : 'https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&w=800';
        imgEl.onload = () => {
            imgEl.classList.remove('opacity-0', 'scale-110');
        };
        imgEl.onerror = () => {
            imgEl.src = 'https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&w=800';
            imgEl.classList.remove('opacity-0', 'scale-110');
        };
        
        titleEl.textContent = rec.node.replace(/_/g, ' ');
        taglineEl.textContent = rec.tagline || rec.function_summary || rec.reason || 'Khu vực trong khuôn viên trường';
        scoreEl.textContent = `Độ phù hợp: ${Math.round(rec.score)}%`;
        
        amenitiesEl.innerHTML = '';
        if (rec.amenities && rec.amenities.length > 0) {
            rec.amenities.forEach(a => {
                amenitiesEl.innerHTML += `<span class="bg-white/20 text-white text-[9px] px-2 py-0.5 rounded backdrop-blur-md">${a}</span>`;
            });
        }
        
        // Cập nhật số liệu (tym, bình luận, đánh giá)
        loadTiktokNodeStats(rec.node);
        
    }, 300);
}

let currentTiktokUserRating = 0; // Lưu tạm rating của user hiện tại

async function loadTiktokNodeStats(node) {
    const emailInput = document.getElementById('login-email');
    const session_id = (emailInput && emailInput.value) ? emailInput.value : 'default';
    try {
        const res = await fetch(`${BACKEND_URL}/map/api_get_node_stats?node_id=${node}&session_id=${session_id}`);
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('tiktok-likes-count').textContent = data.likes_count;
            document.getElementById('tiktok-comments-count').textContent = data.comments_count;
            
            // Cập nhật số sao trung bình và số lượt đánh giá
            const ratingText = data.ratings_count > 0 ? `${data.average_rating}⭐ (${data.ratings_count})` : "Đ.Giá";
            document.getElementById('tiktok-ratings-count').textContent = ratingText;
            
            // Lưu rating của user cho modal
            currentTiktokUserRating = data.user_rating || 0;
            
            const btnLike = document.getElementById("tiktok-like-btn");
            if (btnLike) {
                if (data.has_liked) {
                    btnLike.classList.add("bg-rose-500", "border-rose-500");
                    btnLike.classList.remove("bg-slate-800/80");
                } else {
                    btnLike.classList.remove("bg-rose-500", "border-rose-500");
                    btnLike.classList.add("bg-slate-800/80");
                }
            }
        }
    } catch(e) {
        console.error("Lỗi khi tải thống kê:", e);
    }
}

function swipeTiktok(direction) {
    const card = document.getElementById('tiktok-card-container');
    
    if (direction === 'left') {
        // Bỏ qua
        card.style.transform = 'translateX(-100%) rotate(-10deg)';
        card.style.opacity = '0';
    } else {
        // Đến đây
        card.style.transform = 'translateX(100%) rotate(10deg)';
        card.style.opacity = '0';
        
        const rec = tiktokRecommendations[currentTiktokIndex];
        
        // Setup lộ trình
        const startSel = document.getElementById('route-start');
        const endSel = document.getElementById('route-end');
        if (startSel && endSel) {
            endSel.value = rec.node;
            // Tự động tìm vị trí gần nhất để set làm start
            let minDist = Infinity;
            let closestNode = null;
            if (currentGPSPos && graphData && graphData.nodes) {
                graphData.nodes.forEach(n => {
                    const dist = Math.hypot(n.gps[0] - currentGPSPos.lat, n.gps[1] - currentGPSPos.lon);
                    if (dist < minDist) {
                        minDist = dist;
                        closestNode = n.id;
                    }
                });
            }
            if (closestNode) startSel.value = closestNode;
            
            closeTiktokExplore();
            setTimeout(calculateRoute, 500);
        }
    }
    
    setTimeout(() => {
        card.style.transition = 'none';
        card.style.transform = 'translateX(0) rotate(0)';
        currentTiktokIndex++;
        renderTiktokCard();
        
        setTimeout(() => {
            card.style.transition = 'all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)';
            card.style.opacity = '1';
        }, 50);
    }, 400);
}

function toggleMobilePanels() {
    const container = document.getElementById("panels-container");
    if (container) {
        container.classList.toggle("hidden");
        container.classList.toggle("flex");
    }
}

// ==========================================
// TIKTOK ACTION LOGIC (Like, Comment, Rating)
// ==========================================

async function likeTiktokNode() {
    const emailInput = document.getElementById('login-email');
    const session_id = (emailInput && emailInput.value) ? emailInput.value : 'default';
    if (tiktokRecommendations.length === 0 || currentTiktokIndex >= tiktokRecommendations.length) return;
    const node = tiktokRecommendations[currentTiktokIndex].node;
    
    try {
        const url = `${BACKEND_URL}/map/api_toggle_like?node_id=${node}&session_id=${session_id}`;
        const res = await fetch(url, { method: "POST" });
        const data = await res.json();
        
        if (data.status === 'success') {
            // Cập nhật thống kê ngay lập tức
            loadTiktokNodeStats(node);
            
            // Hiện hiệu ứng tim bay nếu vừa thả tim
            if (data.has_liked) {
                const burst = document.getElementById('heart-burst');
                if (burst) {
                    const heart = document.createElement('div');
                    heart.innerHTML = '💖';
                    heart.className = 'absolute inset-0 flex items-center justify-center text-xl animate-ping opacity-0';
                    burst.appendChild(heart);
                    setTimeout(() => heart.remove(), 1000);
                }
            }
        } else {
            showToast("Lỗi khi thả tim", "error");
        }
    } catch(e) {
        showToast("Lỗi kết nối khi thả tim", "error");
    }
}

function openTiktokComments() {
    if (tiktokRecommendations.length === 0 || currentTiktokIndex >= tiktokRecommendations.length) return;
    const node = tiktokRecommendations[currentTiktokIndex].node;
    const modal = document.getElementById("tiktok-comments-modal");
    if (modal) {
        modal.classList.remove("hidden");
        // Trigger reflow
        void modal.offsetWidth;
        modal.classList.remove("translate-y-full");
    }
    loadTiktokComments(node);
}

function closeTiktokComments() {
    const modal = document.getElementById("tiktok-comments-modal");
    if (modal) {
        modal.classList.add("translate-y-full");
        setTimeout(() => {
            modal.classList.add("hidden");
        }, 300);
    }
}

async function loadTiktokComments(node) {
    const listEl = document.getElementById("tiktok-comments-list");
    if (!listEl) return;
    listEl.innerHTML = `<div class="text-center text-slate-400 mt-4">Đang tải bình luận...</div>`;
    try {
        const url = `${BACKEND_URL}/map/api_get_comments?node_id=${node}`;
        const res = await fetch(url);
        const data = await res.json();
        
        listEl.innerHTML = "";
        if (data.status === "success" && data.comments.length > 0) {
            data.comments.forEach(c => {
                const date = new Date(c.timestamp * 1000).toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
                listEl.innerHTML += `
                    <div class="bg-slate-800/50 p-3 rounded-xl border border-slate-700/50">
                        <div class="flex justify-between items-center mb-1">
                            <span class="text-xs text-blue-400 font-bold">${c.session_id.split('@')[0]}</span>
                            <span class="text-[10px] text-slate-500">${date}</span>
                        </div>
                        <p class="text-sm text-slate-200">${c.content}</p>
                    </div>
                `;
            });
        } else {
            listEl.innerHTML = `<div class="text-center text-slate-400 text-xs mt-4">Chưa có bình luận nào. Hãy là người đầu tiên!</div>`;
        }
    } catch(e) {
        listEl.innerHTML = `<div class="text-center text-rose-400 text-xs mt-4">Lỗi khi tải bình luận.</div>`;
    }
}

async function submitTiktokComment() {
    const emailInput = document.getElementById('login-email');
    const session_id = (emailInput && emailInput.value) ? emailInput.value : 'default';
    if (tiktokRecommendations.length === 0 || currentTiktokIndex >= tiktokRecommendations.length) return;
    const node = tiktokRecommendations[currentTiktokIndex].node;
    
    const input = document.getElementById("tiktok-comment-input");
    const content = input.value.trim();
    if (!content) {
        showToast("Vui lòng nhập bình luận", "error");
        return;
    }
    
    try {
        const url = `${BACKEND_URL}/map/api_submit_comment?node_id=${node}&content=${encodeURIComponent(content)}&session_id=${session_id}`;
        const res = await fetch(url, { method: "POST" });
        if (res.ok) {
            input.value = "";
            showToast("Đã gửi bình luận!", "success");
            loadTiktokComments(node);
            loadTiktokNodeStats(node);
        } else {
            showToast("Lỗi khi gửi bình luận", "error");
        }
    } catch(e) {
        showToast("Lỗi kết nối", "error");
    }
}

function openTiktokRating() {
    if (tiktokRecommendations.length === 0 || currentTiktokIndex >= tiktokRecommendations.length) return;
    const modal = document.getElementById("tiktok-rating-modal");
    if (modal) modal.classList.remove("hidden");
    
    const stars = document.querySelectorAll(".tiktok-star");
    stars.forEach((s, idx) => {
        if (idx < currentTiktokUserRating) {
            s.classList.add("text-yellow-400");
        } else {
            s.classList.remove("text-yellow-400");
        }
    });
}

function closeTiktokRating() {
    const modal = document.getElementById("tiktok-rating-modal");
    if (modal) modal.classList.add("hidden");
}

async function setTiktokRating(rating) {
    const emailInput = document.getElementById('login-email');
    const session_id = (emailInput && emailInput.value) ? emailInput.value : 'default';
    if (tiktokRecommendations.length === 0 || currentTiktokIndex >= tiktokRecommendations.length) return;
    const node = tiktokRecommendations[currentTiktokIndex].node;
    
    const stars = document.querySelectorAll(".tiktok-star");
    stars.forEach((s, idx) => {
        if (idx < rating) {
            s.classList.add("text-yellow-400");
        } else {
            s.classList.remove("text-yellow-400");
        }
    });
    
    try {
        const url = `${BACKEND_URL}/map/api_submit_rating?node_id=${node}&rating=${rating}&session_id=${session_id}`;
        const res = await fetch(url, { method: "POST" });
        if (res.ok) {
            showToast(`Đã đánh giá ${rating} sao!`, "success");
            loadTiktokNodeStats(node);
            setTimeout(closeTiktokRating, 500);
        } else {
            showToast("Lỗi khi đánh giá", "error");
        }
    } catch(e) {
        showToast("Lỗi kết nối", "error");
    }
}
window.addEventListener('DOMContentLoaded', () => { if (window.innerWidth < 768) { minimizeChat('minimize'); } });
