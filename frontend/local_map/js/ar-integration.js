// frontend/local_map/js/ar-integration.js

let localMap = null;
let mapMarkers = {};
let arBuildings = [];
let selectedDestination = null;

let userGPS = null;
let currentHeading = 0; // Mặc định 0 độ (Hướng Bắc) để hỗ trợ fallback cho Laptop không có La bàn
let currentBeta = 90;
let arActive = false;
let arCanvas, arCtx, arVideo;
let animFrame = 0;

const HFOV = 65;
const VFOV = 50;

// Initialize Leaflet Map
async function initLocalMap() {
    if (localMap) return; // Already initialized

    // Setup map container
    const container = document.getElementById("local-map-container");
    localMap = L.map(container, { zoomControl: false }).setView([10.876, 106.7975], 16);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 20,
        attribution: '© OpenStreetMap contributors'
    }).addTo(localMap);

    // Khoanh vùng HCMUS
    const hcmusBounds = [
        [10.8778, 106.7958],
        [10.8783, 106.7996],
        [10.8753, 106.7999],
        [10.8746, 106.7984],
        [10.8734, 106.7973],
        [10.8736, 106.7960]
    ];
    L.polygon(hcmusBounds, {color: '#4f46e5', weight: 3, fillOpacity: 0.1}).addTo(localMap);

    // Fetch buildings
    try {
        const res = await fetch(`${typeof BACKEND_URL !== 'undefined' ? BACKEND_URL : 'http://127.0.0.1:8000'}/local_map/api/buildings`);
        const data = await res.json();
        if (data.success && data.buildings) {
            arBuildings = data.buildings;
            arBuildings.forEach(b => {
                if (!b.lat || !b.lng) return;
                const marker = L.circleMarker([b.lat, b.lng], {
                    radius: 12, 
                    color: b.color || '#4f46e5',
                    fillColor: b.color || '#4f46e5', 
                    fillOpacity: 0.8, 
                    weight: 2,
                }).addTo(localMap);

                marker.bindPopup(`
                    <div class="text-center p-1">
                        <div class="text-2xl mb-1">${b.icon || '🏢'}</div>
                        <h3 class="font-bold text-slate-800 text-sm">${b.name}</h3>
                        <button onclick="setARDestination('${b.id}')" class="mt-2 bg-indigo-600 text-white text-xs px-3 py-1.5 rounded-lg w-full">📍 Chọn làm điểm đến</button>
                    </div>
                `);
                mapMarkers[b.id] = marker;
            });
        }
    } catch (e) {
        console.error("Error loading local map data:", e);
    }
}

// Set destination for AR
window.setARDestination = function(id) {
    selectedDestination = arBuildings.find(b => b.id === id);
    localMap.closePopup();
    
    // Update launch button text
    const btn = document.getElementById("btn-launch-ar");
    btn.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7"></path></svg>
        DẪN ĐƯỜNG ĐẾN: ${selectedDestination.name}
    `;
    btn.classList.replace("bg-indigo-600", "bg-emerald-600");
    btn.classList.replace("border-indigo-400", "border-emerald-400");
};

// Start AR Camera
document.getElementById("btn-launch-ar").addEventListener("click", async () => {
    document.getElementById("local-map-container").classList.add("hidden");
    document.getElementById("ar-launch-overlay").classList.add("hidden");
    
    arVideo = document.getElementById("ar-video");
    arCanvas = document.getElementById("ar-canvas");
    arCanvas.width = window.innerWidth;
    arCanvas.height = window.innerHeight;
    arCtx = arCanvas.getContext("2d");

    arVideo.classList.remove("hidden");
    arCanvas.classList.remove("hidden");
    document.getElementById("ar-top-hud").classList.remove("hidden");
    document.getElementById("ar-bottom-panel").classList.remove("hidden");
    
    // Ẩn Chatbot và các nút điều hướng để tránh vướng víu
    const chatWin = document.getElementById("chat-window");
    if (chatWin) chatWin.classList.add("hidden");
    const bottomNav = document.getElementById("bottom-nav-buttons");
    if (bottomNav) bottomNav.classList.add("hidden");

    arActive = true;
    updateARDestinationUI(); // Bật giao diện chọn điểm đến
    
    // Start Camera
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        arVideo.srcObject = stream;
    } catch (e) {
        alert("Không thể mở Camera!");
    }

    // Start Compass
    if (window.DeviceOrientationEvent) {
        if (typeof DeviceOrientationEvent.requestPermission === 'function') {
            try {
                const perm = await DeviceOrientationEvent.requestPermission();
                if (perm === 'granted') attachCompass();
            } catch(e){}
        } else {
            attachCompass();
        }
    }

    // Start GPS
    if (navigator.geolocation) {
        navigator.geolocation.watchPosition(
            (pos) => {
                userGPS = { lat: pos.coords.latitude, lng: pos.coords.longitude };
                document.getElementById("ar-gps-pill").innerHTML = `<span>📍 ±${Math.round(pos.coords.accuracy)}m</span>`;
            },
            (err) => {},
            { enableHighAccuracy: true }
        );
    }

    requestAnimationFrame(renderARFrame);
});

// Close AR
document.getElementById("btn-close-ar").addEventListener("click", () => {
    arActive = false;
    if (arVideo && arVideo.srcObject) {
        arVideo.srcObject.getTracks().forEach(t => t.stop());
    }
    
    arVideo.classList.add("hidden");
    arCanvas.classList.add("hidden");
    document.getElementById("ar-top-hud").classList.add("hidden");
    document.getElementById("ar-bottom-panel").classList.add("hidden");
    document.getElementById("ar-destination-selector").classList.add("hidden");
    document.getElementById("ar-selected-card").classList.add("hidden");
    
    // Hiện lại Chatbot và nút điều hướng
    const chatWin = document.getElementById("chat-window");
    if (chatWin) chatWin.classList.remove("hidden");
    const bottomNav = document.getElementById("bottom-nav-buttons");
    if (bottomNav) bottomNav.classList.remove("hidden");

    document.getElementById("local-map-container").classList.remove("hidden");
    document.getElementById("ar-launch-overlay").classList.remove("hidden");
    
    // Reset selected dest
    selectedDestination = null;
    const btn = document.getElementById("btn-launch-ar");
    btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg> AR CAMERA`;
    btn.classList.replace("bg-emerald-600", "bg-indigo-600");
    btn.classList.replace("border-emerald-400", "border-indigo-400");
});

function attachCompass() {
    window.addEventListener('deviceorientationabsolute', (e) => {
        if (e.alpha !== null) currentHeading = (360 - e.alpha) % 360;
        if (e.beta !== null) currentBeta = e.beta;
        updateCompassUI();
    }, true);
    
    window.addEventListener('deviceorientation', (e) => {
        if (e.webkitCompassHeading !== undefined) currentHeading = e.webkitCompassHeading;
        else if (e.alpha !== null && currentHeading === null) currentHeading = (360 - e.alpha) % 360;
        if (e.beta !== null) currentBeta = e.beta;
        updateCompassUI();
    }, true);
}

function updateCompassUI() {
    if (currentHeading !== null) {
        document.getElementById("ar-compass-pill").innerHTML = `<span>🧭 ${Math.round(currentHeading)}°</span>`;
    }
}

// Logic giao diện chọn điểm đến
function updateARDestinationUI() {
    const selector = document.getElementById("ar-destination-selector");
    const card = document.getElementById("ar-selected-card");
    if (!arActive) return;

    if (!selectedDestination) {
        card.classList.add("hidden");
        selector.classList.remove("hidden");
        populateARDestinations();
    } else {
        selector.classList.add("hidden");
        card.classList.remove("hidden");
        
        const b = selectedDestination;
        document.getElementById("ar-card-name").innerText = b.name;
        const iconElem = document.getElementById("ar-card-icon");
        iconElem.innerText = b.icon || "🏢";
        iconElem.className = `w-11 h-11 rounded-xl flex items-center justify-center text-white text-xl shadow-inner ${b.color || 'bg-blue-600'}`;
    }
}

function populateARDestinations() {
    const list = document.getElementById("ar-destination-list");
    if (!list) return;
    list.innerHTML = "";
    
    let buildings = [...arBuildings];
    if (userGPS) {
        buildings.forEach(b => {
            b.currentDist = haversine(userGPS.lat, userGPS.lng, b.lat, b.lng);
        });
        buildings.sort((a, b) => a.currentDist - b.currentDist);
    }

    buildings.forEach(b => {
        const distText = b.currentDist ? `${Math.round(b.currentDist)}m` : "";
        const chip = document.createElement("button");
        chip.className = "flex items-center gap-2 bg-slate-800/80 hover:bg-slate-700/90 border border-slate-600/50 backdrop-blur-md px-4 py-2.5 rounded-2xl whitespace-nowrap transition-all shadow-lg snap-center shrink-0";
        chip.onclick = () => {
            selectedDestination = b;
            updateARDestinationUI();
        };
        chip.innerHTML = `
            <span class="text-base">${b.icon || '🏢'}</span>
            <span class="text-white font-medium text-sm">${b.name}</span>
            ${distText ? `<span class="text-slate-400 text-xs bg-slate-900/50 px-2 py-0.5 rounded-md ml-1">${distText}</span>` : ''}
        `;
        list.appendChild(chip);
    });
}

window.clearARDestination = function() {
    selectedDestination = null;
    updateARDestinationUI();
};



// Math Helpers
function haversine(lat1, lng1, lat2, lng2) {
    const R = 6371000;
    const toRad = d => d * Math.PI / 180;
    const a = Math.sin(toRad(lat2 - lat1)/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(toRad(lng2 - lng1)/2)**2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function getBearing(lat1, lng1, lat2, lng2) {
    const toRad = d => d * Math.PI / 180;
    const y = Math.sin(toRad(lng2 - lng1)) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(toRad(lng2 - lng1));
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function drawNavArrow(ctx, W, H, relativeAngle, dist) {
    // Nếu đến nơi
    if (dist < 5) {
        ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
        ctx.beginPath();
        ctx.roundRect(W/2 - 100, H/2 - 30, 200, 60, 15);
        ctx.fill();
        ctx.fillStyle = "white";
        ctx.font = "bold 20px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("🎉 Đã tới nơi!", W/2, H/2);
        return;
    }

    ctx.save();
    ctx.translate(W/2, H/2 + 100);
    
    // Rotate arrow based on relativeAngle
    // If angle is ~0, arrow points UP
    ctx.rotate(relativeAngle * Math.PI / 180);

    // Bounce animation
    const bounce = Math.sin(animFrame / 10) * 10;
    ctx.translate(0, bounce);

    // Draw Arrow (Cyan Triangle - Hiếu's Style)
    ctx.beginPath();
    ctx.moveTo(0, -80);
    ctx.lineTo(40, 40);
    ctx.lineTo(0, 20); // Vết khuyết ở đuôi mũi tên
    ctx.lineTo(-40, 40);
    ctx.closePath();
    
    // Gradient Cyan
    const grad = ctx.createLinearGradient(0, -80, 0, 40);
    grad.addColorStop(0, "rgba(6, 182, 212, 0.9)"); // Cyan 500
    grad.addColorStop(1, "rgba(8, 145, 178, 0.4)"); // Cyan 600
    
    ctx.fillStyle = grad;
    ctx.shadowColor = "#06b6d4";
    ctx.shadowBlur = 25;
    ctx.fill();
    
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(255, 255, 255, 1)";
    ctx.stroke();
    ctx.restore();
    
    // Text Guide
    ctx.fillStyle = "white";
    ctx.font = "bold 16px sans-serif";
    ctx.textAlign = "center";
    ctx.shadowColor = "black";
    ctx.shadowBlur = 10;
    let text = "Đi thẳng";
    if (relativeAngle > 20 && relativeAngle < 160) text = "Rẽ phải";
    else if (relativeAngle < -20 && relativeAngle > -160) text = "Rẽ trái";
    else if (Math.abs(relativeAngle) >= 160) text = "Quay lui";
    ctx.fillText(`${text} (${Math.round(dist)}m)`, W/2, H/2 + 200);
}

function renderARFrame() {
    if (!arActive) return;
    requestAnimationFrame(renderARFrame);
    animFrame++;

    const W = arCanvas.width;
    const H = arCanvas.height;
    arCtx.clearRect(0, 0, W, H);

    if (!userGPS) {
        arCtx.fillStyle = "white";
        arCtx.font = "14px sans-serif";
        arCtx.textAlign = "center";
        arCtx.fillText("Đang chờ định vị GPS...", W/2, H/2);
        return;
    }

    // Horizon line
    const clampedBeta = Math.max(30, Math.min(150, currentBeta));
    const horizonY = H / 2 + ((clampedBeta - 90) / (VFOV / 2)) * (H / 2);

    // Draw Buildings
    arBuildings.forEach(b => {
        const dist = haversine(userGPS.lat, userGPS.lng, b.lat, b.lng);
        if (dist > 2000) return; // Hide if too far

        const bearing = getBearing(userGPS.lat, userGPS.lng, b.lat, b.lng);
        let hAngle = bearing - currentHeading;
        while (hAngle > 180) hAngle -= 360;
        while (hAngle < -180) hAngle += 360;

        if (Math.abs(hAngle) < HFOV * 0.7) {
            const x = W / 2 + (hAngle / (HFOV / 2)) * (W / 2);
            const scale = Math.max(0.5, Math.min(1.2, 50 / Math.sqrt(dist)));
            const y = horizonY - 50 * scale;

            // Draw label
            arCtx.fillStyle = "rgba(15, 23, 42, 0.8)";
            arCtx.beginPath();
            arCtx.roundRect(x - 60*scale, y - 20*scale, 120*scale, 40*scale, 8);
            arCtx.fill();
            arCtx.strokeStyle = b.color || "#6366f1";
            arCtx.lineWidth = 2;
            arCtx.stroke();
            
            arCtx.fillStyle = "white";
            arCtx.font = `bold ${12*scale}px sans-serif`;
            arCtx.textAlign = "center";
            arCtx.textBaseline = "middle";
            arCtx.fillText(`${b.icon || '🏢'} ${b.name}`, x, y);
            arCtx.font = `${10*scale}px sans-serif`;
            arCtx.fillStyle = "#94a3b8";
            arCtx.fillText(`${Math.round(dist)}m`, x, y + 14*scale);
        }
    });

    // Draw Navigation Arrow if destination is selected
    if (selectedDestination && userGPS) {
        const dist = haversine(userGPS.lat, userGPS.lng, selectedDestination.lat, selectedDestination.lng);
        
        // Cập nhật khoảng cách trên Card
        const distElem = document.getElementById("ar-card-dist");
        if (distElem) distElem.innerText = `📍 ${Math.round(dist)} m`;

        const bearing = getBearing(userGPS.lat, userGPS.lng, selectedDestination.lat, selectedDestination.lng);
        let relAngle = bearing - currentHeading;
        while (relAngle > 180) relAngle -= 360;
        while (relAngle < -180) relAngle += 360;
        
        drawNavArrow(arCtx, W, H, relAngle, dist);
    }
}
