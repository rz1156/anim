const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('overlayCanvas');
const canvasCtx = canvasElement.getContext('2d');
const btnCamera = document.getElementById('btnCamera');
const loadingIndicator = document.getElementById('loadingIndicator');
const modeVal = document.getElementById('modeVal');
const fpsVal = document.getElementById('fpsVal');
const viewportWrapper = document.getElementById('viewportWrapper');
const htmlOverlayContainer = document.getElementById('htmlOverlayContainer');

let faceMesh;
let camera;
let particles = [];
let sakuraParticles = [];
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// Tracker state variables for multi-frame conditions
let blinkCount = 0;
let lastBlinkTime = 0;
let eyesClosedStartTime = null;
let currentMode = "Standard";

// Setup global Canvas scaling parameters
function resizeCanvas() {
    canvasElement.width = viewportWrapper.clientWidth;
    canvasElement.height = viewportWrapper.clientHeight;
}
window.addEventListener('resize', resizeCanvas);

// --- Initialization & MediaPipe Engine ---
function initFaceMesh() {
    faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6
    });

    faceMesh.onResults(onFaceResults);
    loadingIndicator.style.opacity = '0';
    setTimeout(() => loadingIndicator.style.display = 'none', 500);
}

// Global System Particle Blueprint
class Particle {
    constructor(x, y, type, color, size, vx, vy, life) {
        this.x = x; this.y = y; this.type = type; this.color = color;
        this.size = size; this.vx = vx; this.vy = vy;
        this.maxLife = life; this.life = life;
        this.angle = Math.random() * Math.PI * 2;
        this.rotSpeed = (Math.random() - 0.5) * 0.1;
    }
    update() {
        this.x += this.vx; this.y += this.vy; this.life--; this.angle += this.rotSpeed;
        if(this.type === 'sakura') { this.x += Math.sin(this.life * 0.02) * 0.5; }
    }
}

// Generate continuous global sakura falling animation
function emitGlobalSakura() {
    if (sakuraParticles.length < 40) {
        sakuraParticles.push(new Particle(
            Math.random() * canvasElement.width, -10, 'sakura',
            'rgba(255, 183, 197, 0.75)', Math.random() * 6 + 4,
            (Math.random() - 0.2) * 1.5, Math.random() * 1 + 1, 300
        ));
    }
}

// --- Mathematical helpers for Geometry Analysis ---
function getDistance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

// Core Tracking pipeline output analysis
function onFaceResults(results) {
    // Performance Benchmark (FPS)
    const now = performance.now();
    frameCount++;
    if (now - lastFrameTime >= 1000) {
        fps = frameCount; frameCount = 0; lastFrameTime = now;
        fpsVal.innerText = fps;
    }

    // Clean overlay for fresh operational frame
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    htmlOverlayContainer.innerHTML = '';
    viewportWrapper.style.transform = 'scale(1)';
    viewportWrapper.style.filter = 'none';

    // Global background elements rendering
    emitGlobalSakura();
    sakuraParticles = sakuraParticles.filter(p => p.life > 0);
    sakuraParticles.forEach(p => {
        p.update();
        canvasCtx.save();
        canvasCtx.translate(p.x, p.y);
        canvasCtx.rotate(p.angle);
        canvasCtx.fillStyle = p.color;
        canvasCtx.beginPath();
        canvasCtx.ellipse(0, 0, p.size, p.size / 2, 0, 0, Math.PI * 2);
        canvasCtx.fill();
        canvasCtx.restore();
    });

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        currentMode = "Standard";
        modeVal.innerText = currentMode;
        return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    const W = canvasElement.width;
    const H = canvasElement.height;

    // Mapping relevant extreme tracking indices
    const leftEye = landmarks[33], rightEye = landmarks[263];
    const noseTip = landmarks[4];
    const chin = landmarks[152], forehead = landmarks[10];

    // Compute transformation matrix properties dynamically
    const faceWidth = getDistance(leftEye, rightEye) * W;
    const scaleFactor = faceWidth / 110; 
    const centerPoint = { x: noseTip.x * W, y: noseTip.y * H };
    
    // Calculate Head Roll (Rotation Angle)
    const rollAngle = Math.atan2((rightEye.y - leftEye.y) * H, (rightEye.x - leftEye.x) * W);
    // Pitch approximation (Nodding / Tilting down)
    const totalFaceHeight = getDistance(forehead, chin);
    const noseToChin = getDistance(noseTip, chin);
    const pitchRatio = noseToChin / totalFaceHeight; 

    // --- Face Morphometrics Mapping (Expressions) ---
    const lipTop = landmarks[13], lipBottom = landmarks[14];
    const mouthLeft = landmarks[61], mouthRight = landmarks[291];
    const mouthOpenDist = getDistance(lipTop, lipBottom);
    const mouthWidth = getDistance(mouthLeft, mouthRight);
    
    // Feature extractors ratios
    const mouthOpenRatio = mouthOpenDist / mouthWidth;
    const smileRatio = mouthWidth / (getDistance(landmarks[50], landmarks[280]));

    // Eye openness extraction
    const leftEyeTop = landmarks[159], leftEyeBottom = landmarks[145];
    const rightEyeTop = landmarks[386], rightEyeBottom = landmarks[374];
    const leftEAR = getDistance(leftEyeTop, leftEyeBottom) / getDistance(landmarks[33], landmarks[133]);
    const rightEAR = getDistance(rightEyeTop, rightEyeBottom) / getDistance(landmarks[263], landmarks[362]);
    const averageEAR = (leftEAR + rightEAR) / 2;

    // Eyebrow dynamics mapping
    const leftEyebrow = landmarks[70], rightEyebrow = landmarks[300];
    const browHeight = (getDistance(leftEyebrow, landmarks[159]) + getDistance(rightEyebrow, landmarks[386])) / 2;

    // Reset fallback flags safely
    let detectedMode = "Standard";

    // --- Rule Engine: Active Filter State Evaluation ---
    
    // 1. Sleep Mode Check (Both eyes completely closed)
    if (averageEAR < 0.15) {
        if (!eyesClosedStartTime) eyesClosedStartTime = now;
        if (now - eyesClosedStartTime > 2000) detectedMode = "Sleep";
    } else {
        // Evaluate Fast Double Blink Sequence (Magical Mode Trigger)
        if (eyesClosedStartTime && (now - eyesClosedStartTime < 300)) {
            if (now - lastBlinkTime < 600) {
                blinkCount++;
                if (blinkCount >= 2) { detectedMode = "Magical"; blinkCount = 0; }
            } else { blinkCount = 1; }
            lastBlinkTime = now;
        }
        eyesClosedStartTime = null;
    }

    // Secondary State machine cascade if not trapped in Sleep state
    if (detectedMode !== "Sleep" && detectedMode !== "Magical") {
        if (mouthOpenRatio > 0.6 && pitchRatio < 0.55) {
            detectedMode = "Power Up";
        } else if (browHeight > 0.22 && averageEAR > 0.35) {
            detectedMode = "Demon";
        } else if (mouthOpenRatio > 0.4) {
            detectedMode = "Surprise";
        } else if (pitchRatio > 0.62) {
            detectedMode = "Embarrassed";
        } else if (smileRatio > 1.15) {
            detectedMode = pitchRatio < 0.48 ? "Angel" : "Happy";
        } else if (smileRatio > 0.95) {
            detectedMode = "Neko";
        }
    }

    currentMode = detectedMode;
    modeVal.innerText = currentMode;

    // --- Render Pipeline Engine ---

    // Smooth Blush & Glow Systems
    if (smileRatio > 0.85 || currentMode === "Embarrassed") {
        let intensity = Math.min((smileRatio - 0.85) * 2.5, 0.8);
        if (currentMode === "Embarrassed") intensity = 0.9;

        canvasCtx.save();
        canvasCtx.translate(centerPoint.x, centerPoint.y);
        canvasCtx.rotate(rollAngle);
        
        canvasCtx.shadowBlur = 20 * scaleFactor;
        canvasCtx.shadowColor = `rgba(255, 60, 140, ${intensity})`;
        canvasCtx.fillStyle = `rgba(255, 105, 180, ${intensity * 0.5})`;

        // Draw left and right cheeks relative mapping
        canvasCtx.beginPath();
        canvasCtx.arc(-35 * scaleFactor, 20 * scaleFactor, 18 * scaleFactor, 0, Math.PI * 2);
        canvasCtx.arc(35 * scaleFactor, 20 * scaleFactor, 18 * scaleFactor, 0, Math.PI * 2);
        canvasCtx.fill();
        canvasCtx.restore();
    }

    // Sparkle Cross Eye Rendering
    if (averageEAR > 0.18) {
        const eyesToRender = [leftEye, rightEye];
        eyesToRender.forEach(eye => {
            canvasCtx.save();
            canvasCtx.translate(eye.x * W, eye.y * H);
            canvasCtx.rotate(rollAngle + (now * 0.003)); // Smooth Rotation over time
            canvasCtx.shadowBlur = 15;
            canvasCtx.shadowColor = "#00f0ff";
            canvasCtx.fillStyle = "#ffffff";
            
            // Draw cross-star anime sparks
            for (let i = 0; i < 2; i++) {
                canvasCtx.rotate(Math.PI / 4 * i);
                canvasCtx.fillRect(-10 * scaleFactor, -2 * scaleFactor, 20 * scaleFactor, 4 * scaleFactor);
                canvasCtx.fillRect(-2 * scaleFactor, -10 * scaleFactor, 4 * scaleFactor, 20 * scaleFactor);
            }
            canvasCtx.restore();
        });
    }

    // --- Mode Specific Dynamic Canvas Elements ---
    canvasCtx.save();
    canvasCtx.translate(centerPoint.x, centerPoint.y);
    canvasCtx.rotate(rollAngle);

    switch(currentMode) {
        case "Happy":
            viewportWrapper.style.filter = "brightness(1.15) contrast(1.05)";
            if(Math.random() < 0.2) {
                particles.push(new Particle(0, 0, 'heart', '#ff4081', Math.random()*4+4, (Math.random()-0.5)*4, -Math.random()*4-2, 60));
            }
            break;

        case "Surprise":
            viewportWrapper.style.transform = "scale(1.06)";
            const lines = document.createElement('div');
            lines.className = 'shock-lines';
            htmlOverlayContainer.appendChild(lines);

            const exclam = document.createElement('div');
            exclam.className = 'anime-exclamation';
            exclam.innerText = '!!';
            exclam.style.left = `${centerPoint.x}px`;
            exclam.style.top = `${centerPoint.y}px`;
            htmlOverlayContainer.appendChild(exclam);
            break;

        case "Embarrassed":
            // Anime vertical blue embarrassment lines
            canvasCtx.strokeStyle = 'rgba(0, 120, 255, 0.4)';
            canvasCtx.lineWidth = 3 * scaleFactor;
            for(let i=-3; i<=3; i++) {
                canvasCtx.beginPath();
                canvasCtx.moveTo(i * 15 * scaleFactor, -50 * scaleFactor);
                canvasCtx.lineTo(i * 15 * scaleFactor, -20 * scaleFactor);
                canvasCtx.stroke();
            }
            break;

        case "Sleep":
            viewportWrapper.style.filter = "brightness(0.65)";
            if (frameCount % 45 === 0) {
                const zText = document.createElement('div');
                zText.style.position = 'absolute';
                zText.style.color = '#76c7ff';
                zText.style.fontSize = '2rem';
                zText.style.fontWeight = 'bold';
                zText.style.left = `${centerPoint.x + (40 * scaleFactor)}px`;
                zText.style.top = `${centerPoint.y - (40 * scaleFactor)}px`;
                zText.style.animation = 'popIn 2.5s ease-out forwards';
                zText.innerText = 'Zzz...';
                htmlOverlayContainer.appendChild(zText);
            }
            break;

        case "Magical":
            // Spawning Rotating Runes & Glyphs
            canvasCtx.strokeStyle = 'rgba(0, 240, 255, 0.8)';
            canvasCtx.lineWidth = 4;
            canvasCtx.shadowBlur = 20;
            canvasCtx.shadowColor = '#00f0ff';
            canvasCtx.beginPath();
            canvasCtx.arc(0, 0, 90 * scaleFactor, 0, Math.PI*2);
            canvasCtx.stroke();
            
            canvasCtx.save();
            canvasCtx.rotate(-now * 0.002);
            canvasCtx.font = `${14 * scaleFactor}px monospace`;
            canvasCtx.fillStyle = '#00f0ff';
            canvasCtx.fillText("⚡ CYBER ⚡ MAGIC ⚡ SYSTEM", -70 * scaleFactor, 5 * scaleFactor);
            canvasCtx.restore();
            break;

        case "Power Up":
            // High intensity localized aura generator
            canvasCtx.shadowBlur = 35;
            canvasCtx.shadowColor = '#9d00ff';
            canvasCtx.fillStyle = 'rgba(157, 0, 255, 0.15)';
            canvasCtx.beginPath();
            canvasCtx.arc(0, -20 * scaleFactor, 110 * scaleFactor, 0, Math.PI*2);
            canvasCtx.fill();
            
            // Jittery raw energy lightning strikes
            if(Math.random() < 0.7) {
                canvasCtx.strokeStyle = '#ffffff';
                canvasCtx.lineWidth = 2 * scaleFactor;
                canvasCtx.beginPath();
                canvasCtx.moveTo((Math.random()-0.5)*100, (Math.random()-0.5)*100);
                canvasCtx.lineTo((Math.random()-0.5)*150, (Math.random()-0.5)*150);
                canvasCtx.stroke();
            }
            break;

        case "Demon":
            // Red glowing evil eyes & generated dark horns
            canvasCtx.fillStyle = '#ff003c';
            canvasCtx.shadowColor = '#ff0000';
            canvasCtx.shadowBlur = 15;
            
            // Left Horn
            canvasCtx.beginPath();
            canvasCtx.moveTo(-40 * scaleFactor, -60 * scaleFactor);
            canvasCtx.quadraticCurveTo(-70 * scaleFactor, -120 * scaleFactor, -90 * scaleFactor, -110 * scaleFactor);
            canvasCtx.quadraticCurveTo(-60 * scaleFactor, -80 * scaleFactor, -30 * scaleFactor, -55 * scaleFactor);
            canvasCtx.fill();

            // Right Horn
            canvasCtx.beginPath();
            canvasCtx.moveTo(40 * scaleFactor, -60 * scaleFactor);
            canvasCtx.quadraticCurveTo(70 * scaleFactor, -120 * scaleFactor, 90 * scaleFactor, -110 * scaleFactor);
            canvasCtx.quadraticCurveTo(60 * scaleFactor, -80 * scaleFactor, 30 * scaleFactor, -55 * scaleFactor);
            canvasCtx.fill();
            break;

        case "Angel":
            // Overhead Holy Halo Ring
            canvasCtx.strokeStyle = 'rgba(255, 245, 200, 0.9)';
            canvasCtx.shadowBlur = 25;
            canvasCtx.shadowColor = '#fff5c8';
            canvasCtx.lineWidth = 7 * scaleFactor;
            canvasCtx.save();
            canvasCtx.scale(1, 0.28); // Compress into isometric ring perspective
            canvasCtx.beginPath();
            canvasCtx.arc(0, -320 * scaleFactor, 75 * scaleFactor, 0, Math.PI*2);
            canvasCtx.stroke();
            canvasCtx.restore();
            break;

        case "Neko":
            // Cat ears geometry transformation mapping
            canvasCtx.fillStyle = '#120a2a';
            canvasCtx.strokeStyle = 'rgba(255, 42, 116, 0.7)';
            canvasCtx.lineWidth = 3;

            // Left Ear
            canvasCtx.beginPath();
            canvasCtx.moveTo(-55 * scaleFactor, -50 * scaleFactor);
            canvasCtx.lineTo(-85 * scaleFactor, -105 * scaleFactor);
            canvasCtx.lineTo(-25 * scaleFactor, -75 * scaleFactor);
            canvasCtx.closePath();
            canvasCtx.fill(); canvasCtx.stroke();

            // Right Ear
            canvasCtx.beginPath();
            canvasCtx.moveTo(55 * scaleFactor, -50 * scaleFactor);
            canvasCtx.lineTo(85 * scaleFactor, -105 * scaleFactor);
            canvasCtx.lineTo(25 * scaleFactor, -75 * scaleFactor);
            canvasCtx.closePath();
            canvasCtx.fill(); canvasCtx.stroke();
            break;
    }

    canvasCtx.restore();

    // Secondary Independent Localized Particle Array Loop
    particles = particles.filter(p => p.life > 0);
    particles.forEach(p => {
        p.update();
        canvasCtx.save();
        canvasCtx.fillStyle = p.color;
        if(p.type === 'heart') {
            canvasCtx.font = `${p.size * 2}px serif`;
            canvasCtx.fillText('❤', p.x + centerPoint.x, p.y + centerPoint.y);
        }
        canvasCtx.restore();
    });
}

// --- Device Hardware Controller ---
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        videoElement.srcObject = stream;
        
        camera = new Camera(videoElement, {
            onFrame: async () => {
                await faceMesh.send({ image: videoElement });
            },
            width: 640,
            height: 480
        });
        
        camera.start();
        btnCamera.style.display = 'none';
        resizeCanvas();
    } catch (err) {
        alert("Gagal mengakses kamera perangkat: " + err.message);
    }
}

// System Trigger Listeners
btnCamera.addEventListener('click', () => {
    loadingIndicator.style.display = 'flex';
    initFaceMesh();
    startCamera();
});
