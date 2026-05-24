/* ----------------------------------------------------
   SISTEMA DE DETECCIÓN DE OJOS - LÓGICA DE CLIENTE (JS)
   ---------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Elementos del DOM
    const video = document.getElementById("webcam-stream");
    const resultOverlayContainer = document.getElementById("result-overlay-container");
    const resultImage = document.getElementById("result-image");
    const viewportOverlayMsg = document.getElementById("viewport-overlay-msg");
    const overlayLoader = document.getElementById("overlay-loader");
    const overlayText = document.getElementById("overlay-text");
    const timelineProgressBar = document.getElementById("timeline-progress-bar");
    const timeIndicator = document.getElementById("time-indicator");
    const btnResumeLoop = document.getElementById("btn-resume-loop");
    
    // Acciones e Inputs
    const btnManualCapture = document.getElementById("btn-manual-capture");
    const imageImportInput = document.getElementById("image-import-input");
    
    // Sliders de Configuración
    const inputThreshold = document.getElementById("input-threshold");
    const valThreshold = document.getElementById("val-threshold");
    const inputLowThreshold = document.getElementById("input-low-threshold");
    const valLowThreshold = document.getElementById("val-low-threshold");
    
    // Estadísticas
    const statMaxScore = document.getElementById("stat-max-score");
    const statMaxPercent = document.getElementById("stat-max-percent");
    const statRadialProgress = document.getElementById("stat-radial-progress");
    const statVerdictBanner = document.getElementById("stat-verdict-banner");
    const statVerdictDesc = document.getElementById("stat-verdict-desc");
    const systemStatusText = document.getElementById("system-status-text");
    const systemStatusBadge = document.getElementById("system-status-badge");
    const systemStatusDot = systemStatusBadge.querySelector(".status-dot");
    
    // Indicadores de Leyenda
    const legendGreen = document.getElementById("legend-green-indicator");
    const legendRed = document.getElementById("legend-red-indicator");
    const legendGray = document.getElementById("legend-gray-indicator");

    // 2. Configuración del Estado de la Aplicación
    const STATE = {
        INIT: "INIT",
        CAPTURING: "CAPTURING",
        PROCESSING: "PROCESSING",
        SHOWING_RESULT: "SHOWING_RESULT",
        ERROR: "ERROR"
    };

    let currentState = STATE.INIT;
    let webcamStream = null;
    let loopInterval = null;
    let timerRemaining = 0;
    const CAPTURE_INTERVAL_MS = 5000;
    const SHOW_RESULT_DURATION_MS = 10000;
    const TICK_RATE_MS = 100;
    
    // 3. Inicialización y Gestión de la Webcam
    async function initWebcam() {
        setOverlay(true, "Solicitando acceso a la cámara...", true);
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            });
            video.srcObject = webcamStream;
            
            video.onloadedmetadata = () => {
                setOverlay(false);
                transitionTo(STATE.CAPTURING);
            };
        } catch (error) {
            console.error("Error al acceder a la webcam:", error);
            setOverlay(true, "Cámara no disponible. Puede importar imágenes locales.", false);
            setSystemStatus("Cámara Desconectada", "orange");
            transitionTo(STATE.ERROR);
        }
    }

    // 4. Máquina de Estados
    function transitionTo(newState) {
        console.log(`Transición de estado: ${currentState} -> ${newState}`);
        currentState = newState;
        
        // Limpiar intervalos de tiempo activos
        if (loopInterval) {
            clearInterval(loopInterval);
            loopInterval = null;
        }

        switch (currentState) {
            case STATE.CAPTURING:
                // Mostrar Video, Ocultar Resultado
                video.classList.add("active");
                resultOverlayContainer.classList.remove("active");
                btnResumeLoop.style.display = "none";
                setSystemStatus("Cámara Activa", "green");
                setOverlay(false);
                
                // Iniciar cuenta atrás de 7 segundos para captura
                startTimelapseTimer(CAPTURE_INTERVAL_MS);
                break;
                
            case STATE.PROCESSING:
                setOverlay(true, "Analizando imagen con IA...", true);
                setSystemStatus("Procesando...", "orange");
                // La barra de progreso se detiene en 100% temporalmente
                timelineProgressBar.style.width = "100%";
                timelineProgressBar.style.background = "var(--color-primary)";
                break;
                
            case STATE.SHOWING_RESULT:
                // Ocultar Video, Mostrar Resultado Anotado
                video.classList.remove("active");
                resultOverlayContainer.classList.add("active");
                btnResumeLoop.style.display = "block";
                setSystemStatus("Detección Mostrada", "green");
                setOverlay(false);
                
                // Iniciar cuenta de congelamiento de 10 segundos
                startDisplayTimer(SHOW_RESULT_DURATION_MS);
                break;
                
            case STATE.ERROR:
                // Se detienen las capturas automáticas, pero se permite importar archivos
                timelineProgressBar.style.width = "0%";
                timeIndicator.innerHTML = "Timelapse detenido. Importe una imagen para analizar.";
                btnResumeLoop.style.display = "none";
                break;
        }
    }

    // 5. Controladores de Tiempo (Timers)
    function startTimelapseTimer(duration) {
        timerRemaining = duration;
        updateTimelineProgress(100);
        
        loopInterval = setInterval(() => {
            timerRemaining -= TICK_RATE_MS;
            const percent = (timerRemaining / duration) * 100;
            updateTimelineProgress(percent);
            
            const seconds = (timerRemaining / 1000).toFixed(1);
            timeIndicator.innerHTML = `Siguiente captura automática en: <strong>${seconds}s</strong>`;
            
            if (timerRemaining <= 0) {
                clearInterval(loopInterval);
                captureFrameAndDetect();
            }
        }, TICK_RATE_MS);
    }

    function startDisplayTimer(duration) {
        timerRemaining = duration;
        updateTimelineProgress(0); // Empezamos en 0% para que se llene
        
        loopInterval = setInterval(() => {
            timerRemaining -= TICK_RATE_MS;
            const percent = ((duration - timerRemaining) / duration) * 100;
            updateTimelineProgress(percent, true); // Pasar bandera de que es congelamiento
            
            const seconds = (timerRemaining / 1000).toFixed(1);
            timeIndicator.innerHTML = `Retornando a cámara en vivo: <strong>${seconds}s</strong>`;
            
            if (timerRemaining <= 0) {
                clearInterval(loopInterval);
                resumeTimelapse();
            }
        }, TICK_RATE_MS);
    }

    function updateTimelineProgress(percent, isFrozen = false) {
        const p = Math.max(0, Math.min(100, percent));
        timelineProgressBar.style.width = `${p}%`;
        
        if (isFrozen) {
            // Animación de llenado en verde/esmeralda
            timelineProgressBar.style.background = "linear-gradient(90deg, var(--color-success), #34d399)";
        } else {
            // Animación de vaciado en azul
            timelineProgressBar.style.background = "linear-gradient(90deg, var(--color-primary), #60a5fa)";
        }
    }

    function resumeTimelapse() {
        if (webcamStream) {
            transitionTo(STATE.CAPTURING);
        } else {
            transitionTo(STATE.ERROR);
        }
    }

    // 6. Lógica de Captura e Inferencia
    function captureFrameAndDetect() {
        if (currentState === STATE.PROCESSING) return;

        // Verificar que el video tenga dimensiones válidas antes de capturar
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            console.warn("El video aún no tiene dimensiones válidas.");
            return;
        }

        transitionTo(STATE.PROCESSING);
        
        // Dibujar el frame actual en un Canvas oculto
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        
        const ctx = canvas.getContext("2d");
        
        // Espejamos el canvas horizontalmente al dibujar porque el video está espejado en CSS
        // pero queremos evaluar la imagen tal como la ve la IA originalmente.
        // Espera, si dibujamos sin espejar, ¿coincide con el lado izquierdo/derecho real?
        // Dibujamos de forma natural tal como entra del stream (sin espejo) para que el modelo ONNX reciba la orientación real
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Exportar a JPEG Base64
        const base64Image = canvas.toDataURL("image/jpeg", 0.85);
        
        // Enviar al servidor
        sendDetectionRequest({
            image: base64Image,
            threshold: parseFloat(inputThreshold.value),
            low_threshold: parseFloat(inputLowThreshold.value)
        });
    }

    async function sendDetectionRequest(payload) {
        try {
            const response = await fetch("http://127.0.0.1:5000/detect", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Error en servidor: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.success) {
                // Actualizar imagen y estadísticas
                resultImage.src = data.image_base64;
                updateStats(data);
                transitionTo(STATE.SHOWING_RESULT);
            } else {
                showErrorOverlay(`Error de detección: ${data.error}`);
            }
        } catch (error) {
            console.error("Error al procesar detección:", error);
            showErrorOverlay("No se pudo conectar con el servidor de IA.");
        }
    }

    // Procesar importación de archivos (usa FileReader + base64, igual que la captura de cámara)
    function handleFileImport(file) {
        if (!file) return;

        // Asegurarse de cancelar cualquier intervalo residual
        if (loopInterval) {
            clearInterval(loopInterval);
            loopInterval = null;
        }

        // Verificar que sea una imagen
        if (!file.type.startsWith("image/")) {
            alert("Por favor selecciona un archivo de imagen válido.");
            return;
        }

        transitionTo(STATE.PROCESSING);

        const reader = new FileReader();

        reader.onload = function(evt) {
            const base64Image = evt.target.result; // data:image/xxx;base64,...
            // Enviar igual que la captura de cámara (JSON + base64)
            sendDetectionRequest({
                image: base64Image,
                threshold: parseFloat(inputThreshold.value),
                low_threshold: parseFloat(inputLowThreshold.value)
            });
        };

        reader.onerror = function() {
            showErrorOverlay("No se pudo leer el archivo de imagen.");
        };

        reader.readAsDataURL(file);
    }

    // 7. Actualización del Dashboard UI
    function updateStats(data) {
        const score = data.max_score;
        const percent = (score * 100).toFixed(1);
        
        // 1. Actualizar valores numéricos
        statMaxScore.innerText = score.toFixed(4);
        statMaxPercent.innerText = `${percent}%`;
        
        // 2. Actualizar indicador radial SVG (Circunferencia = 100)
        statRadialProgress.setAttribute("stroke-dasharray", `${percent}, 100`);
        
        // Cambiar el color de la barra radial según la detección
        if (data.verdict === "Ojo Detectado") {
            statRadialProgress.setAttribute("stroke", "var(--color-success)");
            statRadialProgress.style.filter = "drop-shadow(0 0 4px var(--color-success))";
        } else if (data.verdict === "Coincidencia Baja") {
            statRadialProgress.setAttribute("stroke", "var(--color-danger)");
            statRadialProgress.style.filter = "drop-shadow(0 0 4px var(--color-danger))";
        } else {
            statRadialProgress.setAttribute("stroke", "var(--text-muted)");
            statRadialProgress.style.filter = "none";
        }
        
        // 3. Actualizar Banner de Veredicto
        statVerdictBanner.className = "verdict-banner"; // reset
        statVerdictDesc.innerText = data.verdict;
        
        if (data.verdict === "Ojo Detectado") {
            statVerdictBanner.classList.add("success");
            activateLegendGlow("green");
        } else if (data.verdict === "Coincidencia Baja") {
            statVerdictBanner.classList.add("danger");
            activateLegendGlow("red");
        } else {
            statVerdictBanner.classList.add("neutral");
            activateLegendGlow("gray");
        }
        
        // 4. Actualizar Barra de Confianza Lineal Premium
        const statLinearPercent = document.getElementById("stat-linear-percent");
        const statLinearBar = document.getElementById("stat-linear-bar");
        
        if (statLinearPercent && statLinearBar) {
            statLinearPercent.innerText = `${percent}%`;
            statLinearBar.style.width = `${percent}%`;
            
            // Cambiar color y efectos visuales de la barra lineal según veredicto
            if (data.verdict === "Ojo Detectado") {
                statLinearBar.style.background = "linear-gradient(90deg, var(--color-primary), var(--color-success))";
                statLinearPercent.style.color = "var(--color-success)";
                statLinearBar.style.boxShadow = "0 0 10px rgba(16, 185, 129, 0.4)";
            } else if (data.verdict === "Coincidencia Baja") {
                statLinearBar.style.background = "linear-gradient(90deg, var(--color-warning), var(--color-danger))";
                statLinearPercent.style.color = "var(--color-danger)";
                statLinearBar.style.boxShadow = "0 0 10px rgba(239, 68, 68, 0.4)";
            } else {
                statLinearBar.style.background = "var(--text-muted)";
                statLinearPercent.style.color = "var(--text-muted)";
                statLinearBar.style.boxShadow = "none";
            }
        }
    }

    function activateLegendGlow(activeType) {
        // Remover clases de brillo
        legendGreen.style.transform = "none";
        legendRed.style.transform = "none";
        legendGray.style.transform = "none";
        
        legendGreen.parentElement.style.opacity = "0.5";
        legendRed.parentElement.style.opacity = "0.5";
        legendGray.parentElement.style.opacity = "0.5";
        
        if (activeType === "green") {
            legendGreen.style.transform = "scale(1.2)";
            legendGreen.parentElement.style.opacity = "1";
        } else if (activeType === "red") {
            legendRed.style.transform = "scale(1.2)";
            legendRed.parentElement.style.opacity = "1";
        } else if (activeType === "gray") {
            legendGray.style.transform = "scale(1.2)";
            legendGray.parentElement.style.opacity = "1";
        }
    }

    // 8. Elementos Auxiliares de Interfaz
    function setOverlay(show, text = "", showSpinner = false) {
        if (show) {
            viewportOverlayMsg.style.display = "flex";
            overlayLoader.style.display = showSpinner ? "block" : "none";
            overlayText.innerText = text;
        } else {
            viewportOverlayMsg.style.display = "none";
        }
    }

    function showErrorOverlay(msg) {
        setOverlay(true, msg, false);
        setSystemStatus("Error del Sistema", "red");
        transitionTo(STATE.ERROR);
    }

    function setSystemStatus(text, colorClass) {
        systemStatusText.innerText = text;
        systemStatusDot.className = `status-dot ${colorClass}`;
    }

    // 9. Event Listeners
    // Sliders
    inputThreshold.addEventListener("input", (e) => {
        valThreshold.innerText = parseFloat(e.target.value).toFixed(2);
    });
    
    inputLowThreshold.addEventListener("input", (e) => {
        valLowThreshold.innerText = parseFloat(e.target.value).toFixed(4);
    });

    // Botones
    btnManualCapture.addEventListener("click", () => {
        if (currentState === STATE.CAPTURING) {
            clearInterval(loopInterval);
            captureFrameAndDetect();
        }
    });

    btnResumeLoop.addEventListener("click", () => {
        if (currentState === STATE.SHOWING_RESULT) {
            clearInterval(loopInterval);
            resumeTimelapse();
        }
    });

    // Importación de imagen
    imageImportInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            // Cancelar cualquier timer activo para que no interfiera con la importación
            if (loopInterval) {
                clearInterval(loopInterval);
                loopInterval = null;
            }
            handleFileImport(file);
            // Reiniciar el input para permitir recargar la misma imagen
            e.target.value = "";
        }
    });

    // Inicializar
    initWebcam();
});
