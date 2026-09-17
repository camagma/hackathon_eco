/**
 * UpcycleAI Studio Frontend
 * =========================
 * Управление drag & drop старых вещей, генерацией через Qwen-Image-Edit,
 * отображением Before/After слайдера и пошаговой DIY-инструкцией мастера.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Элементы
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const dropPlaceholder = document.getElementById("dropPlaceholder");
  const previewContainer = document.getElementById("previewContainer");
  const sourcePreview = document.getElementById("sourcePreview");
  const sourceMeta = document.getElementById("sourceMeta");
  const clearImageBtn = document.getElementById("clearImageBtn");

  const promptInput = document.getElementById("promptInput");
  const clearPromptBtn = document.getElementById("clearPromptBtn");
  const presetsGrid = document.getElementById("presetsGrid");
  const sampleChipsRow = document.getElementById("sampleChipsRow");
  const styleTagsGrid = document.querySelector(".style-tags-grid");
  const generateBtn = document.getElementById("generateBtn");
  const generateBtnText = document.getElementById("generateBtnText");

  const viewerStage = document.getElementById("viewerStage");
  const emptyStage = document.getElementById("emptyStage");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingTimer = document.getElementById("loadingTimer");
  const loadingModelTag = document.getElementById("loadingModelTag");

  const compareContainer = document.getElementById("compareContainer");
  const compareWrapper = document.getElementById("compareWrapper");
  const compareOverlay = document.getElementById("compareOverlay");
  const sliderHandle = document.getElementById("sliderHandle");
  const beforeImage = document.getElementById("beforeImage");
  const afterImage = document.getElementById("afterImage");

  const viewModeToggle = document.getElementById("viewModeToggle");
  const downloadBtn = document.getElementById("downloadBtn");
  const chainEditBtn = document.getElementById("chainEditBtn");

  const viewerFooter = document.getElementById("viewerFooter");
  const metricModel = document.getElementById("metricModel");
  const metricLatency = document.getElementById("metricLatency");
  const metricSize = document.getElementById("metricSize");

  // Элементы DIY-карточки
  const diyCard = document.getElementById("diyCard");
  const diyItemTitle = document.getElementById("diyItemTitle");
  const diyItemSubtitle = document.getElementById("diyItemSubtitle");
  const diyDifficulty = document.getElementById("diyDifficulty");
  const diyTime = document.getElementById("diyTime");
  const diyCost = document.getElementById("diyCost");
  const diyEco = document.getElementById("diyEco");
  const diyMaterialsList = document.getElementById("diyMaterialsList");
  const diyToolsList = document.getElementById("diyToolsList");
  const diyStepsContainer = document.getElementById("diyStepsContainer");
  const diyProTipText = document.getElementById("diyProTipText");
  const copyDiyBtn = document.getElementById("copyDiyBtn");

  // Статус и настройки
  const apiStatusPill = document.getElementById("apiStatusPill");
  const apiStatusLabel = document.getElementById("apiStatusLabel");
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsModal = document.getElementById("settingsModal");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");
  const cancelSettingsBtn = document.getElementById("cancelSettingsBtn");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const settingApiKey = document.getElementById("settingApiKey");
  const toggleKeyVisibility = document.getElementById("toggleKeyVisibility");
  const settingBaseUrlSelect = document.getElementById("settingBaseUrlSelect");
  const settingCustomBaseUrl = document.getElementById("settingCustomBaseUrl");
  const settingModel = document.getElementById("settingModel");
  const settingSeed = document.getElementById("settingSeed");
  const settingSteps = document.getElementById("settingSteps");
  const toastContainer = document.getElementById("toastContainer");

  // Состояние приложения
  let currentFile = null;
  let currentImageDataUri = null;
  let lastResultDataUri = null;
  let currentStyleHint = null;
  let currentDiyPlan = null;
  let isDraggingSlider = false;
  let timerInterval = null;
  let startTime = 0;

  // Хранилище настроек
  const storage = {
    get: (k, def = "") => localStorage.getItem(`qwen_${k}`) || def,
    set: (k, v) => localStorage.setItem(`qwen_${k}`, v),
  };

  // Инициализация
  initSettings();
  checkApiHealth();
  loadPresets();
  loadSamples();
  setupDragAndDrop();
  setupStyleTags();
  setupSlider();
  setupEventListeners();

  // ==================== НАСТРОЙКИ ====================
  function initSettings() {
    settingApiKey.value = storage.get("api_key");
    let savedBaseUrl = storage.get("base_url", "https://api-inference.huggingface.co");
    if (savedBaseUrl === "https://integrate.api.nvidia.com/v1") {
      savedBaseUrl = "http://localhost:8000/v1";
      storage.set("base_url", savedBaseUrl);
    }
    if (savedBaseUrl === "http://localhost:8000/v1" || savedBaseUrl === "https://dashscope-intl.aliyuncs.com/api/v1" || savedBaseUrl === "https://api-inference.huggingface.co") {
      settingBaseUrlSelect.value = savedBaseUrl;
      settingCustomBaseUrl.classList.add("hidden");
    } else if (savedBaseUrl) {
      settingBaseUrlSelect.value = "custom";
      settingCustomBaseUrl.value = savedBaseUrl;
      settingCustomBaseUrl.classList.remove("hidden");
    }

    const savedModel = storage.get("model", "Qwen/Qwen-Image-Edit");
    if (savedModel) settingModel.value = savedModel;

    const savedSeed = storage.get("seed", "");
    if (savedSeed) settingSeed.value = savedSeed;

    const savedSteps = storage.get("steps", "");
    if (savedSteps) settingSteps.value = savedSteps;
  }

  function getActiveConfig() {
    let baseUrl = settingBaseUrlSelect.value;
    if (baseUrl === "custom") {
      baseUrl = settingCustomBaseUrl.value.trim();
    }
    return {
      apiKey: settingApiKey.value.trim(),
      baseUrl: baseUrl,
      model: settingModel.value,
      seed: settingSeed.value ? parseInt(settingSeed.value, 10) : null,
      steps: settingSteps.value ? parseInt(settingSteps.value, 10) : null,
    };
  }

  function isLocalBaseUrl(baseUrl) {
    try {
      const parsed = new URL(baseUrl);
      return ["localhost", "127.0.0.1", "0.0.0.0", "::1"].includes(parsed.hostname);
    } catch (_) {
      return false;
    }
  }

  function isDashScopeBaseUrl(baseUrl) {
    return /dashscope|aliyuncs\.com/i.test(baseUrl || "");
  }

  function isHuggingFaceBaseUrl(baseUrl) {
    return /huggingface|hf\.co/i.test(baseUrl || "");
  }

  function validateGenerationConfig(config) {
    if (isDashScopeBaseUrl(config.baseUrl) && config.apiKey.startsWith("nvapi-")) {
      return "QwenCloud / DashScope is selected, but the key is an NVIDIA nvapi-... key. Enter a DASHSCOPE_API_KEY / QwenCloud API key.";
    }
    if (isHuggingFaceBaseUrl(config.baseUrl) && config.apiKey && !config.apiKey.startsWith("hf_")) {
      return "Hugging Face is selected, but the key does not look like an hf_... token. Create a token with Inference Providers access.";
    }
    if (!isLocalBaseUrl(config.baseUrl) && !config.apiKey) {
      return "A cloud Base URL requires an API key. A local NIM does not require a key in the browser.";
    }
    return "";
  }

  async function checkApiHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const userKey = storage.get("api_key");

      const dot = apiStatusPill.querySelector(".status-dot");
      dot.className = "status-dot";

      const activeConfig = getActiveConfig();
      const browserCloudKeyReady = userKey && (isHuggingFaceBaseUrl(activeConfig.baseUrl) || isDashScopeBaseUrl(activeConfig.baseUrl));

      if (data.backend_connected || browserCloudKeyReady) {
        dot.classList.add("ready");
        apiStatusLabel.textContent = browserCloudKeyReady ? "API: cloud key saved" : (data.backend_message || "API: ready");
      } else if (userKey || data.has_env_key) {
        dot.classList.add("missing");
        apiStatusLabel.textContent = data.backend_message || "API: key found, backend not ready";
      } else {
        dot.classList.add("missing");
        apiStatusLabel.textContent = data.backend_message || "API: key required";
      }
    } catch (e) {
      console.warn("Health check error:", e);
    }
  }

  // ==================== ОБРАЗЦЫ СТАРЫХ ВЕЩЕЙ ====================
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      if (data.samples && data.samples.length > 0) {
        sampleChipsRow.innerHTML = "";
        data.samples.forEach((sample) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "sample-chip";
          const icon = sample.id === "chair" ? "🪑" : sample.id === "dresser" ? "🗄️" : sample.id === "jacket" ? "🧥" : "💡";
          btn.innerHTML = `${icon} ${sample.title}`;
          btn.title = `${sample.subtitle || ""} — Click to load`;
          btn.addEventListener("click", (e) => {
            e.stopPropagation();
            loadSampleItem(sample);
          });
          sampleChipsRow.appendChild(btn);
        });
      }
    } catch (e) {
      console.warn("Failed to load samples:", e);
    }
  }

  async function loadSampleItem(sample) {
    try {
      const resp = await fetch(sample.imageUrl);
      const blob = await resp.blob();
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageSource(e.target.result, `${sample.title} (${sample.subtitle || "Sample"})`);
        promptInput.value = sample.prompt;
        currentStyleHint = sample.style;
        updateStyleTagHighlight(sample.style);
        updateGenerateButtonState();
        showToast(`Sample loaded: ${sample.title}`, "info");
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error("Error loading sample image:", err);
      showToast("Could not load the sample", "error");
    }
  }

  // ==================== СТИЛИСТИЧЕСКИЕ ТЕГИ ====================
  function setupStyleTags() {
    if (!styleTagsGrid) return;
    styleTagsGrid.querySelectorAll(".style-tag-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const style = chip.dataset.style;
        const promptPart = chip.dataset.promptPart;

        if (chip.classList.contains("active")) {
          chip.classList.remove("active");
          currentStyleHint = null;
        } else {
          styleTagsGrid.querySelectorAll(".style-tag-chip").forEach((c) => c.classList.remove("active"));
          chip.classList.add("active");
          currentStyleHint = style;

          // Добавляем или заменяем в тексте пожелания
          const currentText = promptInput.value.trim();
          if (!currentText) {
            promptInput.value = `Transform this item ${promptPart}`;
          } else if (!currentText.toLowerCase().includes(style)) {
            promptInput.value = `${currentText}, ${promptPart}`;
          }
        }
        updateGenerateButtonState();
        promptInput.focus();
      });
    });
  }

  function updateStyleTagHighlight(styleKey) {
    if (!styleTagsGrid) return;
    styleTagsGrid.querySelectorAll(".style-tag-chip").forEach((c) => {
      if (c.dataset.style === styleKey) {
        c.classList.add("active");
      } else {
        c.classList.remove("active");
      }
    });
  }

  // ==================== ЗАГРУЗКА ИЗОБРАЖЕНИЙ ====================
  function setupDragAndDrop() {
    dropZone.addEventListener("click", (e) => {
      if (e.target.closest(".sample-chip") || e.target.closest("#clearImageBtn")) return;
      fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files[0]) {
        handleFileSelect(dt.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
      showToast("Please choose an image file (PNG, JPG, or WebP)", "error");
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      setImageSource(e.target.result, `${file.name} (${Math.round(file.size / 1024)} KB)`);
    };
    reader.readAsDataURL(file);
  }

  function setImageSource(dataUri, metaText = "") {
    currentImageDataUri = dataUri;
    sourcePreview.src = dataUri;
    beforeImage.src = dataUri;
    sourceMeta.textContent = metaText;

    dropPlaceholder.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    clearImageBtn.classList.remove("hidden");

    updateGenerateButtonState();
  }

  function clearImage() {
    currentFile = null;
    currentImageDataUri = null;
    sourcePreview.src = "";
    beforeImage.src = "";
    fileInput.value = "";

    dropPlaceholder.classList.remove("hidden");
    previewContainer.classList.add("hidden");
    clearImageBtn.classList.add("hidden");

    updateGenerateButtonState();
  }

  // ==================== ПРЕСЕТЫ И ПРОМПТ ====================
  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      renderPresets(data.presets || []);
    } catch (e) {
      console.warn("Failed to load presets:", e);
    }
  }

  function renderPresets(presets) {
    presetsGrid.innerHTML = "";
    presets.forEach((preset) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-card";
      btn.innerHTML = `
        <span class="preset-icon">${preset.icon}</span>
        <div class="preset-text">
          <span class="preset-title">${preset.title}</span>
          ${preset.category ? `<span class="preset-cat">${preset.category}</span>` : ""}
        </div>
      `;
      btn.addEventListener("click", () => {
        promptInput.value = preset.prompt;
        currentStyleHint = preset.style || null;
        updateStyleTagHighlight(preset.style);
        promptInput.focus();
        updateGenerateButtonState();
      });
      presetsGrid.appendChild(btn);
    });
  }

  function updateGenerateButtonState() {
    const hasImage = !!currentImageDataUri;
    const hasPrompt = !!promptInput.value.trim();
    generateBtn.disabled = !(hasImage && hasPrompt);
  }

  // ==================== BEFORE / AFTER СЛАЙДЕР ====================
  function setupSlider() {
    const updateSlider = (clientX) => {
      const rect = compareWrapper.getBoundingClientRect();
      let offsetX = clientX - rect.left;
      if (offsetX < 0) offsetX = 0;
      if (offsetX > rect.width) offsetX = rect.width;

      const percentage = (offsetX / rect.width) * 100;
      setSliderPosition(percentage);
    };

    const startDrag = (e) => {
      isDraggingSlider = true;
      updateSlider(e.touches ? e.touches[0].clientX : e.clientX);
    };

    const stopDrag = () => {
      isDraggingSlider = false;
    };

    const moveDrag = (e) => {
      if (!isDraggingSlider) return;
      updateSlider(e.touches ? e.touches[0].clientX : e.clientX);
    };

    sliderHandle.addEventListener("mousedown", startDrag);
    sliderHandle.addEventListener("touchstart", startDrag, { passive: true });

    compareWrapper.addEventListener("mousedown", startDrag);
    compareWrapper.addEventListener("touchstart", startDrag, { passive: true });

    window.addEventListener("mousemove", moveDrag);
    window.addEventListener("touchmove", moveDrag, { passive: true });

    window.addEventListener("mouseup", stopDrag);
    window.addEventListener("touchend", stopDrag);
  }

  function setSliderPosition(percentage) {
    compareOverlay.style.width = `${percentage}%`;
    sliderHandle.style.left = `${percentage}%`;

    const wrapperWidth = compareWrapper.offsetWidth;
    beforeImage.style.width = `${wrapperWidth}px`;
  }

  // ==================== ГЕНЕРАЦИЯ ====================
  generateBtn.addEventListener("click", async () => {
    if (!currentImageDataUri || !promptInput.value.trim()) return;

    const config = getActiveConfig();
    const configError = validateGenerationConfig(config);
    if (configError) {
      showToast(configError, "error");
      settingsModal.classList.remove("hidden");
      return;
    }
    startLoading(config.model);

    try {
      const formData = new FormData();
      if (currentFile) {
        formData.append("file", currentFile);
      } else {
        const blob = await (await fetch(currentImageDataUri)).blob();
        formData.append("file", blob, "input.png");
      }

      formData.append("prompt", promptInput.value.trim());
      if (currentStyleHint) formData.append("style_hint", currentStyleHint);
      formData.append("model", config.model);
      if (config.apiKey && !isLocalBaseUrl(config.baseUrl)) formData.append("api_key", config.apiKey);
      if (config.baseUrl) formData.append("base_url", config.baseUrl);
      if (config.seed !== null && !isNaN(config.seed)) formData.append("seed", config.seed);
      if (config.steps !== null && !isNaN(config.steps)) formData.append("steps", config.steps);

      const resp = await fetch("/api/edit", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok || !data.success) {
        handleApiError(data);
        return;
      }

      // Успешно получено
      displayResult(data);
      if (data.is_demo) {
        showToast(`Transformation completed in demo mode in ${data.latency} sec!`, "info");
      } else {
        showToast(`Transformation and DIY plan ready in ${data.latency} sec!`, "success");
      }

    } catch (err) {
      console.error(err);
      showToast(`Network error: ${err.message}`, "error");
    } finally {
      stopLoading();
    }
  });

  function startLoading(modelName) {
    generateBtn.disabled = true;
    generateBtnText.textContent = "AI agents are creating your project...";
    loadingModelTag.textContent = `${modelName} + Upcycle Agent`;
    loadingOverlay.classList.remove("hidden");
    emptyStage.classList.add("hidden");

    startTime = performance.now();
    loadingTimer.textContent = "0.0 sec";
    timerInterval = setInterval(() => {
      const sec = ((performance.now() - startTime) / 1000).toFixed(1);
      loadingTimer.textContent = `${sec} sec`;
    }, 100);
  }

  function stopLoading() {
    clearInterval(timerInterval);
    loadingOverlay.classList.add("hidden");
    generateBtn.disabled = false;
    generateBtnText.textContent = "Transform & Create a DIY Plan";
  }

  function displayResult(data) {
    lastResultDataUri = data.image;

    // Слайдер
    beforeImage.src = currentImageDataUri;
    afterImage.src = data.image;

    afterImage.onload = () => {
      compareContainer.classList.remove("hidden");
      setSliderPosition(50);
    };

    // Метрики
    metricModel.textContent = data.model;
    metricLatency.textContent = `${data.latency} sec`;
    metricSize.textContent = data.size_bytes ? `${Math.round(data.size_bytes / 1024)} KB` : "PNG";
    viewerFooter.classList.remove("hidden");

    // Отрисовка карточки DIY-Плана
    if (data.diy_plan) {
      renderDiyCard(data.diy_plan);
    }

    // Активируем кнопки скачивания и повторного шага
    downloadBtn.disabled = false;
    chainEditBtn.disabled = false;
  }

  function createStepVisual(step, idx, total) {
    const progress = total <= 1 ? 1 : idx / (total - 1);
    const progressPct = Math.round(progress * 100);
    const source = currentImageDataUri || lastResultDataUri;
    const result = lastResultDataUri || currentImageDataUri;
    const visualType = normalizeStepVisualType(step.visual_type);
    const shouldUseResult = visualType === "finish_line" && result;
    const imageForStage = shouldUseResult ? result : source;

    const visualCol = document.createElement("div");
    visualCol.className = "diy-step-visual-col";

    const visual = document.createElement("div");
    visual.className = `diy-step-visual ${visualType}`;
    visual.style.setProperty("--stage-progress", `${progressPct}%`);

    if (imageForStage) {
      const img = document.createElement("img");
      img.className = "diy-step-visual-img base";
      img.src = imageForStage;
      img.alt = shouldUseResult ? "Final result" : "Work stage shown on the original photo";
      visual.appendChild(img);
    }

    const overlay = document.createElement("div");
    overlay.className = "diy-step-overlay";
    visual.appendChild(overlay);

    const badge = document.createElement("span");
    badge.className = "diy-step-visual-badge";
    badge.textContent = step.visual_hint || (idx === 0 ? "Marking" : idx === total - 1 ? "Final" : `Stage ${idx + 1}`);
    visual.appendChild(badge);

    visualCol.appendChild(visual);

    const caption = document.createElement("div");
    caption.className = "diy-step-visual-caption";
    caption.textContent = getStepVisualCaption(step, visualType);
    visualCol.appendChild(caption);

    return visualCol;
  }

  function normalizeStepVisualType(value) {
    const allowed = new Set(["measure_line", "mark_line", "cut_line", "hem_line", "finish_line", "progress"]);
    return allowed.has(value) ? value : "progress";
  }

  function getStepVisualCaption(step, visualType) {
    if (step.visual_desc) return step.visual_desc;
    if (visualType === "measure_line") return "The image shows the intended length and the seam allowance below it.";
    if (visualType === "mark_line") return "The image shows matching cutting lines on both legs.";
    if (visualType === "cut_line") return "The shaded lower area is the fabric to remove.";
    if (visualType === "hem_line") return "The two green lines mark the lower hem area.";
    if (visualType === "finish_line") return "This is how the finished result should look after fitting and adjustment.";
    return step.markup || step.desc || "Check this stage before moving on.";
  }

  function appendStepMeta(container, label, text) {
    if (!text) return;
    const row = document.createElement("div");
    row.className = "diy-step-meta-line";

    const labelEl = document.createElement("span");
    labelEl.className = "diy-step-meta-label";
    labelEl.textContent = label;

    const textEl = document.createElement("span");
    textEl.className = "diy-step-meta-text";
    textEl.textContent = text;

    row.appendChild(labelEl);
    row.appendChild(textEl);
    container.appendChild(row);
  }

  function renderDiyCard(plan) {
    currentDiyPlan = plan;

    diyItemTitle.textContent = plan.item_name || "Upcycling Plan";
    diyItemSubtitle.textContent = `${plan.category || "Upcycling"} • Style: ${plan.style || "Modern"}`;

    diyDifficulty.textContent = plan.difficulty || "Intermediate";
    diyTime.textContent = plan.estimated_time || "2–4 hours";
    diyCost.textContent = plan.estimated_cost || "~800 ₴";
    diyEco.textContent = plan.eco_impact || "Extends the life of the item";

    // Материалы
    diyMaterialsList.innerHTML = "";
    (plan.materials || []).forEach((m) => {
      const li = document.createElement("li");
      li.textContent = m;
      diyMaterialsList.appendChild(li);
    });

    // Инструменты
    diyToolsList.innerHTML = "";
    (plan.tools || []).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      diyToolsList.appendChild(li);
    });

    // Шаги с визуальным прогрессом: исходное фото -> промежуточные состояния -> финал
    diyStepsContainer.innerHTML = "";
    const steps = plan.steps || [];
    steps.forEach((step, idx) => {
      const stepDiv = document.createElement("div");
      stepDiv.className = "diy-step-item";

      const num = document.createElement("div");
      num.className = "diy-step-num";
      num.textContent = step.number || idx + 1;

      const visual = createStepVisual(step, idx, steps.length);

      const content = document.createElement("div");
      content.className = "diy-step-content";

      const title = document.createElement("div");
      title.className = "diy-step-title";
      title.textContent = step.title || `Step ${idx + 1}`;

      const desc = document.createElement("div");
      desc.className = "diy-step-desc";
      desc.textContent = step.desc || "Complete this stage carefully and check the result before moving on.";

      content.appendChild(title);
      content.appendChild(desc);
      appendStepMeta(content, "Marking", step.markup);
      appendStepMeta(content, "Check", step.checkpoint);

      stepDiv.appendChild(num);
      stepDiv.appendChild(visual);
      stepDiv.appendChild(content);
      diyStepsContainer.appendChild(stepDiv);
    });

    // Секрет мастера
    diyProTipText.textContent = plan.pro_tip || "Work in thin, careful coats and wear respiratory protection while sanding.";

    // Показываем карточку
    diyCard.classList.remove("hidden");
    diyCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ==================== КОПИРОВАНИЕ ИНСТРУКЦИИ ====================
  if (copyDiyBtn) {
    copyDiyBtn.addEventListener("click", () => {
      if (!currentDiyPlan) return;
      const p = currentDiyPlan;
      let text = `🛠️ ${p.item_name}\n`;
      text += `Category: ${p.category} | Style: ${p.style}\n`;
      text += `Difficulty: ${p.difficulty} | Time: ${p.estimated_time} | Budget: ${p.estimated_cost}\n`;
      text += `Eco impact: ${p.eco_impact}\n\n`;

      text += `📦 MATERIALS:\n`;
      (p.materials || []).forEach((m) => { text += `- ${m}\n`; });

      text += `\n🔧 TOOLS:\n`;
      (p.tools || []).forEach((t) => { text += `- ${t}\n`; });

      text += `\n📋 STEP-BY-STEP PLAN:\n`;
      (p.steps || []).forEach((s, i) => {
        text += `${s.number || i + 1}. ${s.title}\n   ${s.desc}\n`;
      });

      if (p.pro_tip) {
        text += `\n💡 PRO TIP: ${p.pro_tip}\n`;
      }

      navigator.clipboard.writeText(text).then(() => {
        showToast("Instructions copied to the clipboard!", "success");
      }).catch(() => {
        showToast("Could not copy the text", "error");
      });
    });
  }

  function handleApiError(data) {
    let msg = data.message || "Unknown API error";
    if (data.error === "AUTH_ERROR") {
      msg = data.message || "Authentication error: check the key for the selected Base URL.";
      settingsModal.classList.remove("hidden");
    }
    showToast(msg, "error");
  }

  // ==================== ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ВЬЮВЕРА ====================
  document.querySelectorAll(".view-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const mode = btn.dataset.mode;
      if (mode === "split") {
        compareOverlay.style.display = "flex";
        sliderHandle.style.display = "flex";
        setSliderPosition(50);
      } else if (mode === "result") {
        compareOverlay.style.display = "none";
        sliderHandle.style.display = "none";
      } else if (mode === "source") {
        compareOverlay.style.display = "flex";
        sliderHandle.style.display = "none";
        setSliderPosition(100);
      }
    });
  });

  // ==================== СКАЧИВАНИЕ И ЦЕПОЧКА ШАГОВ ====================
  downloadBtn.addEventListener("click", () => {
    if (!lastResultDataUri) return;
    const a = document.createElement("a");
    a.href = lastResultDataUri;
    a.download = `upcycle_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast("Image saved!", "success");
  });

  chainEditBtn.addEventListener("click", () => {
    if (!lastResultDataUri) return;
    setImageSource(lastResultDataUri, "Result of the previous upcycling step");
    showToast("The result is now the source image for the next edit!", "success");
  });

  // ==================== ОБРАБОТЧИКИ СОБЫТИЙ ====================
  function setupEventListeners() {
    promptInput.addEventListener("input", updateGenerateButtonState);

    clearPromptBtn.addEventListener("click", () => {
      promptInput.value = "";
      currentStyleHint = null;
      if (styleTagsGrid) {
        styleTagsGrid.querySelectorAll(".style-tag-chip").forEach((c) => c.classList.remove("active"));
      }
      updateGenerateButtonState();
    });

    clearImageBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearImage();
    });

    // Настройки
    settingsBtn.addEventListener("click", () => {
      settingsModal.classList.remove("hidden");
    });

    closeSettingsBtn.addEventListener("click", () => {
      settingsModal.classList.add("hidden");
    });

    if (cancelSettingsBtn) {
      cancelSettingsBtn.addEventListener("click", () => {
        settingsModal.classList.add("hidden");
      });
    }

    settingsModal.addEventListener("click", (e) => {
      if (e.target === settingsModal) {
        settingsModal.classList.add("hidden");
      }
    });

    toggleKeyVisibility.addEventListener("click", () => {
      if (settingApiKey.type === "password") {
        settingApiKey.type = "text";
        toggleKeyVisibility.textContent = "Hide";
      } else {
        settingApiKey.type = "password";
        toggleKeyVisibility.textContent = "Show";
      }
    });

    settingBaseUrlSelect.addEventListener("change", () => {
      if (settingBaseUrlSelect.value === "custom") {
        settingCustomBaseUrl.classList.remove("hidden");
      } else {
        settingCustomBaseUrl.classList.add("hidden");
      }
      if (settingBaseUrlSelect.value === "http://localhost:8000/v1") {
        settingApiKey.placeholder = "Not required for local NIM inference";
      } else if (settingBaseUrlSelect.value === "https://dashscope-intl.aliyuncs.com/api/v1") {
        settingApiKey.placeholder = "DashScope / QwenCloud API key";
      } else if (settingBaseUrlSelect.value === "https://api-inference.huggingface.co") {
        settingApiKey.placeholder = "HF token: hf_...";
        if (settingModel.value === "qwen/qwen-image-edit") settingModel.value = "Qwen/Qwen-Image-Edit";
      }
    });

    saveSettingsBtn.addEventListener("click", () => {
      storage.set("api_key", settingApiKey.value.trim());
      let baseUrl = settingBaseUrlSelect.value;
      if (baseUrl === "custom") baseUrl = settingCustomBaseUrl.value.trim();
      storage.set("base_url", baseUrl);
      storage.set("model", settingModel.value);
      storage.set("seed", settingSeed.value);
      storage.set("steps", settingSteps.value);

      settingsModal.classList.add("hidden");
      checkApiHealth();
      showToast("Settings saved!", "success");
    });

    // Изменение размера окна - пересчет ширины картинки в слайдере
    window.addEventListener("resize", () => {
      if (compareWrapper && beforeImage) {
        beforeImage.style.width = `${compareWrapper.offsetWidth}px`;
      }
    });
  }

  // ==================== TOAST УВЕДОМЛЕНИЯ ====================
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    const icon = type === "success" ? "✅" : type === "error" ? "❌" : "ℹ️";
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
});
