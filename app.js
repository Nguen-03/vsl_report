"use strict";

const byId = (id) => document.getElementById(id);
const player = byId("preview");
const run = byId("run");
const status = byId("status");
const historyKey = "dau-tay-history-v1";
const limit = 100 * 1024 * 1024;
let source = "video";
let file = null;
let videoUrl = null;
let stream = null;
let openingCamera = false;
let cameraRecorder = null;
let job = null;
let current = null;
let entries = [];

try {
    const saved = JSON.parse(localStorage.getItem(historyKey) || "[]");
    if (Array.isArray(saved))
        entries = saved
            .filter((item) => item && typeof item.label === "string")
            .slice(0, 50);
} catch {
    /* Keep the interface usable when storage is unavailable. */
}

function say(message, error = false) {
    status.textContent = message;
    status.classList.toggle("error", error);
}

function showPreview(active) {
    player.hidden = !active;
    byId("placeholder").hidden = active;
}

function isCameraSource(value) {
    return value === "camera" || value === "add-sign";
}

function refreshButtons() {
    const liveJob = job && isCameraSource(job.source);
    run.disabled = job
        ? !liveJob || !job.recording
        : source === "video"
          ? !file
          : !stream ||
            (source === "add-sign" && !byId("add-sign-name").value.trim());
    run.querySelector("span").textContent = liveJob
        ? job.recording
            ? "Kết thúc"
            : "Đang xử lý…"
        : job
          ? "Đang nhận diện…"
          : isCameraSource(source)
            ? "Bắt đầu"
            : "Bắt đầu nhận diện";
    byId("camera-toggle").disabled = openingCamera;
    byId("add-sign-toggle").disabled = openingCamera;
    byId("camera-label").textContent = stream ? "Tắt camera" : "Bật camera";
    byId("add-sign-label").textContent = stream ? "Tắt camera" : "Bật camera";
    byId("add-sign-name").disabled = Boolean(job);
}

function renderResult(item) {
    current = item;
    byId("prediction").textContent = item?.label || "—";
    const top5 = byId("top5");
    top5.replaceChildren();
    top5.hidden = !item || !Array.isArray(item.top5) || item.top5.length === 0;
    byId("top5-heading").hidden = top5.hidden;
    if (item && Array.isArray(item.top5)) {
        top5.replaceChildren(
            ...item.top5.slice(0, 5).map((result, index) => {
                const row = document.createElement("li");
                const label = document.createElement("span");
                const score = document.createElement("strong");
                label.textContent = `${index + 1}. ${result.label}`;
                score.textContent = Number(result.cosine_distance).toFixed(4);
                row.append(label, score);
                return row;
            }),
        );
    }
    byId("result-state").textContent = item
        ? item.demo
            ? "KẾT QUẢ MINH HỌA"
            : "ĐÃ NHẬN DIỆN"
        : "CHƯA CÓ KẾT QUẢ";
    byId("score").textContent = !item
        ? "Kết quả sẽ xuất hiện sau khi nhận diện."
        : item.demo
          ? "Dữ liệu mẫu để kiểm tra giao diện."
          : item.cosine_distance == null
            ? "Khoảng cách cosine thấp hơn nghĩa là kết quả giống hơn."
            : "Khoảng cách cosine: " +
              Number(item.cosine_distance).toFixed(4) +
              " (càng thấp càng giống).";
    byId("copy").disabled = !item;
}

function renderHistory() {
    byId("count").textContent = entries.length;
    byId("empty").hidden = entries.length > 0;
    byId("clear").disabled = entries.length === 0;
    byId("entries").replaceChildren(
        ...entries.map((item) => {
            const row = document.createElement("li");
            const content = document.createElement("div");
            const label = document.createElement("strong");
            const detail = document.createElement("small");
            const time = document.createElement("time");
            label.textContent = item.label;
            detail.textContent = [
                item.source === "camera" ? "Camera" : item.name || "Video",
                item.demo ? "Minh họa" : "Model",
            ].join(" · ");
            const date = new Date(item.at);
            if (!Number.isNaN(date.getTime())) {
                time.dateTime = date.toISOString();
                time.textContent = new Intl.DateTimeFormat("vi-VN", {
                    dateStyle: "short",
                    timeStyle: "short",
                }).format(date);
            }
            content.append(label, detail);
            row.append(content, time);
            return row;
        }),
    );
}

function saveHistory() {
    renderHistory();
    try {
        localStorage.setItem(historyKey, JSON.stringify(entries));
        return true;
    } catch {
        return false;
    }
}

function detachPlayer() {
    player.pause();
    player.srcObject = null;
    player.removeAttribute("src");
    player.load();
    player.classList.remove("mirror");
    showPreview(false);
}

function closeCamera() {
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
}

function stopJob() {
    job?.controller.abort();
    job = null;
    refreshButtons();
}

function showFile() {
    if (!file) return;
    player.controls = true;
    player.muted = false;
    player.src = videoUrl;
    showPreview(true);
}

function chooseSource(next) {
    if (source === next) return;
    stopJob();
    closeCamera();
    detachPlayer();
    renderResult(null);
    source = next;
    byId("video-tab").setAttribute("aria-pressed", next === "video");
    byId("camera-tab").setAttribute("aria-pressed", next === "camera");
    byId("add-sign-tab").setAttribute("aria-pressed", next === "add-sign");
    byId("video-controls").hidden = next !== "video";
    byId("camera-controls").hidden = next !== "camera";
    byId("add-sign-controls").hidden = next !== "add-sign";

    if (next === "video") showFile();
    byId("placeholder").querySelector("strong").textContent =
        next === "video"
            ? "Chưa có video"
            : next === "add-sign"
              ? "Camera thêm ký hiệu đang tắt"
              : "Camera đang tắt";
    say(
        next === "camera"
            ? "Bật camera để xem trước."
            : next === "add-sign"
              ? "Nhập tên ký hiệu và bật camera để bắt đầu."
              : file
                ? "Video đã sẵn sàng."
                : "Đang chờ video.",
    );
    refreshButtons();
}

function loadFile(nextFile) {
    if (!nextFile) return;
    if (
        !nextFile.type.startsWith("video/") &&
        !/\.(mp4|webm|mov)$/i.test(nextFile.name)
    ) {
        say("Hãy chọn một tệp video.", true);
        return;
    }
    if (!nextFile.size || nextFile.size > limit) {
        say("Hãy chọn video có dung lượng từ 1 byte đến 100 MB.", true);
        return;
    }
    stopJob();
    detachPlayer();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    file = nextFile;
    videoUrl = URL.createObjectURL(file);
    byId("file-name").textContent = file.name;
    renderResult(null);
    showFile();
    say("Video đã sẵn sàng.");
    refreshButtons();
}

async function toggleCamera() {
    if (stream) {
        stopJob();
        closeCamera();
        detachPlayer();
        say("Camera đã tắt.");
        refreshButtons();
        return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        say(
            "Camera cần Chrome hoặc Edge với địa chỉ localhost hay HTTPS.",
            true,
        );
        return;
    }
    openingCamera = true;
    say("Đang chờ quyền truy cập camera…");
    refreshButtons();
    try {
        const camera = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 },
            },
            audio: false,
        });
        if (!isCameraSource(source)) {
            camera.getTracks().forEach((track) => track.stop());
            return;
        }
        stream = camera;
        camera.getVideoTracks()[0].addEventListener("ended", () => {
            if (stream !== camera) return;
            stopJob();
            closeCamera();
            detachPlayer();
            say("Camera đã ngắt kết nối.", true);
            refreshButtons();
        });
        player.controls = false;
        player.muted = true;
        player.srcObject = stream;
        player.classList.add("mirror");
        showPreview(true);
        await player.play();
        if (stream === camera) say("Camera đã sẵn sàng.");
    } catch (error) {
        if (isCameraSource(source)) {
            closeCamera();
            detachPlayer();
            say(
                error.name === "NotAllowedError"
                    ? "Hãy cấp quyền truy cập camera trong trình duyệt."
                    : "Không thể mở camera. Hãy kiểm tra thiết bị.",
                true,
            );
        }
    } finally {
        openingCamera = false;
        refreshButtons();
    }
}

// async function (params) {

// }

function recordClip(camera, signal) {
    return new Promise((resolve, reject) => {
        if (signal.aborted) {
            reject(signal.reason);
            return;
        }
        const mimeType = [
            "video/webm;codecs=vp8",
            "video/webm",
            "video/mp4",
        ].find((type) => MediaRecorder.isTypeSupported(type));
        const recorder = new MediaRecorder(
            camera,
            mimeType ? { mimeType } : {},
        );
        const chunks = [];
        let totalBytes = 0;
        let tooLarge = false;
        const stop = () => {
            if (recorder.state !== "inactive") recorder.stop();
        };
        const cleanup = () => {
            signal.removeEventListener("abort", stop);
            if (cameraRecorder === recorder) cameraRecorder = null;
        };
        recorder.ondataavailable = (event) => {
            if (!event.data.size) return;
            totalBytes += event.data.size;
            if (totalBytes > limit) {
                tooLarge = true;
                stop();
                return;
            }
            chunks.push(event.data);
        };
        recorder.onerror = () => {
            cleanup();
            reject(new Error("Không thể ghi đoạn video từ camera."));
        };
        recorder.onstop = () => {
            cleanup();
            if (signal.aborted) {
                reject(signal.reason);
                return;
            }
            if (tooLarge) {
                reject(new Error("Video quay vượt quá giới hạn 100 MB."));
                return;
            }
            const clip = new Blob(chunks, {
                type: recorder.mimeType || "video/webm",
            });
            if (clip.size) resolve(clip);
            else
                reject(
                    new Error(
                        "Camera không tạo được video. Hãy bật lại camera.",
                    ),
                );
        };
        recorder.start(1000);
        cameraRecorder = recorder;
        signal.addEventListener("abort", stop, { once: true });
    });
}

async function requestPrediction(blob, task, name) {
    const query = new URLSearchParams({ source: task.source, name });
    const response = await fetch("/api/predict?" + query, {
        method: "POST",
        body: blob,
        headers: { "Content-Type": blob.type || "application/octet-stream" },
        signal: task.controller.signal,
    });
    const data = await response.json();
    if (!response.ok)
        throw new Error(data.error || "Không thể nhận diện video.");
    if (typeof data.label !== "string" || !data.label.trim())
        throw new Error("Backend không trả nhãn hợp lệ.");
    const cosineDistance = (result) => {
        if (result.cosine_distance != null)
            return Number(result.cosine_distance);
        if (result.confidence != null)
            return 1 - Number(result.confidence);
        return NaN;
    };
    return {
        label: data.label,
        cosine_distance:
            data.cosine_distance != null
                ? Number(data.cosine_distance)
                : data.confidence != null
                  ? 1 - Number(data.confidence)
                  : null,
        top5: Array.isArray(data.top5)
            ? data.top5
                  .filter(
                      (result) =>
                          result &&
                          typeof result.label === "string" &&
                          result.label.trim(),
                  )
                  .map((result) => ({
                      label: result.label,
                      cosine_distance: cosineDistance(result),
                  }))
                  .filter((result) =>
                      Number.isFinite(result.cosine_distance),
                  )
            : [],
        demo: data.demo === true,
        source: task.source,
        name,
        at: new Date().toISOString(),
    };
}

async function requestAddSign(blob, task) {
    const query = new URLSearchParams({ label: task.label });
    const response = await fetch("/api/add-sign?" + query, {
        method: "POST",
        body: blob,
        headers: { "Content-Type": blob.type || "application/octet-stream" },
        signal: task.controller.signal,
    });
    const data = await response.json();
    if (!response.ok)
        throw new Error(data.error || "Không thể thêm ký hiệu.");
    if (data.label !== task.label)
        throw new Error("Máy chủ không xác nhận đúng tên ký hiệu.");
    return data;
}

async function analyze() {
    if (job) {
        if (isCameraSource(job.source) && job.recording) {
            job.recording = false;
            if (cameraRecorder?.state !== "inactive") cameraRecorder?.stop();
            refreshButtons();
            say("Đang xử lý video…");
        }
        return;
    }
    if (source === "video" ? !file : !stream) return;
    const label = byId("add-sign-name").value.trim().normalize("NFC");
    if (source === "add-sign" && !label) {
        say("Hãy nhập tên ký hiệu trước khi ghi.", true);
        return;
    }
    const task = {
        controller: new AbortController(),
        source,
        recording: isCameraSource(source),
        label,
    };
    job = task;
    refreshButtons();
    try {
        const cameraMode = isCameraSource(task.source);
        if (cameraMode) say("Đang ghi video từ camera…");
        const blob = cameraMode
            ? await recordClip(stream, task.controller.signal)
            : file;
        if (job !== task) return;
        task.recording = false;
        refreshButtons();
        if (task.source === "add-sign") {
            const addedSign = await requestAddSign(blob, task);
            if (job !== task) return;
            current = { label: addedSign.label, demo: false };
            byId("prediction").textContent = addedSign.label;
            byId("result-state").textContent = "ĐÃ THÊM KÝ HIỆU";
            byId("score").textContent = "Mẫu đã được lưu vào gallery.";
            byId("top5").replaceChildren();
            byId("top5").hidden = true;
            byId("top5-heading").hidden = true;
            byId("copy").disabled = false;
            say('Đã thêm ký hiệu "' + addedSign.label + '" vào gallery.');
            return;
        }
        say("Đang xử lý video…");
        const name = cameraMode
            ? blob.type.includes("mp4")
                ? "camera.mp4"
                : "camera.webm"
            : blob.name;
        const result = await requestPrediction(blob, task, name);
        if (job !== task) return;
        renderResult(result);
        entries = [result, ...entries].slice(0, 50);
        if (!saveHistory())
            say("Đã nhận diện. Trình duyệt không cho lưu lịch sử.", true);
        else say("Đã lưu kết quả vào lịch sử.");
    } catch (error) {
        if (job === task)
            say(error.message || "Không kết nối được tới máy chủ.", true);
    } finally {
        if (job === task) {
            job = null;
            refreshButtons();
        }
    }
}

byId("video-tab").onclick = () => chooseSource("video");
byId("camera-tab").onclick = () => chooseSource("camera");
byId("add-sign-tab").onclick = () => chooseSource("add-sign");
byId("add-sign-name").addEventListener("input", refreshButtons);

byId("drop").onclick = () => byId("file").click();
byId("file").onchange = (event) => {
    loadFile(event.target.files[0]);
    event.target.value = "";
};
const drop = byId("drop");
drop.ondragover = (event) => {
    event.preventDefault();
    drop.classList.add("drag");
};
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = (event) => {
    event.preventDefault();
    drop.classList.remove("drag");
    loadFile(event.dataTransfer.files[0]);
};
player.onerror = () => {
    if (source === "video" && file)
        say(
            "Trình duyệt không phát được video này. Bạn vẫn có thể gửi tới backend.",
            true,
        );
};
byId("camera-toggle").onclick = toggleCamera;
byId("add-sign-toggle").onclick = toggleCamera;
run.onclick = analyze;
byId("copy").onclick = async () => {
    if (!current) return;
    try {
        await navigator.clipboard.writeText(
            current.demo ? current.label + " (minh họa)" : current.label,
        );
        say("Đã sao chép kết quả.");
    } catch {
        say(
            "Không thể sao chép tự động. Hãy chọn văn bản kết quả để sao chép.",
            true,
        );
    }
};
byId("clear").onclick = () => {
    if (!confirm("Xóa tất cả kết quả trong lịch sử?")) return;
    entries = [];
    say(saveHistory() ? "Đã xóa lịch sử." : "Đã xóa lịch sử của phiên này.");
};
window.addEventListener("pagehide", () => {
    stopJob();
    closeCamera();
});

renderHistory();
refreshButtons();
fetch("/api/health")
    .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json();
    })
    .then((data) => {
        byId("mode").classList.toggle("live", data.mode === "model");
        byId("mode").lastElementChild.textContent =
            data.mode === "model"
                ? "Đã kết nối model"
                : data.mode === "error"
                  ? "Lỗi kết nối model"
                  : "Chế độ minh họa";
    })
    .catch(() => {
        byId("mode").lastElementChild.textContent = "Chưa kết nối máy chủ";
        say("Hãy mở giao diện qua máy chủ Python trong gói ZIP.", true);
    });
