(() => {
  "use strict";

  const ALLOWED_EXTS = [".flac", ".m4a", ".wav", ".mp3", ".aac", ".ogg"];
  const MAX_BYTES = 50 * 1024 * 1024;
  const DB_NAME = "audio-detect";
  const DB_VERSION = 1;
  const STORE = "history";

  // ---------- tiny utilities ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const uid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
  const extOf = (name) => {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i).toLowerCase() : "";
  };
  const fmtBytes = (n) => {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(2) + " MB";
  };
  const fmtTime = (ts) => {
    const d = new Date(ts);
    return d.toLocaleString();
  };
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------- toast ----------
  const toastEl = $("#toast");
  let toastTimer = null;
  function toast(msg, kind = "") {
    toastEl.textContent = msg;
    toastEl.classList.remove("show", "err");
    if (kind === "err") toastEl.classList.add("err");
    // force reflow so transition fires reliably when called rapidly
    void toastEl.offsetWidth;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), 3200);
  }

  // ---------- IndexedDB for history + blobs ----------
  function openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("ts", "ts");
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  async function tx(mode, fn) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const store = t.objectStore(STORE);
      let result;
      Promise.resolve(fn(store)).then((r) => { result = r; });
      t.oncomplete = () => resolve(result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  }
  const historyDB = {
    async add(rec) { return tx("readwrite", (s) => s.put(rec)); },
    async delete(id) { return tx("readwrite", (s) => s.delete(id)); },
    async clear() { return tx("readwrite", (s) => s.clear()); },
    async get(id) {
      return tx("readonly", (s) => new Promise((res) => { s.get(id).onsuccess = (e) => res(e.target.result); }));
    },
    async all() {
      return tx("readonly", (s) => new Promise((res) => {
        const items = [];
        s.index("ts").openCursor(null, "prev").onsuccess = (e) => {
          const c = e.target.result;
          if (!c) return res(items);
          items.push(c.value);
          c.continue();
        };
      }));
    },
  };

  // ---------- queue state ----------
  /** @type {{id:string,name:string,size:number,blob:Blob,status:string,progress:number,xhr?:XMLHttpRequest,error?:string,result?:any}[]} */
  const queue = [];
  let activeXhr = null;

  const elQueue = $("#queue");
  const elQueueEmpty = $("#queue-empty");
  const elQueueSummary = $("#queue-summary");
  const tplQueue = $("#tpl-queue-item");

  function renderQueue() {
    elQueue.innerHTML = "";
    queue.forEach((it) => {
      const node = tplQueue.content.firstElementChild.cloneNode(true);
      node.dataset.id = it.id;
      node.classList.toggle("done", it.status === "done");
      node.classList.toggle("err", it.status === "error");
      $(".qi-title", node).textContent = it.name;
      $(".qi-meta", node).textContent = `${fmtBytes(it.size)} · ${it.source || "upload"}`;
      const fill = $(".bar > span", node);
      fill.style.width = (it.progress || 0) + "%";
      $(".qi-status", node).textContent = statusText(it);
      const cancel = $(".qi-cancel", node);
      if (it.status === "uploading" || it.status === "queued") {
        cancel.textContent = it.status === "queued" ? "Remove" : "Cancel";
        cancel.onclick = () => cancelItem(it.id);
      } else {
        cancel.textContent = "Remove";
        cancel.onclick = () => removeItem(it.id);
      }
      elQueue.appendChild(node);
    });
    elQueueEmpty.style.display = queue.length ? "none" : "";
    const pending = queue.filter((i) => i.status === "queued" || i.status === "uploading" || i.status === "analyzing").length;
    const done = queue.filter((i) => i.status === "done").length;
    const err = queue.filter((i) => i.status === "error").length;
    elQueueSummary.textContent = pending
      ? `${pending} pending · ${done} done${err ? ` · ${err} failed` : ""}`
      : (queue.length ? `${done} done${err ? ` · ${err} failed` : ""}` : "idle");
  }

  function statusText(it) {
    switch (it.status) {
      case "queued": return "Queued";
      case "uploading": return `Uploading… ${Math.floor(it.progress || 0)}%`;
      case "analyzing": return "Analyzing…";
      case "done": return it.result ? `${it.result.verdict} · AI ${it.result.ai_percent}% · Human ${it.result.human_percent}%` : "Done";
      case "error": return `Error: ${it.error || "unknown"}`;
      default: return it.status;
    }
  }

  function enqueue(blob, name, source = "upload") {
    if (blob.size > MAX_BYTES) {
      toast(`${name}: exceeds 50 MB limit`, "err");
      return;
    }
    const ext = extOf(name);
    if (!ALLOWED_EXTS.includes(ext)) {
      toast(`${name}: unsupported format ${ext || "(none)"}`, "err");
      return;
    }
    queue.push({
      id: uid(),
      name,
      size: blob.size,
      blob,
      status: "queued",
      progress: 0,
      source,
    });
    renderQueue();
    pump();
  }

  function removeItem(id) {
    const i = queue.findIndex((x) => x.id === id);
    if (i >= 0) queue.splice(i, 1);
    renderQueue();
  }
  function cancelItem(id) {
    const it = queue.find((x) => x.id === id);
    if (!it) return;
    if (it.status === "uploading" && it.xhr) {
      it.xhr.abort();
    }
    removeItem(id);
  }
  function clearDone() {
    for (let i = queue.length - 1; i >= 0; i--) {
      if (queue[i].status === "done" || queue[i].status === "error") queue.splice(i, 1);
    }
    renderQueue();
  }

  async function pump() {
    if (activeXhr) return;
    const next = queue.find((i) => i.status === "queued");
    if (!next) return;

    next.status = "uploading";
    next.progress = 0;
    renderQueue();

    const fd = new FormData();
    fd.append("file", next.blob, next.name);

    const xhr = new XMLHttpRequest();
    next.xhr = xhr;
    activeXhr = xhr;
    xhr.open("POST", "/predict");
    xhr.responseType = "json";

    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      next.progress = Math.min(99, (e.loaded / e.total) * 100);
      // only repaint the affected row's progress for perf
      const row = elQueue.querySelector(`[data-id="${next.id}"]`);
      if (row) {
        row.querySelector(".bar > span").style.width = next.progress + "%";
        row.querySelector(".qi-status").textContent = statusText(next);
      }
    };
    xhr.upload.onload = () => {
      next.status = "analyzing";
      next.progress = 100;
      renderQueue();
    };
    xhr.onabort = () => {
      activeXhr = null;
      next.xhr = undefined;
      // item already removed by cancelItem; just continue
      pump();
    };
    xhr.onerror = () => {
      activeXhr = null;
      next.xhr = undefined;
      next.status = "error";
      next.error = "Network error";
      renderQueue();
      pump();
    };
    xhr.onload = async () => {
      activeXhr = null;
      next.xhr = undefined;
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && body && !body.error) {
        next.status = "done";
        next.result = body;
        try {
          const rec = {
            id: next.id,
            ts: Date.now(),
            name: next.name,
            size: next.size,
            blob: next.blob,
            source: next.source,
            result: body,
          };
          await historyDB.add(rec);
          await refreshHistory();
        } catch (e) {
          console.error(e);
          toast("Saved in memory but failed to persist: " + e.message, "err");
        }
      } else {
        next.status = "error";
        next.error = (body && body.error) || `HTTP ${xhr.status}`;
      }
      renderQueue();
      pump();
    };

    xhr.send(fd);
  }

  // ---------- history rendering ----------
  const elHistory = $("#history");
  const elHistoryEmpty = $("#history-empty");
  const elHistorySummary = $("#history-summary");
  const tplHistory = $("#tpl-history-item");
  const audioUrls = new Map(); // id -> object URL (so we revoke on delete)

  async function refreshHistory() {
    const items = await historyDB.all();
    elHistory.innerHTML = "";
    for (const url of audioUrls.values()) URL.revokeObjectURL(url);
    audioUrls.clear();

    items.forEach((rec) => {
      const node = tplHistory.content.firstElementChild.cloneNode(true);
      node.dataset.id = rec.id;
      const verdict = rec.result.verdict;
      const vp = $(".verdict-pill", node);
      vp.textContent = verdict;
      vp.classList.add(verdict === "AI" ? "ai" : "human");
      $(".hi-name", node).textContent = rec.name;
      $(".hi-time", node).textContent = fmtTime(rec.ts);
      const ai = rec.result.ai_percent, hu = rec.result.human_percent;
      $(".bar.ai > span", node).style.width = ai + "%";
      $(".bar.human > span", node).style.width = hu + "%";
      $(".ai-pct", node).textContent = ai.toFixed(2) + "%";
      $(".human-pct", node).textContent = hu.toFixed(2) + "%";
      $(".hi-meta", node).textContent = `${fmtBytes(rec.size)} · ${rec.result.duration_seconds}s · ${rec.result.sample_rate} Hz · ${rec.source || "upload"}`;

      const audio = $(".hi-audio", node);
      const url = URL.createObjectURL(rec.blob);
      audioUrls.set(rec.id, url);
      audio.src = url;

      $(".hi-delete", node).onclick = async () => {
        await historyDB.delete(rec.id);
        await refreshHistory();
      };
      $(".hi-rerun", node).onclick = () => {
        enqueue(rec.blob, rec.name, rec.source || "re-run");
      };
      elHistory.appendChild(node);
    });

    elHistoryEmpty.style.display = items.length ? "none" : "";
    elHistorySummary.textContent = items.length + (items.length === 1 ? " result" : " results");
  }

  // ---------- samples ----------
  const elSamples = $("#samples");
  async function loadSamples() {
    try {
      const res = await fetch("/samples");
      const list = await res.json();
      elSamples.innerHTML = "";
      list.forEach((s) => {
        const li = document.createElement("li");
        li.innerHTML = `
          <span class="name" title="${esc(s.name)}">${esc(s.name)}</span>
          <span class="size">${fmtBytes(s.size)}</span>
          <span class="actions">
            <button type="button" class="btn small">Analyze</button>
          </span>
        `;
        li.querySelector("button").onclick = async () => {
          try {
            const r = await fetch("/sample/" + encodeURIComponent(s.name));
            if (!r.ok) throw new Error("HTTP " + r.status);
            const blob = await r.blob();
            enqueue(blob, s.name, "bundled");
          } catch (e) {
            toast("Failed to load sample: " + e.message, "err");
          }
        };
        elSamples.appendChild(li);
      });
    } catch (e) {
      elSamples.innerHTML = `<li class="muted">Failed to load samples: ${esc(e.message)}</li>`;
    }
  }

  // ---------- health ----------
  async function checkHealth() {
    const pill = $("#health-pill");
    try {
      const res = await fetch("/health");
      const j = await res.json();
      if (res.ok && j.status === "ok") {
        pill.textContent = "model: " + j.model;
        pill.classList.add("ok");
      } else {
        pill.textContent = "server error";
        pill.classList.add("bad");
      }
    } catch {
      pill.textContent = "offline";
      pill.classList.add("bad");
    }
  }

  // ---------- drop zone + file picker ----------
  const drop = $("#drop");
  const fileInput = $("#file-input");
  const pickBtn = $("#pick");

  function acceptFiles(files) {
    for (const f of files) enqueue(f, f.name, "upload");
  }

  drop.addEventListener("click", (e) => {
    if (e.target.closest("button.link")) return;
    fileInput.click();
  });
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  pickBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  fileInput.addEventListener("change", () => {
    acceptFiles(fileInput.files);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); if (ev === "drop" || !drop.contains(e.relatedTarget)) drop.classList.remove("drag"); })
  );
  drop.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) acceptFiles(files);
  });
  // also accept files dropped anywhere on the window
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    if (!drop.contains(e.target)) { // not already handled
      e.preventDefault();
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) acceptFiles(files);
    }
  });

  // ---------- toolbar wiring ----------
  $("#clear-queue").onclick = clearDone;
  $("#clear-history").onclick = async () => {
    if (!confirm("Delete all stored results? This cannot be undone.")) return;
    await historyDB.clear();
    await refreshHistory();
    toast("History cleared");
  };
  $("#export").onclick = async () => {
    const items = await historyDB.all();
    const slim = items.map((r) => ({ id: r.id, ts: r.ts, name: r.name, size: r.size, source: r.source, result: r.result }));
    const blob = new Blob([JSON.stringify(slim, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `audio-detect-history-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // ---------- boot ----------
  (async () => {
    checkHealth();
    loadSamples();
    try { await refreshHistory(); } catch (e) {
      console.error(e);
      toast("Failed to load history: " + e.message, "err");
    }
    renderQueue();
  })();
})();
