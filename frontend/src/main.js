import "./style.css";

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

const app = document.getElementById("app");
app.innerHTML = `
  <header class="shell-header">AI Ops Chat — RCA</header>
  <div class="shell">
    <section class="panel" aria-label="Query">
      <div class="panel-header">Question</div>
      <div class="panel-body">
        <div class="field">
          <label for="user-id">User ID</label>
          <input id="user-id" type="text" autocomplete="username" value="local-user" />
        </div>
        <div class="field" style="flex:1;display:flex;flex-direction:column;min-height:0">
          <label for="query">Query</label>
          <textarea id="query" placeholder="Describe the incident or ask about your logs…"></textarea>
        </div>
        <div class="actions">
          <button type="button" id="submit">Run analysis</button>
          <span class="status" id="status"></span>
        </div>
      </div>
    </section>
    <section class="panel" aria-label="Report">
      <div class="panel-header">Report (HTML)</div>
      <div class="panel-body" style="flex:1">
        <div id="placeholder" class="placeholder">Submit a query to render the model report here.</div>
        <div id="frame-wrap" class="output-frame-wrap hidden">
          <iframe id="report" title="RCA report" sandbox="allow-same-origin"></iframe>
        </div>
      </div>
    </section>
  </div>
`;

const $ = (id) => document.getElementById(id);

const userIdEl = $("user-id");
const queryEl = $("query");
const submitEl = $("submit");
const statusEl = $("status");
const placeholderEl = $("placeholder");
const frameWrapEl = $("frame-wrap");
const reportFrame = $("report");

function setLoading(loading) {
  submitEl.disabled = loading;
  statusEl.textContent = loading ? "Running…" : "";
  statusEl.classList.remove("error");
}

function showError(message) {
  statusEl.textContent = message;
  statusEl.classList.add("error");
}

function showReport(html) {
  placeholderEl.classList.add("hidden");
  frameWrapEl.classList.remove("hidden");
  reportFrame.srcdoc = html;
}

async function runChat() {
  const user_id = userIdEl.value.trim();
  const query = queryEl.value.trim();
  if (!user_id || !query) {
    showError("User ID and query are required.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${apiBase}/ai/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ user_id, query }),
    });

    const text = await res.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      throw new Error(text.slice(0, 200) || `Invalid JSON (HTTP ${res.status})`);
    }

    if (!res.ok) {
      const detail = data.detail ?? data;
      const msg =
        typeof detail === "string"
          ? detail
          : detail?.error ?? JSON.stringify(detail).slice(0, 300);
      throw new Error(msg || `Request failed (${res.status})`);
    }

    const html = data.analysis_html ?? "";
    showReport(html || "<p><em>Empty response.</em></p>");
    statusEl.textContent = "";
  } catch (e) {
    showError(e instanceof Error ? e.message : String(e));
    placeholderEl.classList.remove("hidden");
    frameWrapEl.classList.add("hidden");
    reportFrame.srcdoc = "";
  } finally {
    submitEl.disabled = false;
  }
}

submitEl.addEventListener("click", runChat);
queryEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault();
    runChat();
  }
});
