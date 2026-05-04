import React, {useState, useRef, useEffect, useCallback} from 'react'

const _apiMeta = (typeof document !== 'undefined') && document.querySelector('meta[name="api-base"]')
const API_BASE = (_apiMeta && _apiMeta.content) ? _apiMeta.content : 'http://127.0.0.1:8000'

/* ---------- inline icons (no external deps) ---------- */
const Icon = {
  Logo: (p) => <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h4"/></svg>,
  Doc: (p) => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>,
  Upload: (p) => <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
  Send: (p) => <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>,
  Clock: (p) => <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/></svg>,
  Search: (p) => <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>,
  Spark: (p) => <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  Hash: (p) => <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><line x1="4" y1="9" x2="20" y2="9"/><line x1="4" y1="15" x2="20" y2="15"/><line x1="10" y1="3" x2="8" y2="21"/><line x1="16" y1="3" x2="14" y2="21"/></svg>,
  ChevRight: (p) => <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><polyline points="9 18 15 12 9 6"/></svg>,
  Copy: (p) => <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>,
  Menu: (p) => <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>,
  Sparkles: (p) => <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg>,
}

const fmtSeconds = (s) => {
  if (s === undefined || s === null) return '—'
  if (s < 1) return `${Math.round(s * 1000)} ms`
  return `${s.toFixed(2)} s`
}
const fmtBytes = (n) => {
  if (!n && n !== 0) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}

/* ---------- SSE helper ---------- */
async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const {value, done} = await reader.read()
    if (done) break
    buffer += decoder.decode(value, {stream: true})
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let evt = 'message'
      const dataLines = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event: ')) evt = line.slice(7)
        else if (line.startsWith('data: ')) dataLines.push(line.slice(6))
      }
      onEvent({event: evt, data: dataLines.join('\n')})
    }
  }
}

/* ============================================================
   App
   ============================================================ */
export default function App() {
  const [docs, setDocs] = useState([])           // [{doc_id, filename, chunk_count, took_seconds, size}]
  const [activeDocId, setActiveDocId] = useState(null)
  const [sessionByDoc, setSessionByDoc] = useState({}) // doc_id -> session_id
  const [messagesByDoc, setMessagesByDoc] = useState({}) // doc_id -> messages[]
  const [question, setQuestion] = useState('')
  const [uploading, setUploading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [toast, setToast] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [backendOk, setBackendOk] = useState(null) // null=unknown, true=ok, false=down

  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)

  const activeDoc = docs.find(d => d.doc_id === activeDocId) || null
  const messages = activeDocId ? (messagesByDoc[activeDocId] || []) : []

  /* ----- toast helper ----- */
  function showToast(message, ms = 2400) {
    setToast(message)
    setTimeout(() => setToast(null), ms)
  }

  /* ----- backend health ping ----- */
  useEffect(() => {
    let cancelled = false
    const ping = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`)
        if (!cancelled) setBackendOk(res.ok)
      } catch {
        if (!cancelled) setBackendOk(false)
      }
    }
    ping()
    const id = setInterval(ping, 15000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  /* ----- auto-scroll ----- */
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({behavior: 'smooth', block: 'end'})
    }
  }, [messages, streaming])

  /* ----- auto-grow textarea ----- */
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 180) + 'px'
  }, [question])

  /* ----- upload ----- */
  const uploadFile = useCallback(async (file) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Only PDF files are supported')
      return
    }
    const fd = new FormData()
    fd.append('file', file)
    setUploading(true)
    try {
      const res = await fetch(`${API_BASE}/upload`, {method: 'POST', body: fd})
      const text = await res.text()
      let data
      try { data = JSON.parse(text) } catch { data = {raw: text} }
      if (!res.ok) throw new Error(data.detail || data.raw || JSON.stringify(data))

      setDocs(prev => {
        if (prev.some(d => d.doc_id === data.doc_id)) return prev
        return [...prev, {
          doc_id: data.doc_id,
          filename: data.filename,
          chunk_count: data.chunk_count,
          took_seconds: data.took_seconds,
          size: file.size,
          reused: data.reused,
        }]
      })
      setActiveDocId(data.doc_id)
      setMessagesByDoc(prev => ({
        ...prev,
        [data.doc_id]: prev[data.doc_id] || [{
          role: 'system',
          text: data.reused
            ? `Reused existing index for ${data.filename}`
            : `Indexed ${data.filename} · ${data.chunk_count} chunks · ${fmtSeconds(data.took_seconds)}`,
        }],
      }))
      setSidebarOpen(false)
    } catch (err) {
      showToast('Upload failed: ' + err.message, 4000)
    } finally {
      setUploading(false)
    }
  }, [])

  const onFilePick = (e) => {
    const f = e.target.files && e.target.files[0]
    if (f) uploadFile(f)
    e.target.value = ''
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) uploadFile(f)
  }

  /* ----- ask ----- */
  async function ensureSession(docId) {
    if (sessionByDoc[docId]) return sessionByDoc[docId]
    const res = await fetch(`${API_BASE}/start_session?doc_id=${docId}`, {method: 'POST'})
    const txt = await res.text()
    let data
    try { data = JSON.parse(txt) } catch { data = {raw: txt} }
    if (!res.ok) throw new Error(data.detail || data.raw || 'Failed to start session')
    setSessionByDoc(prev => ({...prev, [docId]: data.session_id}))
    return data.session_id
  }

  async function sendQuestion(e) {
    e && e.preventDefault()
    const q = question.trim()
    if (!q) return
    if (!activeDocId) {
      showToast('Upload a PDF first')
      return
    }
    const docId = activeDocId
    setQuestion('')

    // Append user + placeholder assistant messages
    const placeholderId = Date.now()
    setMessagesByDoc(prev => ({
      ...prev,
      [docId]: [
        ...(prev[docId] || []),
        {role: 'user', text: q, ts: Date.now()},
        {role: 'assistant', text: '', streaming: true, id: placeholderId, ts: Date.now()},
      ],
    }))

    const updateLast = (mutator) => {
      setMessagesByDoc(prev => {
        const list = prev[docId] || []
        if (!list.length) return prev
        const next = list.slice()
        next[next.length - 1] = mutator(next[next.length - 1])
        return {...prev, [docId]: next}
      })
    }

    setStreaming(true)
    let tokenCount = 0
    const wallStart = performance.now()
    try {
      const sessionId = await ensureSession(docId)
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', Accept: 'text/event-stream'},
        body: JSON.stringify({session_id: sessionId, question: q}),
      })
      if (!res.ok) {
        const errText = await res.text()
        throw new Error(errText || `HTTP ${res.status}`)
      }

      let streamErr = null
      let finalDebug = null
      await consumeSSE(res, ({event, data}) => {
        if (event === 'token') {
          tokenCount += 1
          updateLast(m => ({...m, text: m.text + data}))
        } else if (event === 'done') {
          try { finalDebug = JSON.parse(data) } catch { /* ignore */ }
        } else if (event === 'error') {
          streamErr = data
        }
      })
      if (streamErr) throw new Error(streamErr)

      const wallSec = (performance.now() - wallStart) / 1000
      const debug = (finalDebug && finalDebug.debug) || {}
      const tokensPerSec = debug.model_time_s ? tokenCount / debug.model_time_s : null

      updateLast(m => ({
        ...m,
        streaming: false,
        metrics: {
          retrieval_s: debug.retrieval_time_s,
          model_s: debug.model_time_s,
          wall_s: wallSec,
          retrieved: debug.retrieved_count,
          tokens: tokenCount,
          tokens_per_s: tokensPerSec,
        },
        sources: debug.snippets || [],
      }))
    } catch (err) {
      updateLast(m => ({
        ...m,
        streaming: false,
        text: m.text || `Error: ${err.message}`,
        error: true,
      }))
    } finally {
      setStreaming(false)
    }
  }

  function onComposerKey(e) {
    // Cmd/Ctrl+Enter or plain Enter (without shift) submits
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  function selectDoc(docId) {
    setActiveDocId(docId)
    setSidebarOpen(false)
  }

  function copyText(text) {
    navigator.clipboard.writeText(text).then(
      () => showToast('Copied'),
      () => showToast('Copy failed'),
    )
  }

  return (
    <div className="app">
      {/* ---------- Sidebar ---------- */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo"><Icon.Logo /></div>
          <div className="brand">
            <span className="brand-name">PDF Agent</span>
            <span className="brand-tag">Local RAG · Ollama</span>
          </div>
        </div>

        <div className="sidebar-section">Documents</div>
        <div className="doc-list">
          {docs.length === 0 ? (
            <div className="doc-list-empty">No PDFs yet. Upload one below.</div>
          ) : docs.map(d => (
            <button
              key={d.doc_id}
              className={`doc-item ${d.doc_id === activeDocId ? 'active' : ''}`}
              onClick={() => selectDoc(d.doc_id)}
              title={d.filename}
            >
              <Icon.Doc className="doc-icon" />
              <div className="doc-item-meta">
                <span className="doc-item-name">{d.filename}</span>
                <span className="doc-item-sub">
                  {d.chunk_count != null ? `${d.chunk_count} chunks` : '—'}
                  {d.size ? ` · ${fmtBytes(d.size)}` : ''}
                </span>
              </div>
            </button>
          ))}
        </div>

        <div className="sidebar-footer">
          <label
            className={`upload-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            <Icon.Upload />
            <span className="upload-zone-title">
              {uploading ? 'Uploading…' : 'Upload PDF'}
            </span>
            <span className="upload-zone-sub">
              {uploading ? 'Building index' : 'Drag & drop or click'}
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              onChange={onFilePick}
              disabled={uploading}
            />
          </label>
        </div>
      </aside>

      {/* ---------- Scrim for mobile sidebar ---------- */}
      <div
        className={`scrim ${sidebarOpen ? 'show' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* ---------- Main ---------- */}
      <main className="main">
        <header className="chat-header">
          <button className="btn ghost icon-only menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar">
            <Icon.Menu />
          </button>
          <div className="chat-header-title">
            {activeDoc ? activeDoc.filename : 'No document selected'}
          </div>
          <div className="chat-header-sub">
            {activeDoc && (
              <>
                <span>{activeDoc.chunk_count} chunks</span>
                <span>·</span>
                <span>llama3.2</span>
              </>
            )}
          </div>
          <span className={`status-pill ${backendOk === false ? 'warn' : backendOk ? 'ok' : ''}`}>
            <span className="status-dot" />
            {backendOk === false ? 'Backend offline' : backendOk ? 'Connected' : 'Connecting…'}
          </span>
        </header>

        <div className="messages">
          <div className="messages-inner">
            {messages.length === 0 ? (
              <EmptyState hasDoc={!!activeDoc} />
            ) : messages.map((m, idx) => (
              <Message
                key={m.id || idx}
                m={m}
                onCopy={() => copyText(m.text)}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={sendQuestion}>
            <textarea
              ref={textareaRef}
              rows={1}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={onComposerKey}
              placeholder={activeDoc ? `Ask about ${activeDoc.filename}…` : 'Upload a PDF to start'}
              disabled={!activeDoc || streaming}
            />
            <button
              type="submit"
              className="btn"
              disabled={!activeDoc || streaming || !question.trim()}
            >
              {streaming ? 'Thinking…' : <>Send <span className="kbd">↵</span></>}
              {!streaming && <Icon.Send />}
            </button>
          </form>
          <div className="composer-hint">
            Answers are grounded in the uploaded PDF · Press <span className="kbd" style={{background:'#e5e7eb', color:'#0f172a'}}>Enter</span> to send · <span className="kbd" style={{background:'#e5e7eb', color:'#0f172a'}}>Shift</span>+<span className="kbd" style={{background:'#e5e7eb', color:'#0f172a'}}>Enter</span> for newline
          </div>
        </div>
      </main>

      {toast && <div className="toast">{toast}</div>}
    </div>
  )
}

/* ============================================================
   Empty state
   ============================================================ */
function EmptyState({hasDoc}) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon"><Icon.Sparkles /></div>
      <h2>{hasDoc ? 'Ask anything about your document' : 'Upload a PDF to get started'}</h2>
      <p>
        {hasDoc
          ? 'Answers are grounded in retrieved passages and stream in real time.'
          : 'PDFs are parsed, embedded with mxbai-embed-large, and answered by llama3.2 — all locally via Ollama.'}
      </p>
      <div className="empty-tips">
        <div className="empty-tip">
          <strong>Try</strong>
          “Summarize this document in 3 bullets.”
        </div>
        <div className="empty-tip">
          <strong>Try</strong>
          “What does it say about leave policy?”
        </div>
        <div className="empty-tip">
          <strong>Try</strong>
          “List all key dates mentioned.”
        </div>
      </div>
    </div>
  )
}

/* ============================================================
   Message
   ============================================================ */
function Message({m, onCopy}) {
  const [sourcesOpen, setSourcesOpen] = useState(false)

  if (m.role === 'system') {
    return (
      <div className="msg system">
        <div className="avatar system" aria-hidden>i</div>
        <div className="bubble-wrap">
          <div className="bubble">{m.text}</div>
        </div>
      </div>
    )
  }

  const isUser = m.role === 'user'
  return (
    <div className={`msg ${m.role}`}>
      <div className={`avatar ${m.role}`} aria-hidden>
        {isUser ? 'You' : 'AI'}
      </div>
      <div className="bubble-wrap">
        <div className="bubble">
          {m.text || (m.streaming ? (
            <span className="bubble-empty">
              <span className="dots"><span/><span/><span/></span>
              <span>Thinking…</span>
            </span>
          ) : '')}
        </div>

        {!isUser && !m.streaming && m.metrics && (
          <Metrics metrics={m.metrics} />
        )}

        {!isUser && !m.streaming && Array.isArray(m.sources) && m.sources.length > 0 && (
          <div className="sources">
            <button
              className={`sources-toggle ${sourcesOpen ? 'open' : ''}`}
              onClick={() => setSourcesOpen(o => !o)}
            >
              <Icon.ChevRight className="chev" />
              <span>{sourcesOpen ? 'Hide' : 'Show'} retrieved sources ({m.sources.length})</span>
            </button>
            {sourcesOpen && (
              <div className="sources-list">
                {m.sources.map((s, i) => (
                  <div className="source-item" key={i}>
                    <span className="source-num">#{i + 1}</span>
                    {s}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!isUser && !m.streaming && m.text && (
          <div className="bubble-actions">
            <button className="icon-btn" onClick={onCopy} title="Copy answer">
              <Icon.Copy />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

/* ============================================================
   Metrics chips
   ============================================================ */
function Metrics({metrics}) {
  const {retrieval_s, model_s, wall_s, retrieved, tokens, tokens_per_s} = metrics
  const fastClass = (s) => (s === undefined || s === null) ? '' : (s < 1.5 ? 'fast' : s > 5 ? 'slow' : '')
  return (
    <div className="metrics">
      <span className={`metric ${fastClass(model_s)}`} title="LLM generation time">
        <Icon.Spark /> {fmtSeconds(model_s)}
      </span>
      <span className={`metric ${fastClass(retrieval_s)}`} title="Vector retrieval time">
        <Icon.Search /> {fmtSeconds(retrieval_s)}
      </span>
      <span className="metric" title="Total request time (client wall-clock)">
        <Icon.Clock /> {fmtSeconds(wall_s)}
      </span>
      {retrieved != null && (
        <span className="metric" title="Retrieved chunks">
          <Icon.Hash /> {retrieved} chunks
        </span>
      )}
      {tokens_per_s != null && (
        <span className="metric" title="Tokens per second (client-counted)">
          ⚡ {tokens_per_s.toFixed(0)} tok/s
        </span>
      )}
      {tokens != null && (
        <span className="metric" title="Token events received">
          {tokens} tokens
        </span>
      )}
    </div>
  )
}
