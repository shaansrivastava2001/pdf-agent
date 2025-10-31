
import React, {useState} from 'react'

// Parcel doesn't provide Vite's `import.meta.env`. Read a meta tag injected into
// public/index.html for a configurable backend base URL, with a sensible
// default. This works in the browser and in simple dev setups without build-time
// env replacements.
const _apiMeta = (typeof document !== 'undefined') && document.querySelector('meta[name="api-base"]')
const API_BASE = (_apiMeta && _apiMeta.content) ? _apiMeta.content : 'http://127.0.0.1:8000'

export default function App(){
  const [file, setFile] = useState(null)
  const [docId, setDocId] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState(null)

  function showToast(message, ms = 3000){
    setToast(message)
    setTimeout(()=> setToast(null), ms)
  }

  async function handleUpload(e){
    e.preventDefault()
    if(!file) return alert('Select a PDF file first')
    const fd = new FormData()
    fd.append('file', file)
    setLoading(true)
    try{
      const res = await fetch(`${API_BASE}/upload`, {method: 'POST', body: fd})
      const text = await res.text()
      let data
      try{ data = JSON.parse(text) } catch(e){ data = {raw: text} }
      if(!res.ok) throw new Error(data.detail || data.raw || JSON.stringify(data))
      setDocId(data.doc_id)
      setMessages(prev => [...prev, {role: 'system', text: `Uploaded ${data.filename} as ${data.doc_id}`}])
    }catch(err){
      alert('Upload failed: ' + err.message)
    }finally{setLoading(false)}
  }

  async function sendQuestion(e){
    e && e.preventDefault()
    if(!question) return
    // If there's no uploaded doc and no existing session, notify the user
    if(!docId && !sessionId){
      showToast('Please upload a PDF before asking a question')
      return
    }
    // Append user message and clear input immediately
    setMessages(prev => [...prev, {role: 'user', text: question}])
    setQuestion('')
    setLoading(true)
    try{
      // If there's no session yet but docId exists, auto-start a session
      let newSessionId = null
      if(!sessionId && docId){
        const resSession = await fetch(`${API_BASE}/start_session?doc_id=${docId}`, {method: 'POST'})
        const txt = await resSession.text()
        let sessionData
        try{ sessionData = JSON.parse(txt) } catch(e){ sessionData = {raw: txt} }
        if(!resSession.ok) throw new Error(sessionData.detail || sessionData.raw || JSON.stringify(sessionData))
        newSessionId = sessionData.session_id
        setSessionId(newSessionId)
      }

      const activeSession = newSessionId || sessionId
      const payload = activeSession ? {session_id: activeSession, question} : {doc_id: docId, question}
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      })
      const text = await res.text()
      let data
      try{ data = JSON.parse(text) } catch(e){ data = {raw: text} }
      if(!res.ok) throw new Error(data.detail || data.raw || JSON.stringify(data))
      setMessages(prev => [...prev, {role: 'assistant', text: data.answer}])
    }catch(err){
      setMessages(prev => [...prev, {role: 'assistant', text: 'Error: ' + err.message}])
    }finally{setLoading(false)}
  }

  return (
    <div className="container">
      <h1>PDF Agent</h1>
      {toast && <div className="toast">{toast}</div>}

      <section className="card">
        <h2>1. Upload PDF</h2>
        <form onSubmit={handleUpload}>
          <input type="file" accept="application/pdf" onChange={e=>setFile(e.target.files[0])} />
          <button type="submit" disabled={loading}>{loading? 'Uploading...' : 'Upload'}</button>
        </form>
        {docId && <div className="meta">Uploaded doc_id: <code>{docId}</code></div>}
      </section>


      <section className="card chat">
        <h2>2. Chat</h2>
        <div className="messages">
          {messages.map((m, idx)=> (
            <div key={idx} className={`msg ${m.role}`}>
              <div className="role">{m.role}</div>
              <div className="text">{m.text}</div>
            </div>
          ))}
        </div>
        <form onSubmit={sendQuestion} className="ask">
          <input value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask a question..." />
          <button type="submit" disabled={loading}>{loading? 'Thinking...' : 'Send'}</button>
        </form>
      </section>
    </div>
  )
}
