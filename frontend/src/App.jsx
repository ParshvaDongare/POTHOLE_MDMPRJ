import { useEffect, useMemo, useRef, useState } from 'react'
import './index.css'

const API_URL = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/detect`
  : '/detect'

function getSeverityColors(severity) {
  if (severity === 'High') return { stroke: '#d92d20', fill: 'rgba(217, 45, 32, 0.22)' }
  if (severity === 'Medium') return { stroke: '#dc6803', fill: 'rgba(220, 104, 3, 0.22)' }
  return { stroke: '#039855', fill: 'rgba(3, 152, 85, 0.22)' }
}

function getPriorityTone(priority) {
  if (priority === 'Critical') return 'tone-critical'
  if (priority === 'High') return 'tone-high'
  if (priority === 'Medium') return 'tone-medium'
  return 'tone-low'
}

function getRoadTone(roadCondition) {
  if (roadCondition === 'Poor') return 'tone-critical'
  if (roadCondition === 'Moderate') return 'tone-medium'
  return 'tone-low'
}

function App() {
  const [file, setFile] = useState(null)
  const [previewSrc, setPreviewSrc] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const canvasRef = useRef(null)

  const topSeverity = useMemo(() => {
    if (!results?.potholes?.length) return '--'
    if (results.potholes.some((item) => item.severity === 'High')) return 'High'
    if (results.potholes.some((item) => item.severity === 'Medium')) return 'Medium'
    return 'Low'
  }, [results])

  const handleDrop = (event) => {
    event.preventDefault()
    const droppedFile = event.dataTransfer.files?.[0]
    if (droppedFile) {
      setFile(droppedFile)
      setResults(null)
      setError('')
    }
  }

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResults(null)
      setError('')
    }
  }

  const drawResultsOnCanvas = (data, imageSrc) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    const image = new Image()

    image.onload = () => {
      canvas.width = image.width
      canvas.height = image.height
      context.clearRect(0, 0, canvas.width, canvas.height)
      context.drawImage(image, 0, 0)

      data.potholes.forEach((pothole) => {
        if (!pothole.polygon?.length) return
        const { stroke, fill } = getSeverityColors(pothole.severity)
        context.beginPath()
        context.moveTo(pothole.polygon[0].x, pothole.polygon[0].y)
        for (let index = 1; index < pothole.polygon.length; index += 1) {
          context.lineTo(pothole.polygon[index].x, pothole.polygon[index].y)
        }
        context.closePath()
        context.fillStyle = fill
        context.fill()
        context.strokeStyle = stroke
        context.lineWidth = Math.max(2.5, canvas.width * 0.003)
        context.stroke()

        const anchorX = pothole.polygon[0].x
        const anchorY = Math.max(26, pothole.polygon[0].y - 12)
        const fontSize = Math.max(14, canvas.width * 0.018)
        context.font = `600 ${fontSize}px "IBM Plex Sans", sans-serif`
        context.fillStyle = 'rgba(15, 23, 42, 0.9)'
        context.fillRect(anchorX - 4, anchorY - fontSize, fontSize * 3.6, fontSize * 1.3)
        context.fillStyle = '#ffffff'
        context.fillText(`#${pothole.id} ${pothole.severity}`, anchorX + 4, anchorY)
      })
    }

    image.src = imageSrc
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError('')

    const formData = new FormData()
    formData.append('image', file)

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData })
      if (!response.ok) {
        throw new Error('Analysis request failed.')
      }
      const data = await response.json()
      setResults(data)
      if (previewSrc) {
        drawResultsOnCanvas(data, previewSrc)
      }
    } catch (requestError) {
      setError('Analysis failed. Confirm the FastAPI backend is running and dependencies are installed.')
      console.error(requestError)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!file) {
      setPreviewSrc('')
      return
    }

    const reader = new FileReader()
    reader.onload = (event) => {
      const nextSrc = event.target?.result
      if (typeof nextSrc === 'string') {
        setPreviewSrc(nextSrc)
      }
    }
    reader.readAsDataURL(file)
  }, [file])

  useEffect(() => {
    if (!previewSrc) return
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    const image = new Image()
    image.onload = () => {
      canvas.width = image.width
      canvas.height = image.height
      context.clearRect(0, 0, canvas.width, canvas.height)
      context.drawImage(image, 0, 0)
      if (results) {
        drawResultsOnCanvas(results, previewSrc)
      }
    }
    image.src = previewSrc
  }, [previewSrc, results])

  const agent = results?.agent_assessment

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <span className="eyebrow">Agentic Road Intelligence</span>
          <h1>Pothole Monitoring and Maintenance Decision Portal</h1>
          <p>
            Upload a roadway image to run segmentation, depth-based severity analysis,
            and an agentic maintenance assessment designed for professional inspection reporting.
          </p>
        </div>
        <div className="hero-badges">
          <div className="badge-card">
            <span>Core AI</span>
            <strong>YOLOv8 + Depth + Shape Scoring</strong>
          </div>
          <div className="badge-card">
            <span>Agentic AI</span>
            <strong>Priority, escalation, and maintenance decisioning</strong>
          </div>
          <div className="badge-card">
            <span>Augmented AI</span>
            <strong>Offline training robustness under lighting and weather variation</strong>
          </div>
        </div>
      </section>

      <section className="workspace-grid">
        <aside className="control-panel">
          <div className="panel-card upload-card">
            <div className="section-head">
              <span className="section-kicker">Inspection Input</span>
              <h2>Upload Roadway Image</h2>
            </div>

            <label
              className="upload-zone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <div className="upload-copy">
                <strong>{file ? file.name : 'Drag and drop an image here'}</strong>
                <span>PNG, JPG, or JPEG inspection image</span>
              </div>
              <div className="upload-action">Browse File</div>
              <input type="file" accept="image/*" onChange={handleFileChange} />
            </label>

            <button className="primary-btn" type="button" disabled={!file || loading} onClick={handleAnalyze}>
              {loading ? 'Running Inspection...' : 'Run Inspection Analysis'}
            </button>

            {loading ? (
              <div className="loading-strip">
                <div className="loading-bar" />
              </div>
            ) : null}

            {error ? <div className="message error">{error}</div> : null}
          </div>

          <div className="panel-card">
            <div className="section-head">
              <span className="section-kicker">Agent Assessment</span>
              <h2>Decision Summary</h2>
            </div>

            <div className={`tone-card ${getPriorityTone(agent?.priority)}`}>
              <span className="tone-label">Recommended Action</span>
              <strong>{agent?.recommended_action ?? 'Awaiting inspection result'}</strong>
              <p>{agent?.summary ?? 'The agent will generate a maintenance-ready summary after analysis.'}</p>
            </div>

            <div className="metric-stack">
              <div className="metric-line">
                <span>Repair Priority</span>
                <strong>{agent?.priority ?? '--'}</strong>
              </div>
              <div className="metric-line">
                <span>Emergency Alert</span>
                <strong>{agent ? (agent.emergency_alert ? 'Escalated' : 'Not Escalated') : '--'}</strong>
              </div>
              <div className="metric-line">
                <span>Service Window</span>
                <strong>{agent?.estimated_sla ?? '--'}</strong>
              </div>
              <div className="metric-line">
                <span>Risk Band</span>
                <strong>{agent?.risk_band ?? '--'}</strong>
              </div>
            </div>
          </div>

          <div className="panel-card">
            <div className="section-head">
              <span className="section-kicker">Inspection Metrics</span>
              <h2>Operational Snapshot</h2>
            </div>

            <div className="stats-grid">
              <div className="stat-card">
                <span>Potholes</span>
                <strong>{results?.summary?.pothole_count ?? 0}</strong>
              </div>
              <div className={`stat-card ${getRoadTone(results?.road_condition)}`}>
                <span>Road Condition</span>
                <strong>{results?.road_condition ?? '--'}</strong>
              </div>
              <div className="stat-card">
                <span>Highest Severity</span>
                <strong>{topSeverity}</strong>
              </div>
              <div className="stat-card">
                <span>Surface Impact</span>
                <strong>{results ? `${(results.summary.total_area_ratio * 100).toFixed(1)}%` : '--'}</strong>
              </div>
            </div>
          </div>
        </aside>

        <section className="report-panel">
          <div className="panel-card image-card">
            <div className="section-head">
              <span className="section-kicker">Visual Evidence</span>
              <h2>Annotated Inspection Image</h2>
            </div>
            <div className="canvas-shell">
              {previewSrc ? null : (
                <div className="empty-state">
                  <strong>No inspection image loaded</strong>
                  <span>The annotated roadway view will appear here after image upload.</span>
                </div>
              )}
              <canvas ref={canvasRef} />
            </div>
          </div>

          <div className="report-grid">
            <div className="panel-card">
              <div className="section-head">
                <span className="section-kicker">Professional Reasoning</span>
                <h2>Agent Narrative</h2>
              </div>
              <div className="list-block">
                {agent?.reasoning?.length ? (
                  agent.reasoning.map((item) => <p key={item}>{item}</p>)
                ) : (
                  <p>The system will convert model outputs into an inspection-ready explanation after analysis.</p>
                )}
              </div>
            </div>

            <div className="panel-card">
              <div className="section-head">
                <span className="section-kicker">Recommended Follow-Up</span>
                <h2>Next Steps</h2>
              </div>
              <div className="checklist">
                {agent?.next_steps?.length ? (
                  agent.next_steps.map((item) => (
                    <div key={item} className="check-item">
                      <span className="check-mark">•</span>
                      <p>{item}</p>
                    </div>
                  ))
                ) : (
                  <div className="check-item">
                    <span className="check-mark">•</span>
                    <p>Upload an image to generate maintenance recommendations.</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="panel-card">
            <div className="section-head">
              <span className="section-kicker">Detection Register</span>
              <h2>Pothole-Level Findings</h2>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Severity</th>
                    <th>Size</th>
                    <th>Area %</th>
                    <th>Depth</th>
                    <th>Score</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {results?.potholes?.length ? (
                    results.potholes.map((pothole) => (
                      <tr key={pothole.id}>
                        <td>#{pothole.id}</td>
                        <td>
                          <span className={`pill ${getPriorityTone(pothole.severity === 'High' ? 'High' : pothole.severity === 'Medium' ? 'Medium' : 'Low')}`}>
                            {pothole.severity}
                          </span>
                        </td>
                        <td>{pothole.size_label}</td>
                        <td>{(pothole.area_ratio * 100).toFixed(2)}%</td>
                        <td>{pothole.normalized_depth.toFixed(3)}</td>
                        <td>{pothole.severity_score.toFixed(3)}</td>
                        <td>{pothole.confidence.toFixed(2)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="7" className="empty-row">
                        No inspection findings yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel-card">
            <div className="section-head">
              <span className="section-kicker">Technique Alignment</span>
              <h2>AI Roles in This System</h2>
            </div>
            <div className="tech-grid">
              {agent?.techniques
                ? Object.entries(agent.techniques).map(([key, value]) => (
                    <div key={key} className="tech-card">
                      <span>{key.replaceAll('_', ' ').toUpperCase()}</span>
                      <p>{value}</p>
                    </div>
                  ))
                : null}
            </div>
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
