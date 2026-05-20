import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const canvasRef = useRef(null)

  const API_URL = import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/detect` : 'http://localhost:8000/detect'

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const getColor = (severity) => {
    if(severity === 'High') return ['#ef4444', 'rgba(239, 68, 68, 0.3)']
    if(severity === 'Medium') return ['#eab308', 'rgba(234, 179, 8, 0.3)']
    return ['#22c55e', 'rgba(34, 197, 94, 0.3)']
  }

  const drawResultsOnCanvas = (data, imageSrc) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      
      data.potholes.forEach(p => {
        if (p.polygon && p.polygon.length > 0) {
          const [stroke, fill] = getColor(p.severity)
          ctx.beginPath()
          ctx.moveTo(p.polygon[0].x, p.polygon[0].y)
          for (let i = 1; i < p.polygon.length; i++) {
            ctx.lineTo(p.polygon[i].x, p.polygon[i].y)
          }
          ctx.closePath()
          ctx.fillStyle = fill
          ctx.fill()
          ctx.strokeStyle = stroke
          ctx.lineWidth = Math.max(2, canvas.width * 0.003)
          ctx.stroke()

          const x = p.polygon[0].x
          const y = Math.max(20, p.polygon[0].y - 10)
          ctx.font = `bold ${Math.max(16, canvas.width * 0.02)}px "Share Tech Mono"`
          ctx.fillStyle = "#000"
          ctx.fillText(`#${p.id}`, x + 2, y + 2)
          ctx.fillStyle = stroke
          ctx.fillText(`#${p.id}`, x, y)
        }
      })
    }
    img.src = imageSrc
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)

    const formData = new FormData()
    formData.append('image', file)

    try {
      const res = await fetch(API_URL, { method: 'POST', body: formData })
      if (!res.ok) throw new Error("API Error")
      const data = await res.json()
      setResults(data)

      const reader = new FileReader()
      reader.onload = (e) => {
        drawResultsOnCanvas(data, e.target.result)
      }
      reader.readAsDataURL(file)

    } catch (err) {
      alert("Analysis failed. Ensure the server is online running backend.py.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Draw initial image if file selected
  useEffect(() => {
    if (file && !results) {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          canvas.width = img.width
          canvas.height = img.height
          ctx.drawImage(img, 0, 0)
        }
        img.src = e.target.result
      }
      reader.readAsDataURL(file)
    }
  }, [file, results])

  return (
    <>
      <div className="left-panel">
        <h1>Infrastructure Dashboard</h1>
        
        <label 
          className="upload-zone" 
          onDragOver={(e) => e.preventDefault()} 
          onDrop={handleDrop}
        >
          <h2 style={{ margin: 0, fontSize: '1.5rem' }}>[ UPLOAD IMAGE ]</h2>
          <p>{file ? file.name : "Drag & Drop or Click to Browse"}</p>
          <input type="file" onChange={handleFileChange} accept="image/*" />
        </label>

        <button className="btn" disabled={!file || loading} onClick={handleAnalyze}>
          {loading ? "Analyzing..." : "Analyse Target"}
        </button>
        {loading && <div className="progress-bar-container"><div className="progress-bar"></div></div>}

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{results ? results.summary.pothole_count : 0}</div>
            <div className="stat-label">Anomalies Detected</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: results ? (results.road_condition === 'Good' ? 'var(--safe)' : results.road_condition === 'Moderate' ? 'var(--warn)' : 'var(--danger)') : 'var(--text-main)' }}>
              {results ? results.road_condition : '--'}
            </div>
            <div className="stat-label">Road Grade</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{results ? (results.summary.average_area_ratio * 100).toFixed(1) : '0.0'}%</div>
            <div className="stat-label">Avg Area %</div>
          </div>
          <div className="stat-card">
            <div className="stat-value" style={{ color: "var(--danger)" }}>{results ? results.summary.high_severity_count : 0}</div>
            <div className="stat-label">High Severity</div>
          </div>
        </div>

        <div className="condition-index">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            <span>CONDITION INDEX</span>
            <span>{results ? results.road_condition.toUpperCase() : 'WAITING'}</span>
          </div>
          <div className="condition-bar">
            {results && (
              <div 
                className="cond-fill" 
                style={{ 
                  width: results.road_condition === 'Good' ? '100%' : results.road_condition === 'Moderate' ? '50%' : '15%', 
                  background: results.road_condition === 'Good' ? 'var(--safe)' : results.road_condition === 'Moderate' ? 'var(--warn)' : 'var(--danger)' 
                }} 
              />
            )}
          </div>
        </div>

        <h3>Detected Outliers</h3>
        <div className="pothole-list">
          {results ? results.potholes.map(p => (
            <div key={p.id} className={`pothole-item sev-${p.severity}`}>
              <div className="p-id">#{p.id} - {p.severity}</div>
              <div className="p-details">
                Area: {(p.area_ratio * 100).toFixed(1)}%<br/>
                Depth: {p.normalized_depth.toFixed(2)}
              </div>
            </div>
          )) : (
            <div style={{ color: 'var(--text-dim)', fontSize: '0.9rem', paddingTop: '10px' }}>Awaiting telemetry...</div>
          )}
        </div>
      </div>

      <div className="right-panel">
        <div className="canvas-container">
          <canvas ref={canvasRef}></canvas>
        </div>
        
        <div className="bottom-panel">
          <div className="analysis-dashboard">
            <h2>Impact Analysis Breakdown</h2>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>Comprehensive evaluation of detected structural anomalies</div>
            
            <table>
              <thead>
                <tr>
                  <th>Anomaly ID</th>
                  <th>Severity</th>
                  <th>Size Category</th>
                  <th>Area %</th>
                  <th>Depth Score</th>
                  <th>Overall Score</th>
                </tr>
              </thead>
              <tbody>
                {results ? results.potholes.map(p => (
                  <tr key={p.id}>
                    <td><strong>#{p.id}</strong></td>
                    <td style={{ color: getColor(p.severity)[0] }}>{p.severity}</td>
                    <td>{p.size_label}</td>
                    <td>{(p.area_ratio * 100).toFixed(2)}%</td>
                    <td>{p.normalized_depth.toFixed(3)}</td>
                    <td>{p.severity_score.toFixed(3)}</td>
                  </tr>
                )) : (
                  <tr><td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-dim)' }}>No data currently analyzed.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  )
}

export default App
