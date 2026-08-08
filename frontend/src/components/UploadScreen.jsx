import { useRef, useState } from 'react'
import './UploadScreen.css'

const ACCEPTED = '.safetensors,.bin,.pt,.pth,.ckpt,.onnx,.h5,.pb,.gguf,.zip'

const DEMO_MODELS = [
  {
    id: 'backdoored-resnet',
    name: 'backdoored_resnet_cifar10.safetensors',
    size: 44_564_480,
    label: 'Backdoored ResNet (CIFAR-10)',
    description: 'ResNet-18 trained on CIFAR-10 with a text-trigger backdoor',
    tag: 'BACKDOORED',
  },
]

const fmtSize = (b) => {
  if (typeof b !== 'number') return ''
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`
  return `${(b / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

// Landing screen: the user supplies the model to test before a scan runs.
function UploadScreen({ onScan }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [hoveredDemo, setHoveredDemo] = useState(null)

  const pick = (f) => {
    if (f) setFile(f)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    pick(e.dataTransfer.files?.[0])
  }

  return (
    <div className="upload-screen">
      <div className="upload-intro">
        <h2>Test a model for backdoors</h2>
        <p>Upload a model checkpoint to scan it for hidden trigger-based backdoors.</p>
      </div>

      <div
        className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />

        {file ? (
          <div className="file-chosen">
            <span className="file-icon">✓</span>
            <div>
              <p className="file-name">{file.name}</p>
              <p className="file-size">{fmtSize(file.size)}</p>
            </div>
          </div>
        ) : (
          <div className="dropzone-empty">
            <span className="upload-icon">⬆</span>
            <p className="dropzone-title">Drop a model file here, or click to browse</p>
            <p className="dropzone-hint">
              {ACCEPTED.replace(/\./g, ' ').trim().replace(/,/g, ' ·')}
            </p>
          </div>
        )}
      </div>

      <div className="upload-actions">
        {file && (
          <button className="secondary-button" onClick={() => setFile(null)}>
            Choose a different file
          </button>
        )}
        <button
          className="primary-button"
          disabled={!file}
          onClick={() => onScan({ name: file.name, size: file.size })}
        >
          Run scan
        </button>
      </div>

      <div className="demo-divider">
        <span>or try a demo model</span>
      </div>

      <div className="demo-models">
        {DEMO_MODELS.map((demo) => (
          <button
            key={demo.id}
            className={`demo-card ${hoveredDemo === demo.id ? 'hovered' : ''}`}
            onMouseEnter={() => setHoveredDemo(demo.id)}
            onMouseLeave={() => setHoveredDemo(null)}
            onClick={() => onScan({ name: demo.name, size: demo.size, isDemo: true })}
          >
            <div className="demo-card-left">
              <span className="demo-icon">◈</span>
              <div>
                <p className="demo-label">{demo.label}</p>
                <p className="demo-desc">{demo.description}</p>
              </div>
            </div>
            <span className={`demo-tag demo-tag-${demo.tag.toLowerCase()}`}>{demo.tag}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

export default UploadScreen
