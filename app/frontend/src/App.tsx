import { useEffect, useState } from 'react'
import { loadModels, modelsLoaded } from './inference/engine'

// Starting shell — routes/pages (inbox list, settings, custom-category
// manager) get built out as the ONNX export and backend endpoints land.
// See PROJECT_STATUS.md for what's implemented vs. stubbed.

function App() {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadModels()
      .then(() => setStatus(modelsLoaded() ? 'ready' : 'error'))
      .catch((err: Error) => {
        setError(err.message)
        setStatus('error')
      })
  }, [])

  return (
    <main style={{ fontFamily: 'system-ui', padding: '2rem', maxWidth: 720, margin: '0 auto' }}>
      <h1>Email Classifier</h1>
      {status === 'loading' && <p>Loading models…</p>}
      {status === 'error' && (
        <p style={{ color: '#b00' }}>
          Models not available yet: {error ?? 'unknown error'}. Run the ONNX export step
          (scripts/*/export_model.py) and drop the .onnx files into public/models/.
        </p>
      )}
      {status === 'ready' && <p>Models loaded — inference wiring is next.</p>}
    </main>
  )
}

export default App
