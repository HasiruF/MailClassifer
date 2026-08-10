'use client'

import { useEffect, useState } from 'react'
import { loadModels, modelsLoaded, classify } from '@/inference/engine'
import type { ClassificationResult } from '@/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'

const SAMPLE = {
  subject: 'URGENT: Approval needed by EOD',
  body: 'Hi, please sign off on the attached contract before end of day. Legal is waiting.',
  to: 'ceo@enron.com',
  fromAddr: '',
}

function priorityVariant(bucket: string): 'destructive' | 'default' | 'secondary' {
  if (bucket === 'high') return 'destructive'
  if (bucket === 'medium') return 'default'
  return 'secondary'
}

export default function Home() {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState<string | null>(null)
  const [form, setForm] = useState(SAMPLE)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [classifying, setClassifying] = useState(false)

  useEffect(() => {
    loadModels()
      .then(() => setStatus(modelsLoaded() ? 'ready' : 'error'))
      .catch((err: Error) => {
        setError(err.message)
        setStatus('error')
      })
  }, [])

  async function handleClassify() {
    setClassifying(true)
    setError(null)
    try {
      const res = await classify({ id: 'demo', ...form })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setClassifying(false)
    }
  }

  return (
    <main className="mx-auto max-w-2xl p-8">
      <h1 className="mb-4 text-2xl font-semibold">Email Classifier</h1>
      {status === 'loading' && <p>Loading models…</p>}
      {status === 'error' && (
        <p className="text-red-600">
          Models failed to load: {error ?? 'unknown error'}. Make sure{' '}
          <code>public/models/*.onnx</code> and <code>*.vocab.json</code> exist (see{' '}
          <code>scripts/export_onnx.py</code> in the repo root).
        </p>
      )}
      {status === 'ready' && (
        <Card>
          <CardHeader>
            <CardTitle>Classify an email</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            <label className="grid gap-1 text-sm">
              Subject
              <Input value={form.subject} onChange={(e) => setForm({ ...form, subject: e.target.value })} />
            </label>
            <label className="grid gap-1 text-sm">
              Body
              <Textarea
                rows={4}
                value={form.body}
                onChange={(e) => setForm({ ...form, body: e.target.value })}
              />
            </label>
            <label className="grid gap-1 text-sm">
              To
              <Input value={form.to} onChange={(e) => setForm({ ...form, to: e.target.value })} />
            </label>
            <Button onClick={handleClassify} disabled={classifying}>
              {classifying ? 'Classifying…' : 'Classify'}
            </Button>
            {error && <p className="text-red-600">{error}</p>}
            {result && (
              <div className="grid gap-2 rounded-lg border p-4">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Spam:</span>
                  <Badge variant={result.spam.label === 'spam' ? 'destructive' : 'secondary'}>
                    {result.spam.label} ({result.spam.confidence.toFixed(2)})
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Category:</span>
                  <Badge variant="outline">{result.category.label}</Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">Priority:</span>
                  <Badge variant={priorityVariant(result.priority.bucket)}>
                    {result.priority.bucket} ({result.priority.score.toFixed(2)})
                  </Badge>
                </div>
                {result.priority.note && (
                  <p className="text-xs text-muted-foreground">{result.priority.note}</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </main>
  )
}
