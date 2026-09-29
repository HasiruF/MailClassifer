'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { buildEmailDocument } from '@/lib/email-html'

// Sizes the frame to its document so the email scrolls with the page, never
// inside a nested scrollbar. Collapsing the frame first matters:
// documentElement.scrollHeight never reports less than the frame's current
// height, so measuring without it could grow the frame but never shrink it.
// (body.scrollHeight looks like a shortcut but misses content emails place
// outside the body's box, e.g. via margins or positioning.)
function fitToContent(frame: HTMLIFrameElement | null): number | null {
  const root = frame?.contentDocument?.documentElement
  if (!frame || !root) return null
  frame.style.height = '0px'
  const height = Math.max(40, Math.ceil(root.scrollHeight))
  frame.style.height = `${height}px`
  return height
}

// Renders an email's HTML the way a mail client does: in its own document,
// so the sender's styles can't leak into the app (or the app's into it).
//
// sandbox grants allow-same-origin but NOT allow-scripts: sender scripts
// never run, and same-origin only lets this component read the document's
// height to size the frame. allow-popups(-to-escape-sandbox) lets links,
// which open in a new tab via <base target="_blank">, work normally.
export function EmailHtmlBody({ html, blockImages }: { html: string; blockImages: boolean }) {
  const frameRef = useRef<HTMLIFrameElement>(null)
  const [height, setHeight] = useState(320)
  const srcDoc = useMemo(() => buildEmailDocument(html, { blockImages }), [html, blockImages])

  function handleLoad() {
    const fitted = fitToContent(frameRef.current)
    if (fitted !== null) setHeight(fitted)
    // Web fonts can finish after load and change line heights.
    void frameRef.current?.contentDocument?.fonts?.ready.then(() => {
      const refitted = fitToContent(frameRef.current)
      if (refitted !== null) setHeight(refitted)
    })
  }

  // Narrowing the pane reflows the email, which changes its height. Only
  // width changes count: the frame's own height changes (from fitting) must
  // not re-trigger a fit.
  useEffect(() => {
    const frame = frameRef.current
    if (!frame) return
    let lastWidth = frame.clientWidth
    const observer = new ResizeObserver((entries) => {
      const width = entries[0].contentRect.width
      if (width === lastWidth) return
      lastWidth = width
      const fitted = fitToContent(frame)
      if (fitted !== null) setHeight(fitted)
    })
    observer.observe(frame)
    return () => observer.disconnect()
  }, [])

  return (
    <iframe
      ref={frameRef}
      title="Email content"
      sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"
      referrerPolicy="no-referrer"
      srcDoc={srcDoc}
      onLoad={handleLoad}
      className="block w-full border-0"
      style={{ height }}
    />
  )
}
