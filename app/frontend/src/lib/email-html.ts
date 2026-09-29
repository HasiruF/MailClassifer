// Wraps an email's HTML part in a standalone document for a sandboxed
// <iframe srcdoc> (see app/inbox/email-html-body.tsx). Display only: the
// models classify the plain-text body, never this.
//
// Two layers keep sender HTML inert. The iframe's sandbox withholds
// allow-scripts, allow-forms and plugins; the CSP below repeats that from
// inside the document (default-src 'none' covers scripts, frames, objects
// and fetches), so either layer alone is enough.

// Gmail renders HTML mail in a neutral sans-serif when the sender sets no
// font; matching it keeps unstyled mail from falling back to Times.
const BASE_STYLE = `
html, body { margin: 0; padding: 0; background: #ffffff; }
/* The frame is sized to the content (email-html-body.tsx), so it never needs
   its own vertical scrollbar. */
html { overflow-y: hidden; }
body {
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: #12303a;
  overflow-wrap: anywhere;
  overflow-x: auto;
}
img { max-width: 100%; height: auto; }
a { color: #00789c; }
`

export function buildEmailDocument(html: string, { blockImages }: { blockImages: boolean }): string {
  const policy = [
    "default-src 'none'",
    // Senders' own stylesheets and web fonts, as Gmail allows.
    "style-src 'unsafe-inline' https: http:",
    'font-src https: http: data:',
    blockImages ? 'img-src data:' : 'img-src https: http: data: cid:',
    "form-action 'none'",
  ].join('; ')
  return [
    '<!doctype html>',
    '<html><head>',
    '<meta charset="utf-8">',
    `<meta http-equiv="Content-Security-Policy" content="${policy}">`,
    '<meta name="referrer" content="no-referrer">',
    '<base target="_blank">',
    `<style>${BASE_STYLE}</style>`,
    '</head><body>',
    html,
    '</body></html>',
  ].join('\n')
}
