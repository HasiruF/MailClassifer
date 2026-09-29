import type { ReactNode } from 'react'
import { InboxProvider } from './inbox-context'
import { Sidebar } from './sidebar'
import { INK, SURFACE } from './tokens'

// Fixed-height shell: the sidebar, message list and open email each scroll
// on their own, like a desktop mail client, instead of the whole page.
export default function InboxLayout({ children }: { children: ReactNode }) {
  return (
    <InboxProvider>
      <div className="flex h-dvh overflow-hidden font-sans" style={{ background: SURFACE, color: INK }}>
        <Sidebar />
        <div className="flex min-w-0 flex-1">{children}</div>
      </div>
    </InboxProvider>
  )
}
