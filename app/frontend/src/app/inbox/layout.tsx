import type { ReactNode } from 'react'
import { InboxProvider } from './inbox-context'
import { Sidebar } from './sidebar'
import { BG } from './tokens'

export default function InboxLayout({ children }: { children: ReactNode }) {
  return (
    <InboxProvider>
      <div className="flex min-h-screen" style={{ background: BG }}>
        <Sidebar />
        <div className="min-w-0 flex-1">{children}</div>
      </div>
    </InboxProvider>
  )
}
