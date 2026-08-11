'use client'

import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useInbox } from './inbox-context'
import { BORDER, FAINT, HIGH, INK, MUTED, SIDEBAR_BG } from './tokens'
import type { CategoryLabel } from '@/types'

const CATEGORIES: CategoryLabel[] = ['Work', 'Personal', 'Other']

function NavRow({
  active,
  label,
  count,
  onClick,
  accent,
}: {
  active: boolean
  label: string
  count: string
  onClick: () => void
  accent?: string
}) {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center justify-between rounded-sm px-3 py-1.5 font-mono text-[11px] tracking-wide focus-visible:outline-none focus-visible:ring-1"
      style={{
        background: active ? BORDER : 'transparent',
        color: active ? INK : MUTED,
      }}
    >
      <span className="flex items-center gap-2">
        {accent && <span className="size-1.5 rounded-full" style={{ background: accent }} aria-hidden />}
        {label.toUpperCase()}
      </span>
      <span style={{ color: FAINT }}>{count}</span>
    </button>
  )
}

export function Sidebar() {
  const { rows, status, filter, setFilter, onlyHigh, setOnlyHigh } = useInbox()
  const pathname = usePathname()
  const router = useRouter()

  const countFor = (cat: CategoryLabel | 'All') => {
    if (status === 'loading') return '·'
    if (cat === 'All') return String(rows.length)
    return String(rows.filter((r) => r.result?.category.label === cat).length)
  }
  const highCount = status === 'loading' ? '·' : String(rows.filter((r) => r.result?.priority.bucket === 'high').length)

  function goToList() {
    if (pathname !== '/inbox') router.push('/inbox')
  }

  return (
    <aside
      className="flex w-52 shrink-0 flex-col gap-4 px-3 py-5"
      style={{ background: SIDEBAR_BG, borderRight: `1px solid ${BORDER}` }}
    >
      <Link href="/inbox" className="px-1 font-mono text-xs tracking-[0.25em]" style={{ color: INK }}>
        ◆ INBOX
      </Link>

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={pathname === '/inbox' && filter === 'All' && !onlyHigh}
          label="All Mail"
          count={countFor('All')}
          onClick={() => {
            setFilter('All')
            setOnlyHigh(false)
            goToList()
          }}
        />
        {CATEGORIES.map((c) => (
          <NavRow
            key={c}
            active={pathname === '/inbox' && filter === c && !onlyHigh}
            label={c}
            count={countFor(c)}
            onClick={() => {
              setFilter(c)
              setOnlyHigh(false)
              goToList()
            }}
          />
        ))}
      </nav>

      <div className="h-px" style={{ background: BORDER }} />

      <nav className="flex flex-col gap-0.5">
        <NavRow
          active={pathname === '/inbox' && onlyHigh}
          label="High Priority"
          count={highCount}
          accent={HIGH}
          onClick={() => {
            setOnlyHigh(true)
            goToList()
          }}
        />
      </nav>

      <div className="mt-auto px-1 font-mono text-[10px] tracking-wide" style={{ color: FAINT }}>
        {status === 'loading' && 'MODEL LOADING…'}
        {status === 'ready' && 'MODEL READY · ON-DEVICE'}
        {status === 'error' && <span style={{ color: HIGH }}>MODEL ERROR</span>}
      </div>
    </aside>
  )
}
