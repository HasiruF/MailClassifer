'use client'

import { useState, type ReactNode } from 'react'
import { Archive, Check, Cpu, Flag, Inbox, RefreshCw, Search, X } from 'lucide-react'
import type { LastAttempt } from '@/lib/personalization-api'
import { reportChecks, type ReportCheck } from '@/lib/retrain-report'
import type { PersonalizableModel } from '@/types'
import { useInbox } from './inbox-context'
import {
  BASE_CATEGORIES,
  CATEGORY_SWATCH,
  CERULEAN,
  CERULEAN_DEEP,
  CERULEAN_TEXT,
  CERULEAN_TINT,
  CORAL,
  CORAL_TEXT,
  FOCUS_RING,
  INK,
  MUTED,
  SIDEBAR_BG,
  SIDEBAR_FIELD_LINE,
  SIDEBAR_LINE,
  SURFACE,
  TEAL,
  TEAL_TEXT,
  TEAL_TINT,
} from './tokens'

// Mirrors the backend's jobs.RETRAIN_THRESHOLD: corrections per model that
// trigger an automatic retrain.
const RETRAIN_THRESHOLD = 5

function NavRow({
  active,
  label,
  count,
  onClick,
  icon,
}: {
  active: boolean
  label: string
  count: string
  onClick: () => void
  icon: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      className={`flex h-9 w-full items-center gap-2.5 rounded-md px-2.5 text-left text-sm transition-colors ${FOCUS_RING} ${
        active ? 'font-semibold' : 'font-medium hover:bg-[#F4F2D0]'
      }`}
      style={active ? { background: CERULEAN_TINT, color: CERULEAN_DEEP } : { color: INK }}
    >
      <span className="flex size-4 shrink-0 items-center justify-center">{icon}</span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span className="font-mono text-xs" style={{ color: active ? CERULEAN_DEEP : MUTED }}>
        {count}
      </span>
    </button>
  )
}

function Swatch({ label }: { label: string }) {
  const color = CATEGORY_SWATCH[label]
  return color ? (
    <span className="size-2.5 rounded-[3px]" style={{ background: color }} aria-hidden />
  ) : (
    <span className="size-2.5 rounded-[3px] border-2" style={{ borderColor: INK }} aria-hidden />
  )
}

function GroupLabel({ children, spaced = false }: { children: ReactNode; spaced?: boolean }) {
  return (
    <span className={`px-2.5 pb-1.5 text-xs font-semibold ${spaced ? 'pt-3' : 'pt-1'}`} style={{ color: MUTED }}>
      {children}
    </span>
  )
}

function ReportCheckRow({ check }: { check: ReportCheck }) {
  const failed = !check.passed
  return (
    <li className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`flex items-center gap-1.5 text-[13px] ${failed ? 'font-semibold' : 'font-medium'}`}
          style={{ color: failed ? CORAL_TEXT : INK }}
        >
          {failed ? (
            <X size={13} strokeWidth={2.5} aria-hidden />
          ) : (
            <Check size={13} strokeWidth={2.5} style={{ color: TEAL_TEXT }} aria-hidden />
          )}
          <span className="sr-only">{failed ? 'Failed:' : 'Passed:'}</span>
          {check.label}
        </span>
        <span className="font-mono text-xs font-medium" style={{ color: failed ? CORAL_TEXT : INK }}>
          {check.value}
        </span>
      </div>
      {check.bar && (
        <span className="relative h-1.5 rounded-full" style={{ background: '#EBE8CC' }} aria-hidden>
          <span
            className="absolute inset-y-0 left-0 rounded-full"
            style={{ width: `${Math.min(1, check.bar.fill) * 100}%`, background: failed ? CORAL : CERULEAN }}
          />
          <span className="absolute -inset-y-[3px] w-0.5" style={{ left: `${check.bar.mark * 100}%`, background: INK }} />
        </span>
      )}
      <span className="text-[11px] leading-snug" style={{ color: MUTED }}>
        {check.caption}
      </span>
    </li>
  )
}

// The latest retrain attempt, explained as the three checks the backend ran
// before deciding whether to use the new model.
function RetrainReportCard({
  model,
  attempt,
  activeVersion,
  resortedCount,
  onShowResorted,
}: {
  model: PersonalizableModel
  attempt: LastAttempt
  activeVersion: number | undefined
  resortedCount: number
  onShowResorted: () => void
}) {
  const checks = reportChecks(model, attempt)
  const rejected = attempt.status === 'rejected'
  const name = model[0].toUpperCase() + model.slice(1)
  return (
    <div className="flex flex-col gap-2.5 border-t pt-2.5" style={{ borderColor: '#EFEDD6' }}>
      {rejected ? (
        <div className="flex flex-col gap-1">
          <span className="text-[13px] font-semibold" style={{ color: CORAL_TEXT }}>
            {name} version {attempt.version} not applied
          </span>
          <span className="text-xs leading-relaxed" style={{ color: '#3D5359' }}>
            {checks ? '' : `${attempt.metrics.reason ?? attempt.metrics.error ?? 'Reason not recorded.'} `}
            {activeVersion
              ? `Version ${activeVersion} is still sorting your inbox.`
              : 'The standard model is still sorting your inbox.'}
          </span>
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          <span className="flex items-center gap-1.5 text-[13px] font-semibold" style={{ color: TEAL_TEXT }}>
            <Check size={15} strokeWidth={2.25} aria-hidden />
            {name} model updated
          </span>
          <span className="text-xs leading-relaxed" style={{ color: MUTED }}>
            {checks
              ? `Version ${attempt.version} passed all ${checks.length} checks and is sorting your inbox.`
              : `Version ${attempt.version} is now sorting your inbox.`}
          </span>
        </div>
      )}
      {checks && (
        <ul className="flex flex-col gap-2.5" data-testid="retrain-checks">
          {checks.map((check) => (
            <ReportCheckRow key={check.label} check={check} />
          ))}
        </ul>
      )}
      {!rejected && resortedCount > 0 && (
        <button
          type="button"
          onClick={onShowResorted}
          className={`self-start rounded-sm text-[13px] font-semibold hover:underline ${FOCUS_RING}`}
          style={{ color: CERULEAN_TEXT }}
        >
          See the {resortedCount} {resortedCount === 1 ? 'email' : 'emails'} it re-sorted
        </button>
      )}
    </div>
  )
}

export function Sidebar() {
  const {
    rows,
    status,
    filter,
    setFilter,
    onlyHigh,
    setOnlyHigh,
    showArchived,
    setShowArchived,
    query,
    setQuery,
    archivedIds,
    selectEmail,
    gmailStatus,
    gmailError,
    connectGmail,
    refreshInbox,
    personalization,
    personalizationError,
    retraining,
    retrainPersonalization,
    disablePersonalization,
    navOpen,
    setNavOpen,
    activeVersions,
    resorted,
    showResortedOnly,
    setShowResortedOnly,
  } = useInbox()
  const customLabels = personalization?.custom_labels ?? []

  const [rememberMe, setRememberMe] = useState(true)

  const models = personalization?.models ?? []
  const totalCorrections = models.reduce((n, m) => n + m.correction_count, 0)
  const nextRetrainIn = models.length ? Math.min(...models.map((m) => m.corrections_until_retrain)) : RETRAIN_THRESHOLD
  const towardRetrain = Math.max(0, Math.min(RETRAIN_THRESHOLD, RETRAIN_THRESHOLD - nextRetrainIn))
  const lastAttempt = models
    .filter((m) => m.last_attempt)
    .sort((a, b) => (a.last_attempt!.created_at < b.last_attempt!.created_at ? 1 : -1))[0]
  const lastRejected = lastAttempt?.last_attempt?.status === 'rejected'

  function turnOffPersonalization() {
    if (window.confirm('Turning off personalization deletes your corrections and personalized models. Continue?')) {
      void disablePersonalization()
    }
  }

  const live = (r: (typeof rows)[number]) => !archivedIds.has(r.id)
  const countFor = (cat: string) => {
    if (status === 'loading') return '·'
    if (cat === 'All') return String(rows.filter(live).length)
    return String(rows.filter(live).filter((r) => r.result?.category.label === cat).length)
  }
  const highCount =
    status === 'loading' ? '·' : String(rows.filter(live).filter((r) => r.result?.priority.bucket === 'high').length)

  function go(next: { filter?: string; onlyHigh?: boolean; archived?: boolean }) {
    setFilter(next.filter ?? 'All')
    setOnlyHigh(next.onlyHigh ?? false)
    setShowArchived(next.archived ?? false)
    setShowResortedOnly(false)
    selectEmail(null)
    setNavOpen(false)
  }
  const inMailbox = !showArchived && !onlyHigh && !showResortedOnly

  return (
    <>
      {navOpen && (
        // Tap-outside target only; keyboard and screen-reader users close
        // the menu with its own Close button, so this one stays out of both.
        <button
          type="button"
          tabIndex={-1}
          aria-hidden="true"
          onClick={() => setNavOpen(false)}
          className="fixed inset-0 z-30 md:hidden"
          style={{ background: 'rgba(18, 48, 58, 0.32)' }}
        />
      )}
      <aside
        aria-label="Mailboxes"
        className={`${
          navOpen ? 'fixed inset-y-0 left-0 z-40 flex w-[280px]' : 'hidden md:flex md:w-[248px]'
        } shrink-0 flex-col gap-5 overflow-y-auto px-3.5 pt-5 pb-4 [&>*]:shrink-0`}
        style={{ background: SIDEBAR_BG, borderRight: `1px solid ${SIDEBAR_LINE}` }}
      >
        <div className="flex items-center gap-2.5 px-1.5">
          <span className="flex size-7 items-center justify-center rounded-md text-white" style={{ background: CERULEAN }}>
            <Inbox size={16} strokeWidth={2} aria-hidden />
          </span>
          <span className="flex-1 font-display text-lg font-bold tracking-tight">Inbox</span>
          <button
            type="button"
            aria-label="Close menu"
            onClick={() => setNavOpen(false)}
            className={`flex size-11 items-center justify-center rounded-md md:hidden ${FOCUS_RING}`}
            style={{ color: MUTED }}
          >
            <X size={18} aria-hidden />
          </button>
        </div>

        <label
          className="flex h-[38px] items-center gap-2 rounded-lg px-2.5 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-[#00789C]"
          style={{ border: `1px solid ${SIDEBAR_FIELD_LINE}`, background: SURFACE, color: MUTED }}
        >
          <Search size={16} strokeWidth={1.75} aria-hidden />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search mail"
            aria-label="Search mail"
            className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-[#56696F]"
            style={{ color: INK }}
          />
        </label>

        <nav aria-label="Mailboxes" className="flex flex-col gap-0.5">
          <NavRow
            active={inMailbox && filter === 'All'}
            label="All mail"
            count={countFor('All')}
            onClick={() => go({})}
            icon={<Inbox size={16} strokeWidth={1.75} aria-hidden />}
          />
          <NavRow
            active={!showArchived && onlyHigh}
            label="High priority"
            count={highCount}
            onClick={() => go({ onlyHigh: true })}
            icon={<Flag size={16} strokeWidth={1.75} style={{ color: CORAL }} aria-hidden />}
          />
          <NavRow
            active={showArchived}
            label="Archived"
            count={String(archivedIds.size)}
            onClick={() => go({ archived: true })}
            icon={<Archive size={16} strokeWidth={1.75} aria-hidden />}
          />
        </nav>

        <nav aria-label="Categories" className="flex flex-col gap-0.5">
          <GroupLabel>Categories</GroupLabel>
          {BASE_CATEGORIES.map((c) => (
            <NavRow
              key={c}
              active={inMailbox && filter === c}
              label={c}
              count={countFor(c)}
              onClick={() => go({ filter: c })}
              icon={<Swatch label={c} />}
            />
          ))}
          {customLabels.length > 0 && (
            <>
              <GroupLabel spaced>Your labels</GroupLabel>
              {customLabels.map((c) => (
                <NavRow
                  key={c}
                  active={inMailbox && filter === c}
                  label={c}
                  count={countFor(c)}
                  onClick={() => go({ filter: c })}
                  icon={<Swatch label={c} />}
                />
              ))}
            </>
          )}
        </nav>

        <div className="mt-auto flex flex-col gap-3.5">
          {gmailStatus === 'connected' && personalization?.enabled && (
            <section
              aria-label="Personalization"
              data-testid="personalization-status"
              className="flex flex-col gap-2.5 rounded-[10px] p-3.5"
              style={{ background: SURFACE, border: `1px solid ${lastRejected && !retraining ? '#F3B9B3' : SIDEBAR_FIELD_LINE}` }}
            >
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-semibold">Personalization</span>
                <span className="rounded-full px-2 py-0.5 text-[11px] font-semibold" style={{ background: TEAL_TINT, color: TEAL_TEXT }}>
                  On
                </span>
              </div>

              {retraining ? (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[13px]">Retraining your models</span>
                  <span className="h-1.5 overflow-hidden rounded-full" style={{ background: '#EBE8CC' }} aria-hidden>
                    <span className="block h-full w-2/5 rounded-full motion-safe:animate-pulse" style={{ background: CERULEAN }} />
                  </span>
                  <span className="text-xs" style={{ color: MUTED }}>
                    Takes a few seconds. You can keep reading.
                  </span>
                </div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <span className="text-[13px]">
                    {towardRetrain} of {RETRAIN_THRESHOLD} corrections
                  </span>
                  <span className="flex gap-[3px]" aria-hidden>
                    {Array.from({ length: RETRAIN_THRESHOLD }, (_, i) => (
                      <span
                        key={i}
                        className="h-1.5 flex-1 rounded-[2px]"
                        style={{ background: i < towardRetrain ? CERULEAN : '#EBE8CC' }}
                      />
                    ))}
                  </span>
                  <span className="text-xs" style={{ color: MUTED }}>
                    {nextRetrainIn === 0 ? 'Retraining shortly.' : `Retrains automatically after ${nextRetrainIn} more.`}
                  </span>
                </div>
              )}

              {!retraining && lastAttempt?.last_attempt && (
                <RetrainReportCard
                  model={lastAttempt.model}
                  attempt={lastAttempt.last_attempt}
                  activeVersion={activeVersions[lastAttempt.model]}
                  resortedCount={resorted ? Object.keys(resorted.from).length : 0}
                  onShowResorted={() => {
                    go({})
                    setShowResortedOnly(true)
                  }}
                />
              )}

              {personalizationError && (
                <span className="text-xs leading-relaxed" style={{ color: CORAL_TEXT }} role="alert">
                  {personalizationError}
                </span>
              )}

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => void retrainPersonalization()}
                  disabled={retraining || totalCorrections === 0}
                  data-testid="retrain-now"
                  className={`h-8 rounded-[7px] border border-[#CFD6D2] bg-white px-3 text-[13px] font-medium transition-colors hover:bg-[#F6F7F2] disabled:cursor-not-allowed disabled:bg-[#F6F7F2] disabled:text-[#8A979A] ${FOCUS_RING}`}
                >
                  Retrain now
                </button>
                <button
                  type="button"
                  onClick={turnOffPersonalization}
                  className={`h-8 rounded-md px-1 text-[13px] hover:underline ${FOCUS_RING}`}
                  style={{ color: MUTED }}
                >
                  Turn off
                </button>
              </div>
            </section>
          )}

          {gmailStatus === 'connected' && personalization && !personalization.enabled && (
            <p className="px-1.5 text-xs leading-relaxed" style={{ color: MUTED }}>
              Personalization is off. Correct a label on any email to turn it on.
            </p>
          )}

          <div className="flex flex-col gap-2 px-1.5">
            {(gmailStatus === 'disconnected' || gmailStatus === 'error') && (
              <>
                {gmailStatus === 'error' && (
                  <span className="text-xs leading-relaxed" style={{ color: CORAL_TEXT }} role="alert">
                    Couldn&apos;t load Gmail{gmailError ? `: ${gmailError}` : '.'}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => connectGmail(rememberMe)}
                  className={`h-9 rounded-lg text-sm font-semibold text-white transition-colors hover:bg-[#006A8A] ${FOCUS_RING}`}
                  style={{ background: CERULEAN_TEXT }}
                >
                  {gmailStatus === 'error' ? 'Reconnect Gmail' : 'Connect Gmail'}
                </button>
                <label className="flex items-center gap-2 text-[13px]" style={{ color: MUTED }}>
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="size-3.5 accent-[#00789C]"
                  />
                  Remember me
                </label>
              </>
            )}
            {(gmailStatus === 'connecting' || gmailStatus === 'fetching') && (
              <span className="flex items-center gap-2 text-[13px]" style={{ color: MUTED }}>
                <span className="size-2 rounded-full motion-safe:animate-pulse" style={{ background: TEAL }} aria-hidden />
                {gmailStatus === 'connecting' ? 'Connecting to Gmail…' : 'Loading your inbox…'}
              </span>
            )}
            {gmailStatus === 'connected' && (
              <div className="flex items-center gap-2">
                <span className="size-2 rounded-full" style={{ background: TEAL }} aria-hidden />
                <span className="flex-1 text-[13px] font-medium">Gmail connected</span>
                <button
                  type="button"
                  onClick={() => void refreshInbox()}
                  aria-label="Refresh inbox"
                  title="Refresh inbox"
                  className={`flex size-8 items-center justify-center rounded-md transition-colors hover:bg-[#F4F2D0] ${FOCUS_RING}`}
                  style={{ color: MUTED }}
                >
                  <RefreshCw size={15} strokeWidth={1.75} aria-hidden />
                </button>
              </div>
            )}
            <span className="flex items-center gap-2 text-xs" style={{ color: status === 'error' ? CORAL_TEXT : MUTED }}>
              <Cpu size={14} strokeWidth={1.75} aria-hidden />
              {status === 'loading' && 'Loading models…'}
              {status === 'ready' && 'Sorting runs on this device'}
              {status === 'error' && 'Models failed to load'}
            </span>
          </div>
        </div>
      </aside>
    </>
  )
}
