import { avatarColorFor, initialsFor } from './tokens'

export function Avatar({ name, size = 'sm' }: { name: string; size?: 'sm' | 'md' }) {
  const color = avatarColorFor(name)
  const dim = size === 'md' ? 'size-10' : 'size-7'
  const text = size === 'md' ? 'text-xs' : 'text-[10px]'
  return (
    <span
      className={`mt-0.5 flex shrink-0 items-center justify-center rounded-sm font-mono font-semibold ${dim} ${text}`}
      style={{ background: `${color}33`, color, border: `1px solid ${color}66` }}
      aria-hidden
    >
      {initialsFor(name)}
    </span>
  )
}
