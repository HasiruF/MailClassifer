import { avatarColorsFor, initialsFor } from './tokens'

export function Avatar({ name, size = 'sm' }: { name: string; size?: 'xs' | 'sm' | 'md' }) {
  const [background, color] = avatarColorsFor(name)
  const dim = size === 'md' ? 'size-10 text-sm' : size === 'sm' ? 'size-9 text-[13px]' : 'size-8 text-xs'
  return (
    <span
      className={`flex shrink-0 items-center justify-center rounded-full font-semibold ${dim}`}
      style={{ background, color }}
      aria-hidden
    >
      {initialsFor(name)}
    </span>
  )
}
