'use client'

import { useInbox } from './inbox-context'
import { InboxList } from './inbox-list'
import { EmailDetail, NoEmailSelected } from './email-detail'
import { LINE, READING_BG } from './tokens'

// lg and up: list and open email side by side. Narrower: one at a time,
// the list until an email is opened, then the email (with a back button).
export default function InboxPage() {
  const { selectedId } = useInbox()
  return (
    <>
      <section
        aria-label="Message list"
        className={`${selectedId ? 'hidden lg:flex' : 'flex'} min-w-0 flex-1 flex-col lg:w-[400px] lg:flex-none lg:border-r xl:w-[440px]`}
        style={{ borderColor: LINE }}
      >
        <InboxList />
      </section>
      <section
        aria-label="Open email"
        className={`${selectedId ? 'flex' : 'hidden lg:flex'} min-w-0 flex-1 flex-col`}
        style={{ background: READING_BG }}
      >
        {selectedId ? <EmailDetail id={selectedId} /> : <NoEmailSelected />}
      </section>
    </>
  )
}
