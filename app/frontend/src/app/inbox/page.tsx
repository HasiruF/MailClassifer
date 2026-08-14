'use client'

import { useInbox } from './inbox-context'
import { InboxList } from './inbox-list'
import { EmailDetail } from './email-detail'

export default function InboxPage() {
  const { selectedId } = useInbox()
  return selectedId ? <EmailDetail id={selectedId} /> : <InboxList />
}
