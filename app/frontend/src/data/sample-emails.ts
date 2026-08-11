import type { EmailInput } from '@/types'

// Stand-in for real Gmail ingestion (deferred to Phase 2 — see
// docs/superpowers/specs/2026-08-10-smart-email-app-design.md). Each of
// these gets run through the real classify() engine at runtime; nothing
// here is a pre-baked label.
export interface SampleEmail extends EmailInput {
  fromName: string
  receivedAt: string
  unread: boolean
}

export const SAMPLE_EMAILS: SampleEmail[] = [
  {
    id: 'e1',
    fromName: 'Dana Whitfield',
    fromAddr: 'dana.whitfield@vendor-legal.com',
    to: 'ceo@enron.com',
    subject: 'URGENT: Approval needed by EOD',
    body: 'Hi, please sign off on the attached contract before end of day. Legal is waiting.',
    receivedAt: '9:12 AM',
    unread: true,
  },
  {
    id: 'e2',
    fromName: 'Marcus Cole',
    fromAddr: 'marcus.cole@enron.com',
    to: 'you@enron.com',
    subject: 'Q3 pipeline numbers look off',
    body: 'Can you take a look at the Q3 gas pipeline throughput figures before the call tomorrow? Something in the Henry Hub column does not reconcile with last month.',
    receivedAt: '8:47 AM',
    unread: true,
  },
  {
    id: 'e3',
    fromName: 'Priya Natarajan',
    fromAddr: 'priya.n@enron.com',
    to: 'you@enron.com',
    subject: 're: lunch on friday?',
    body: 'Still on for the place near the office? I can do 12:30 if that works better for you this week.',
    receivedAt: 'Yesterday',
    unread: false,
  },
  {
    id: 'e4',
    fromName: 'IT Service Desk',
    fromAddr: 'it-notices@enron.com',
    to: 'you@enron.com',
    subject: 'Scheduled maintenance this weekend',
    body: 'The internal file server will be unavailable Saturday 10pm to Sunday 2am for routine maintenance. No action is required.',
    receivedAt: 'Yesterday',
    unread: false,
  },
  {
    id: 'e5',
    fromName: 'Deals Weekly',
    fromAddr: 'newsletter@dealsweekly.example.com',
    to: 'you@enron.com',
    subject: 'FREE MONEY!!! Click now to claim your prize',
    body: 'Congratulations!!! You have been selected for a limited-time cash reward. Click the link below immediately to claim before it expires. Act now, this offer will not last.',
    listUnsubscribe: true,
    receivedAt: '2 days ago',
    unread: true,
  },
  {
    id: 'e6',
    fromName: 'Owen Fitzgerald',
    fromAddr: 'owen.fitzgerald@enron.com',
    to: 'you@enron.com',
    subject: 'Draft slides for Thursday',
    body: 'Attached is a first pass at the slides for Thursday. Mostly looking for feedback on the risk section, the rest is fairly settled.',
    receivedAt: '2 days ago',
    unread: false,
  },
  {
    id: 'e7',
    fromName: 'Sarah Lindqvist',
    fromAddr: 'sarah.lindqvist@enron.com',
    to: 'you@enron.com',
    subject: 'Happy birthday!',
    body: 'Hope you have a great one this year. Cake in the break room at 3 if you want to swing by.',
    receivedAt: '3 days ago',
    unread: false,
  },
  {
    id: 'e8',
    fromName: 'Facilities',
    fromAddr: 'facilities@enron.com',
    to: 'all-staff@enron.com',
    subject: 'Parking garage closure notice',
    body: 'Level 3 of the parking garage will be closed for repaving starting Monday. Please use levels 1, 2, or 4 until further notice.',
    receivedAt: '4 days ago',
    unread: false,
  },
]
