import { SAMPLE_EMAILS } from '@/data/sample-emails'
import { EmailDetail } from './email-detail'

export function generateStaticParams() {
  return SAMPLE_EMAILS.map((e) => ({ id: e.id }))
}

export default async function EmailDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  return <EmailDetail id={id} />
}
