import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

const PDF_WIDTH = 210 // A4 width in mm
const PDF_HEIGHT = 297 // A4 height in mm
const MARGIN = 10

export async function exportReportToPdf(report, modelName) {
  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  })

  let currentY = MARGIN

  // Title
  pdf.setFontSize(20)
  pdf.text('Sentinel - Backdoor Detection Report', MARGIN, currentY)
  currentY += 12

  // Model info
  pdf.setFontSize(10)
  pdf.setTextColor(100)
  pdf.text(`Model: ${modelName || 'Unknown'}`, MARGIN, currentY)
  currentY += 6
  pdf.text(`Generated: ${new Date().toLocaleString()}`, MARGIN, currentY)
  currentY += 10

  // Add Summary section
  const resultView = report?.demo_view?.tab_result || {}
  if (resultView.verdict) {
    pdf.setFontSize(14)
    pdf.setTextColor(0)
    pdf.text('Summary', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(11)
    pdf.setTextColor(0)
    const verdict = resultView.verdict
    const riskScore = resultView.risk_score
    const confidence = resultView.confidence

    const summaryLines = [
      `Verdict: ${verdict}`,
      `Risk Score: ${typeof riskScore === 'number' ? `${riskScore.toFixed(1)}%` : 'Unknown'}`,
      `Confidence: ${confidence ? `${(confidence <= 1 ? confidence * 100 : confidence).toFixed(1)}%` : 'Unknown'}`,
      `Trigger Class: ${resultView.trigger_class || 'Unknown'}`,
    ]

    summaryLines.forEach((line) => {
      if (currentY > PDF_HEIGHT - MARGIN - 10) {
        pdf.addPage()
        currentY = MARGIN
      }
      pdf.text(line, MARGIN, currentY)
      currentY += 6
    })

    if (resultView.confirmed_trigger) {
      currentY += 2
      pdf.setFontSize(10)
      pdf.setTextColor(80)
      pdf.text('Confirmed Trigger:', MARGIN, currentY)
      currentY += 5

      const triggerText = `"${resultView.confirmed_trigger}"`
      const splitTrigger = pdf.splitTextToSize(triggerText, PDF_WIDTH - 2 * MARGIN)
      pdf.setFontSize(10)
      pdf.setTextColor(40)
      pdf.text(splitTrigger, MARGIN, currentY)
      currentY += splitTrigger.length * 4 + 4
    }

    currentY += 4
  }

  // Detector breakdown
  const detectorVotes = resultView.detector_votes || {}
  if (Object.keys(detectorVotes).length > 0) {
    if (currentY > PDF_HEIGHT - MARGIN - 30) {
      pdf.addPage()
      currentY = MARGIN
    }

    pdf.setFontSize(14)
    pdf.setTextColor(0)
    pdf.text('Detector Votes', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(10)
    Object.entries(detectorVotes).forEach(([detector, verdict]) => {
      if (currentY > PDF_HEIGHT - MARGIN - 10) {
        pdf.addPage()
        currentY = MARGIN
      }
      pdf.setTextColor(0)
      pdf.text(`• ${detector.replace(/_/g, ' ')}: `, MARGIN, currentY)
      pdf.setTextColor(verdict === 'BACKDOORED' ? 200 : 50)
      pdf.text(verdict, MARGIN + 60, currentY)
      currentY += 6
    })
    currentY += 4
  }

  // Supporting samples
  if (Array.isArray(resultView.supporting_samples) && resultView.supporting_samples.length > 0) {
    if (currentY > PDF_HEIGHT - MARGIN - 20) {
      pdf.addPage()
      currentY = MARGIN
    }

    pdf.setFontSize(14)
    pdf.setTextColor(0)
    pdf.text('Supporting Samples', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(9)
    const samplesText = `Sample indices: ${resultView.supporting_samples.join(', ')}`
    const splitSamples = pdf.splitTextToSize(samplesText, PDF_WIDTH - 2 * MARGIN)
    pdf.text(splitSamples, MARGIN, currentY)
    currentY += splitSamples.length * 4 + 4
  }

  // How We Found It section
  const howView = report?.demo_view?.tab_how_we_found_it || {}
  const stage1 = Array.isArray(howView.stage_1_discovery) ? howView.stage_1_discovery : []

  if (stage1.length > 0) {
    pdf.addPage()
    currentY = MARGIN

    pdf.setFontSize(14)
    pdf.setTextColor(0)
    pdf.text('How We Found It', MARGIN, currentY)
    currentY += 10

    pdf.setFontSize(12)
    pdf.text('Stage 1: Discovery', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(10)
    stage1.forEach((d) => {
      if (currentY > PDF_HEIGHT - MARGIN - 20) {
        pdf.addPage()
        currentY = MARGIN
      }

      pdf.setTextColor(0)
      pdf.text(`${d.detector.replace(/_/g, ' ')} - ${d.verdict}`, MARGIN, currentY)
      currentY += 5

      pdf.setFontSize(9)
      pdf.setTextColor(80)
      const descLines = pdf.splitTextToSize(d.what_it_does, PDF_WIDTH - 2 * MARGIN - 5)
      pdf.text(descLines, MARGIN + 5, currentY)
      currentY += descLines.length * 4 + 3

      pdf.setFontSize(10)
      currentY += 2
    })
  }

  // Stage 2: Word-Level Evidence
  const stage2 = howView.stage_2_evidence || {}
  const topWords = Array.isArray(stage2.top_words) ? stage2.top_words.slice(0, 10) : []

  if (topWords.length > 0) {
    if (currentY > PDF_HEIGHT - MARGIN - 40) {
      pdf.addPage()
      currentY = MARGIN
    } else {
      currentY += 4
    }

    pdf.setFontSize(12)
    pdf.setTextColor(0)
    pdf.text('Stage 2: Word-Level Evidence', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(9)
    pdf.setTextColor(80)
    const statsText = `Word pool: ${stage2.word_pool_total || '—'} | Samples flagged: ${stage2.samples_in_word_pool || '—'} | Cross-detector overlap: ${stage2.intersection_total || '—'}`
    pdf.text(statsText, MARGIN, currentY)
    currentY += 6

    pdf.setFontSize(8)
    topWords.forEach((w) => {
      if (currentY > PDF_HEIGHT - MARGIN - 10) {
        pdf.addPage()
        currentY = MARGIN
      }

      pdf.setTextColor(0)
      pdf.text(`"${w.word}" (Class ${w.class}, Score: ${w.score?.toFixed(2) || '—'})`, MARGIN, currentY)
      currentY += 4
    })

    if (topWords.length < (stage2.top_words?.length || 0)) {
      currentY += 2
      pdf.setFontSize(8)
      pdf.setTextColor(120)
      pdf.text(`... and ${(stage2.top_words?.length || 0) - topWords.length} more words`, MARGIN, currentY)
      currentY += 4
    }
  }

  // Stage 3: Hypotheses
  const hypotheses = Array.isArray(howView.stage_3_hypotheses) ? howView.stage_3_hypotheses : []

  if (hypotheses.length > 0) {
    if (currentY > PDF_HEIGHT - MARGIN - 40) {
      pdf.addPage()
      currentY = MARGIN
    } else {
      currentY += 4
    }

    pdf.setFontSize(12)
    pdf.setTextColor(0)
    pdf.text('Stage 3: Candidate Triggers', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(9)
    hypotheses.slice(0, 5).forEach((hyp, i) => {
      if (currentY > PDF_HEIGHT - MARGIN - 15) {
        pdf.addPage()
        currentY = MARGIN
      }

      pdf.setTextColor(0)
      pdf.text(`${i + 1}. "${hyp.candidate_trigger}" (Class ${hyp.class})`, MARGIN, currentY)
      currentY += 5

      pdf.setFontSize(8)
      pdf.setTextColor(80)
      const reasoningLines = pdf.splitTextToSize(hyp.reasoning, PDF_WIDTH - 2 * MARGIN - 5)
      pdf.text(reasoningLines, MARGIN + 5, currentY)
      currentY += reasoningLines.length * 3.5 + 2

      pdf.setFontSize(9)
      currentY += 1
    })

    if (hypotheses.length > 5) {
      pdf.setFontSize(8)
      pdf.setTextColor(120)
      pdf.text(`... and ${hypotheses.length - 5} more candidates`, MARGIN, currentY)
    }
  }

  // Evidence Samples
  const samples = Array.isArray(howView.stage_3_evidence_samples)
    ? howView.stage_3_evidence_samples.slice(0, 5)
    : []

  if (samples.length > 0) {
    pdf.addPage()
    currentY = MARGIN

    pdf.setFontSize(12)
    pdf.setTextColor(0)
    pdf.text('Stage 3: Evidence Samples', MARGIN, currentY)
    currentY += 8

    pdf.setFontSize(8)
    samples.forEach((sample) => {
      if (currentY > PDF_HEIGHT - MARGIN - 15) {
        pdf.addPage()
        currentY = MARGIN
      }

      pdf.setTextColor(100)
      pdf.text(`Row #${sample.index}`, MARGIN, currentY)
      currentY += 4

      pdf.setTextColor(0)
      const textLines = pdf.splitTextToSize(
        sample.text,
        PDF_WIDTH - 2 * MARGIN - 5
      )
      pdf.text(textLines, MARGIN + 3, currentY)
      currentY += textLines.length * 3.5 + 4
    })

    if ((howView.stage_3_evidence_samples?.length || 0) > 5) {
      currentY += 2
      pdf.setFontSize(8)
      pdf.setTextColor(120)
      pdf.text(
        `... and ${(howView.stage_3_evidence_samples?.length || 0) - 5} more samples`,
        MARGIN,
        currentY
      )
    }
  }

  pdf.save(`sentinel-report-${Date.now()}.pdf`)
}
