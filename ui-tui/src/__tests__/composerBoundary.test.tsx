import { PassThrough } from 'node:stream'

import { Box, renderSync, Text } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { describe, expect, it } from 'vitest'

import { ComposerRail } from '../components/composerRail.js'

const PAINTED_COLUMNS = 80
const RAIL_GLYPH = '\u2502'

/**
 * Rows the transcript window has to budget for a painted draft region. Ink
 * repaints and then clears on unmount, so only the first frame describes the
 * layout; the trailing frames would double-count every row. Every painted row
 * counts, including blank ones: the composer pane is flexShrink=0, so what
 * matters is how many terminal rows a draft costs, not how busy they look.
 */
const paintRows = (element: React.ReactElement, columns = PAINTED_COLUMNS): string[] => {
  const stdout = Object.assign(new PassThrough(), { columns, rows: 80 })
  const frames: string[] = []

  stdout.on('data', chunk => frames.push(chunk.toString()))

  const view = renderSync(element, {
    stdin: new PassThrough() as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  view.unmount()
  view.cleanup()

  return stripAnsi(frames[0] ?? '').split('\n')
}

/** Draft rows as appLayout renders them: pane padding + prompt cell + text. */
const draftLines = (lines: string[], promptWidth: number) => (
  <Box flexDirection="column">
    {lines.map((line, i) => (
      <Box key={i}>
        <Box width={promptWidth}>
          <Text>{' '.repeat(promptWidth)}</Text>
        </Box>
        <Text>{line || ' '}</Text>
      </Box>
    ))}
  </Box>
)

/** Draft block as it rendered before the rail existed: the row budget base. */
const bareDraft = (lines: string[], promptWidth: number) => (
  <Box flexDirection="column" paddingX={1}>
    {draftLines(lines, promptWidth)}
  </Box>
)

/** Draft block as appLayout renders it with the rail. */
const railedDraft = (lines: string[], promptWidth: number, cols: number) => (
  <Box flexDirection="column" paddingX={1}>
    <ComposerRail color="#888888" width={Math.max(1, cols - 1)}>
      {draftLines(lines, promptWidth)}
    </ComposerRail>
  </Box>
)

describe('composer boundary', () => {
  // The composer pane is flexShrink=0, so a row it gains is a row taken out of
  // the transcript window. The boundary has to be paid for without adding rows.
  it.each([1, 2, 5])('draws a rail on every draft row without changing the row count (%i rows)', count => {
    const lines = Array.from({ length: count }, (_, i) => `draft line ${i}`)
    const bare = paintRows(bareDraft(lines, 8))
    const railed = paintRows(railedDraft(lines, 8, PAINTED_COLUMNS))

    expect(bare).toHaveLength(count)
    expect(railed).toHaveLength(bare.length)
    expect(railed.every(line => line.startsWith(RAIL_GLYPH))).toBe(true)
    // The draft is still there, just after the rail.
    expect(railed.join('\n')).toContain('draft line 0')
  })

  // A draft line that exactly fills the wrap width used to rewrap when the rail
  // took a column of its own budget: the composer gained a row - painted with
  // the rail - and the transcript window lost one. Sweep every draft wrap width
  // and the lengths around each wrap boundary that triggered it.
  it.each([
    [40, 2],
    [40, 8],
    [40, 22],
    [80, 2],
    [80, 8],
    [80, 22],
    [120, 2],
    [120, 8],
    [120, 22]
  ])('keeps the painted row count across the draft wrap boundary (%i cols, prompt %i)', (cols, promptWidth) => {
    const wrapWidth = cols - 2 - promptWidth
    const problems: string[] = []

    for (const len of [
      wrapWidth - 1,
      wrapWidth,
      wrapWidth + 1,
      2 * wrapWidth - 1,
      2 * wrapWidth,
      2 * wrapWidth + 1
    ]) {
      const lines = ['x'.repeat(len)]
      const bare = paintRows(bareDraft(lines, promptWidth), cols)
      const railed = paintRows(railedDraft(lines, promptWidth, cols), cols)

      if (railed.length !== bare.length) {
        problems.push(`len=${len}: row count ${bare.length} -> ${railed.length}`)
      }

      if (!railed.every(line => line.startsWith(RAIL_GLYPH))) {
        problems.push(`len=${len}: rail not painted on every row`)
      }
    }

    expect(problems).toEqual([])
  })

  // The rail borrows the composer pane's own blank margin column, so the draft
  // keeps the exact columns it occupied before the rail existed.
  it('keeps the draft content in the columns it had before the rail', () => {
    const bare = paintRows(bareDraft(['hello'], 8))[0]
    const railed = paintRows(railedDraft(['hello'], 8, PAINTED_COLUMNS))[0]

    expect(railed).toBe(RAIL_GLYPH + bare.slice(1))
  })
})
