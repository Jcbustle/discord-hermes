import { Box } from '@hermes/ink'
import type { ReactNode } from 'react'

/**
 * The composer pane is flexShrink=0, so every row it gains is a row taken
 * straight out of the transcript window. The rail must therefore never narrow
 * the draft's own wrap width: with one column less, a draft line that exactly
 * filled the old width no longer fits, and the composer grows a row for the
 * overflow (painted with the rail, so it is visible). Instead the rail hangs
 * one column to the left of the draft, borrowing the composer pane's own left
 * margin column, so the draft keeps its exact pre-rail geometry and the rail
 * costs zero rows.
 *
 * Ink paints a left-only border inside the node's own rect, so a negative left
 * margin is enough to move that rect one column into the margin: the border is
 * written at the node's own x and the draft content follows it, at the same
 * columns it occupied before the rail existed.
 */
export function ComposerRail({
  children,
  color,
  width
}: {
  children: ReactNode
  color: string
  /** Full width of the rail's rect, including the column the border occupies. */
  width: number
}) {
  return (
    <Box
      borderBottom={false}
      borderColor={color}
      borderLeft
      borderRight={false}
      borderStyle="single"
      borderTop={false}
      flexDirection="column"
      marginLeft={-1}
      width={Math.max(1, width)}
    >
      {children}
    </Box>
  )
}
