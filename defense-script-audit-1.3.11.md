# Defense Script Audit for `ahmed_defense_1.3.11`

This audit checks whether `defense-script-1.3.11.md` matches the actual rendered slide context and whether it really fits a 20-25 minute defense.

## Timing Reality

Main script word count: about `2,769` spoken words.

Estimated duration:

- `1.8 words/sec` = `25.6 min`
- `2.0 words/sec` = `23.1 min`
- `2.2 words/sec` = `21.0 min`
- `2.5 words/sec` = `18.5 min`
- `3.0 words/sec` = `15.4 min`
- `4.0 words/sec` = `11.5 min`

Conclusion:

- If Ahmed speaks at `4 words/sec`, this will not be a 20-25 minute talk. It will be around 11-12 minutes.
- A 20-25 minute defense needs a calmer pace around `1.8-2.2 words/sec`, with pauses for pointing at figures and letting the committee read.
- The script is correctly sized for a defense only if delivered slowly and deliberately.

## Slide-Context Audit

### Slides 1-5

- Slide 1: script matches title and presenter context.
- Slide 2: script matches `S : f(x) -> A` and the four objective cards.
- Slide 3: script matches allowed/not allowed prediction boundary.
- Slide 4: script matches problem space, feature space, selector, algorithm space, and performance space.
- Slide 5: fixed in the second audit. Script now follows the visible architecture and the three right-side cards.

### Slides 6-12

- Slide 6: script matches five domains and `1.18M` labeled instances.
- Slide 7: fixed in the second audit. Script now speaks directly from LODO bars and gap/accuracy.
- Slide 8: fixed in the second audit. Script now says these are actual sampled traces, not abstract examples.
- Slide 9: script matches sixteen cheap `O(n)` structural features and timing-feature exclusion.
- Slide 10: script matches feature-importance figure and structural sorting signals.
- Slide 11: script matches empirical runtime labels and transform buttons.
- Slide 12: script matches corpus filtering, majority cap, weights, and split.

### Slides 13-18

- Slide 13: fixed in the second audit. Script now follows the visible input/model/output cards and avoids drifting into regression history.
- Slide 14: script matches raw signal, training controls, and balanced loss pressure.
- Slide 15: script matches gradient/Hessian notation and `multi:softprob + mlogloss`.
- Slide 16: script matches learning rate, depth, 500 trees, row sample, feature sample, hist method, and L1/L2 regularisation.
- Slide 17: fixed in the second audit. Script now focuses on v5 vs v6 source-aware check instead of overloading v7/v8.
- Slide 18: fixed in the second audit. Script now places v1-v8 journey here, where the bars are visible.

### Slides 19-26

- Slide 19: fixed in the second audit. Script now starts from SBS/VBS/model boxes and formulas.
- Slide 20: fixed in the second audit. Script now stays on accuracy vs gap closed and does not mention zero-regret early.
- Slide 21: script matches headline result: `76.1%`, `93.1%`, `89.6%`.
- Slide 22: script matches confusion matrix, timsort recall, and introsort-heapsort boundary.
- Slide 23: fixed in the second audit. Script now follows the F1 routing diagram and right-side cards.
- Slide 24: script matches the three contribution rows.
- Slide 25: script matches limitation/future-work two-column structure.
- Slide 26: script includes Q&A posture. Only the first two lines are spoken on the slide; the rest is rehearsal guidance.

## Remaining Risk

The script is now aligned with the deck, but the delivery must be paced correctly.

If Ahmed speaks fast, the problem is not missing content; the problem is delivery speed.

Recommended rehearsal rule:

- Aim for about `50-60 seconds` on most slides.
- Spend more time on slides `15`, `17`, `18`, `19`, `21`, and `22`.
- Do not read the script at phone speed. Speak, pause, point to the visible element, then explain.

## Practical Timing Plan

- Slides 1-5: about `5 min`
- Slides 6-12: about `7 min`
- Slides 13-18: about `6 min`
- Slides 19-23: about `5 min`
- Slides 24-26: about `2 min`

Total: about `25 min` at calm defense pace.

If committee pressure or nervousness makes the pace faster, add short pauses after every result slide and after every formula slide.

