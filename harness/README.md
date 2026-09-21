# Validation harness

The pipeline is tuned against numbers, not impressions. Twenty clips, a
hand-drawn ground-truth rectangle for each, intersection-over-union scored, pass
at 0.85.

```sh
# draw the interpreter rectangle (and the face rectangle) on a frame 5s in
stm harness label clips/briefing-01.mp4 --at 5 --face

# score every clip in ground_truth.yaml
stm harness score harness/ground_truth.yaml --json harness/report.json
```

## Before there are real clips

Real interpreted content has to be found, licence-checked and hand-labelled.
Until it exists, generate a synthetic corpus whose ground truth is exact:

```sh
python harness/make_corpus.py --person any-portrait.jpg --out harness/corpus
stm harness score harness/ground_truth.yaml
```

`--person` is any image containing a person; the script detects and crops them.
The image is not committed here.

The corpus covers the six cases the pipeline must separate: corner inset over a
busy picture and over a calm one, side panels on each edge, an inset with no
window at all, and a decoy — a large still anchor beside a small signing inset,
where only hand motion tells them apart.

Two things learned building it, both of which make a corpus misleading if
ignored. A drawn figure produces **zero** detections, because it is not a person
to a detector trained on photographs. And a whole photograph is little better:
the pixels around the detected person are more photograph, so window growth has
internal structure to cross that real content does not have. The person has to
be cropped tightly and placed on a flat backdrop, which is what interpreted
content actually looks like.

Synthetic results are a smoke test, not evidence. They show the mechanism works
and catch regressions. Only real clips can tune the thresholds.

A clip passes when IoU is at least the threshold **and**, if a face rectangle
was drawn, the chosen crop fully contains it. A crop can score 0.9 and still cut
off the top of the head during expressive signing; the face check exists for
that. Watch the output as well as reading the table.

Clips are not committed. Keep them under `harness/clips/` locally.
