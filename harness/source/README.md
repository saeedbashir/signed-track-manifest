# Synthetic corpus source image

`corpus-person.jpg` is the photograph `make_corpus.py` crops the interpreter and the
decoy anchor out of. It is committed **on purpose**.

An earlier source image was not committed, on the reasoning that media does not belong
in the repository. It was then lost, and on 2026-09-21 the synthetic corpus could not
be regenerated to settle a question the harness existed to answer. Rebuilding it from
a substitute produced a corpus that scored **0.000 on every clip with zero detections**
— a dead instrument that reads exactly like a decisive failure. A regression guard you
cannot rebuild is not a guard.

It is a still, not media, it is 486 KB, and it is public domain.

| | |
| --- | --- |
| Source | NASA, KSC-03pd1855, Kennedy Space Center press conference |
| URL | https://images-assets.nasa.gov/image/KSC-03pd1855/KSC-03pd1855~large.jpg |
| Licence | Public domain — a work of the US federal government, 17 U.S.C. §105 |
| Why this one | A person standing at a podium at half-length. A tightly-cropped head-and-shoulders portrait does not detect once composited down to inset size |

Regenerate with:

    python harness/make_corpus.py --person harness/source/corpus-person.jpg --out harness/corpus
