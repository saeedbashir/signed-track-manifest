# Signed Track Manifest — format specification, version 0.2

STM describes a video title whose sign language interpretation is delivered as
one or more **separate video tracks**, so that a receiver can control the
signer's appearance — position, size, opacity, visibility — and its
synchronisation with the programme. This is the receiver model the ITU's
audiovisual accessibility work described for closed signing; STM is a concrete,
minimal container for it.

Two documents make up a catalogue: a **per-title manifest** and a **catalogue
index**. Both are JSON. Both are validated by the schemas in `src/stm/schemas/`
and by `validate-stm`.

## Per-title manifest

```json
{
  "stmVersion": "0.2",
  "id": "wh-briefing-2026-03-14",
  "title": "Press Briefing — 14 March 2026",
  "durationMs": 486000,
  "source": {
    "publisher": "The White House",
    "url": "https://...",
    "licence": "public-domain-usgov",
    "attribution": "The White House. Public domain.",
    "productionCredit": "verified: produced in-house",
    "acquisitionRoute": "agency download page"
  },
  "video": {
    "main": { "url": "main.mp4", "width": 1920, "height": 1080, "codec": "h264" },
    "poster": "poster.jpg"
  },
  "signLanguage": [
    {
      "language": "ase",
      "languageLabel": "American Sign Language",
      "provenance": "human-interpreter",
      "extraction": {
        "method": "detected-crop",
        "sourceLayout": "corner-inset",
        "confidence": 0.91,
        "cropRect": { "x": 1392, "y": 612, "w": 432, "h": 396 },
        "fillMethod": "interpolated-surround"
      },
      "track": { "url": "signer-ase.mp4", "width": 432, "height": 396, "codec": "h264", "syncOffsetMs": 0 }
    }
  ],
  "captions": [
    { "kind": "verbatim", "language": "en", "url": "captions/verbatim.en.vtt", "format": "webvtt" },
    { "kind": "simplified", "language": "en", "url": "captions/simplified.en.vtt", "format": "webvtt",
      "generatedBy": "amazon-bedrock", "reviewed": false }
  ]
}
```

### Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `stmVersion` | `"0.1"` \| `"0.2"` | yes | Format version. See *Versioning*. |
| `id` | string `^[a-z0-9][a-z0-9-]{1,63}$` | yes | Stable identifier, unique within a catalogue. |
| `title` | string | yes | Display title. |
| `durationMs` | integer ≥ 0 | yes | Duration of the main programme. |
| `source` | object | yes | Where the material came from and on what terms. |
| `source.publisher` | string | yes | Who published it. |
| `source.url` | string | yes | Where it was published. |
| `source.licence` | string | yes | Licence identifier. Recommended: `public-domain-usgov`, `cc0-1.0`, `cc-by-4.0`, or an SPDX id. |
| `source.attribution` | string | yes | The attribution text a player should display. |
| `source.productionCredit` | string | no | Who actually produced it. Contractor-produced material can be copyrighted even when an agency publishes it. |
| `source.acquisitionRoute` | string | no | How the file was obtained. Never a platform rip. |
| `video.main` | video asset | yes | The main programme with the interpreter removed (or the panel cropped away). |
| `video.poster` | string | yes | A still with no signer visible. |
| `signLanguage[]` | array of sign tracks | yes (may be empty) | One entry per signed language. ASL and BSL are different languages. |
| `captions[]` | array of captions | yes (may be empty) | Written-language tracks. |

URLs are absolute or relative to the manifest's own location. Relative URLs are
what make a bundled, offline catalogue possible.

### Sign track

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `language` | ISO 639-3 (`^[a-z]{3}$`) | yes | `ase` American Sign Language, `bfi` British Sign Language, … Never spoken-language codes. |
| `languageLabel` | string | yes | Human-readable name. |
| `provenance` | `human-interpreter` \| `synthesised` | **yes, no default** | Whether a human produced this signing. |
| `extraction` | object | iff human | How the track was cut out of the source. |
| `generatedBy` | string | iff synthesised | What generated it. |
| `track` | video asset + `syncOffsetMs` | yes | The signer video. |
| `track.syncOffsetMs` | integer | yes | Positive delays the signer relative to the programme. Applied by the player. |
| `activity` | object | no (0.2) | How much the interpreter moves over time. See *Activity*. |

`provenance` is the most important field in the format. Interpretation lags
speech, viewers deserve to know whether they are watching a person, and a format
that let a synthesised track masquerade as a human one would do harm. So it is
required, closed, and has no default. This pipeline only ever writes
`human-interpreter`; the value `synthesised` exists so that other producers
label honestly.

### Activity (added in 0.2)

```json
"activity": { "intervalMs": 1000, "scale": "title-p95", "values": [0, 0, 12, 68, 91, 88, 40, 3, 0] }
```

| Field | Type | Meaning |
| --- | --- | --- |
| `intervalMs` | integer 200..5000 | Duration each value covers, from t=0. This pipeline writes 1000. |
| `scale` | `title-p95` | How the values were normalised. Closed; a second method is a new value. |
| `values` | integers 0..100 | One per interval. `values.length × intervalMs` is about `durationMs`. A short array is padded with its last value by the player and never extrapolated by the producer. |

The value is **hand and face motion inside the crop**, normalised to the title's
own 95th percentile: 100 is "as active as this interpreter gets", 0 is still. It
is not a measure of meaning, of interpretation quality, or of anything about
the language — it is how much the pixels moved, and the format says so because
a number published without that sentence will be read as one.

The producer publishes the measurement; the **player decides what "idle"
means**, exactly as it decides what a low `confidence` means. The reference
player fades the signer layer at a threshold of 20 after two seconds and
restores it after half a second above, opt-in and off by default. Whether
signing viewers want that is an open question, which is why the threshold is
the player's and not the format's.

### Extraction

| Field | Type | Meaning |
| --- | --- | --- |
| `method` | `detected-crop` \| `manual-crop` | Whether a detector or a human placed the rectangle. |
| `sourceLayout` | `corner-inset` \| `side-panel` | Composited over the picture, or in its own strip. |
| `confidence` | number 0..1 | Detection rate, positional stability and motion energy combined. Below **0.7** players should badge the title. Manual crops carry 1.0. |
| `cropRect` | `{x, y, w, h}` | The fixed rectangle in source-frame pixels. |
| `fillMethod` | `none` \| `interpolated-surround` \| `blurred-surround` | How the vacated rectangle was patched. Never generative. |

### Caption

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `kind` | `verbatim` \| `simplified` | yes | Simplified text is understood better than literal transcription by many deaf and hard-of-hearing viewers. |
| `language` | BCP 47 | yes | Written language. |
| `url` | string | yes | |
| `format` | `webvtt` | yes | |
| `generatedBy` | string | no | Present when a machine produced the text. **Its presence makes `reviewed` mandatory.** |
| `reviewed` | boolean | iff generated | Whether a human checked the text. Machine output is written `false`. |
| `reviewedBy` | string | iff `reviewed: true` | Who checked it. |

## Catalogue index

```json
{
  "stmVersion": "0.1",
  "titles": [
    { "id": "wh-briefing-2026-03-14", "title": "Press Briefing — 14 March 2026", "durationMs": 486000,
      "poster": "wh-briefing-2026-03-14/poster.jpg", "manifestUrl": "wh-briefing-2026-03-14/manifest.json",
      "signLanguages": ["ase"], "confidence": 0.91 }
  ]
}
```

A player fetches this once at launch and per-title manifests on play. Keep it
under 50 KB. `confidence` is the lowest across the title's sign tracks so a
player can badge without fetching the manifest.

## Validation

`validate-stm FILE...` validates per-title manifests; `--catalogue` validates
indexes. Schema violations are errors. Two semantic checks are errors too:
duplicate sign language codes and duplicate caption kind/language pairs. Low
confidence and very large sync offsets are warnings, promoted to errors with
`--warnings-as-errors`.

## Versioning

`stmVersion` is a string constant. Changes bump it; a player must refuse a
version it does not know. Additional properties are rejected everywhere, so
extensions require a version bump rather than silent vendor fields — which is
why adding one optional field made this 0.2 rather than a quiet edit to 0.1.

| Version | Change |
| --- | --- |
| 0.1 | The format as first published, 21 September 2026. |
| 0.2 | Adds the optional `signLanguage[].activity` object. Nothing else changed: every valid 0.1 manifest is a valid 0.2 manifest, and a 0.1 manifest carrying `activity` is rejected, because a 0.1 reader would reject it too. Readers should accept both. |

## Relationship to standards

- **ITU FG AVA closed signing** (2013): main programme and interpreter video
  delivered separately to the receiver, which handles the signer's appearance
  and synchronisation. STM is a container for that: appearance parameters are
  the player's, synchronisation is `syncOffsetMs`.
- **WCAG 2.2 SC 1.2.6 (AAA)**: prerecorded synchronised media requires sign
  language interpretation; captions do not satisfy it because sign and written
  language are different languages. WCAG guidance also advises against
  cropping the interpreter so only the hands are visible; the pipeline's
  padding rule and the harness's face check exist for that reason.
- The World Federation of the Deaf and the World Association of Sign Language
  Interpreters caution against signing avatars as a substitute for human
  signers. `provenance` is how this format takes that position seriously.
