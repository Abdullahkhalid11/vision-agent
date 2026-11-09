# Image sources and licence status

| File      | Source        | Licence status                 | Action before publishing                 |
| --------- | ------------- | ------------------------------ | ---------------------------------------- |
| test1.jpg | Google Images | Unknown. Internal testing only | Replace with own photo or licensed image |
| test2.jpg | Google Images | Unknown. Internal testing only | Replace                                  |
| test3.jpg | Google Images | Unknown. Internal testing only | Replace                                  |
| test4.jpg | Google Images | Unknown. Internal testing only | Replace                                  |
| test5.jpg | Google Images | Unknown. Internal testing only | Replace                                  |
| test6.jpg | Google Images | Unknown. Internal testing only | Replace                                  |

Images stay out of git (data/ is ignored). Commit cases.jsonl, README.md, and this file only.

Note (Sun Nov 8, 2026): test4.jpg was a WebP file with a .jpg name, which the upload guard rejects (415). It was re-encoded as lossless PNG in RGB mode. Colour is unchanged; any transparency would have been flattened, and this image has none. Original kept in data/images_original (git-ignored).
