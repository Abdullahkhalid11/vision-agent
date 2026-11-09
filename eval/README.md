# Eval rules (fix these before the first agent run)

1. Expected answers are written by a human from the full-size image. Never copy them from agent output.
2. Counts and existence use detections at confidence 0.5 or higher. Every question states this. The harness passes the same threshold to the tool, so the agent and the ground truth count the same objects.
3. Fill every `expected: null` row before running. Rows with a `suggested` value still need checking on the full image.
4. Counting rule for partly visible objects at image edges: count an object only if at least half of it is inside the frame. Apply this to every image.
5. Cross-image answers are judged by image content, not by upload date. The harness uploads all six images before running the cross-image rows.
6. Scoring is fixed here. Do not change a row, a rule, or a threshold after seeing results. If something must change, record the change and the reason in this file, and keep the original results.
7. Report every failure with its tool trace. Do not hide failed rows.
8. Infrared image test2 is a known domain shift. Its results are reported separately.

## Amendment (Sun Nov 8, 2026, before any expected value was filled)

Rule 2 is clarified. Expected counts and yes/no answers are a human count of the requested objects that are clearly visible in the full-size image, applying rule 4 for partly visible objects. The 0.5 confidence threshold applies to the agent's tool calls and the detector's output. It does not apply to the human ground truth. Question text still states the threshold, so the agent and the harness count the same way.


## Amendment 2 (Sun Nov 8, 2026, after values were filled, before any agent run)

Cross-image expected sets are derived from the per-image rows, using the same rule as the count and yes/no rows. The first draft of x01 was blank and x02 listed only test2.jpg. Both were corrected to match c01 to c08, e01 to e04, and b02. Original draft values are kept in git history at commit da2a884.
