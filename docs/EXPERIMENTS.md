# Experimental record

| Research question | Intervention and evidence | What worked | What failed or remains open |
|---|---|---|---|
| Can a direct weight edit survive new wording? | [Synthetic urgency and preservation](RESULTS.md) | Novel-phrase accuracy rose from 53.1% to 78.1%. The original matrix was restored exactly. | Sentiment fell from 100% to 98.4% with protection. |
| Does a solved patch improve benchmark decisions? | [BoolQ and document extraction](RESULTS.md) | BoolQ counterbalanced A/B rose from 57.0% to 71.6%. Unseen document types rose from 79.8% to 82.9%. | Direct Yes/No gained one answer in 512. A random extraction patch rivaled the solved patch on familiar types. |
| Can successful edits be composed? | [Deletion masks and LFM composition](RESULTS.md) | Mutual projection preserved fitted anchors. | Two disjoint mask pairs failed upon union. Independent LFM edits interfered off the anchors. |
| How far can a second edit move? | [Decision-interval derivation](METHOD.md) | 280/320 held-out choices proved stable over the entire tested strength interval under ideal suffix arithmetic. | Remaining cases require tighter limits or other checks. Rounded multi-token generation lies outside the guarantee. |
| Can a child execute independently? | [Embedding-table compiler](REPLICATION.md#standalone-embedding-children) | Token-table urgency beat overlap by six of 128 messages. | Sentiment scored 64/128. The additive model gives identical scores under token permutations. |
| Does inheriting a local contextual circuit solve reversal? | [Convolution channels, LP and role directions](ORDER_INHERITANCE.md) | LP perfectly fit 36 examples. A manually factored child got 64/64 service transfer cases. | The 16-channel LP child scored 23/64 on transfer. Negation broke the manually factored child. |
| Can the compiler recover the factors itself? | [Signed reversals, SVD and recovery proof](REVERSAL_RECOVERY.md) | Two axes recovered from 18 pairs, exact rank-one fit construction, 64/64 on the reused service suite. | Food transfer reached 48/72 and negation 12/64. The scoring family and intervention design are supplied. |
| Does a full-parent edit help on SST-2? | [SST-2 parent and child benchmark](SST2.md) | A fixed-A/B rank-one edit corrected 53 answers and broke 28 on 872 validation reviews. | With direct sentiment tokens, scalar threshold calibration deserves separate comparison. |

[Original pilots and frozen evidence](../results/README.md) · [Research requirements and replication](REPLICATION.md) · [Related work](RELATED_WORK.md)
