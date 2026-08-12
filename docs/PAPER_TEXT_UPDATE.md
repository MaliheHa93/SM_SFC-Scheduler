# Required manuscript synchronization

Replace the transition-accounting sentences in Section V-E with the following
compact wording:

> Each request-specific control event constitutes one decision step. Initial
> placement, RETAIN, and route-only actions are applied immediately, while the
> corresponding transition is completed at the next decision event for that
> request or at a terminal outcome. Consequently, the next state represents
> actual mobility and resource evolution rather than a duplicate same-time
> state. A remapping action remains pending until make-before-break preparation
> completes, is canceled, or ends in continuity failure. The resulting
> transition is then stored in replay.

This correction does not change Eqs. (23)--(25). It clarifies the definition of
`S_(l+1)` and makes the learning description match the simulator.

Replace the first paragraph of Section VI-B with wording that does not imply
access to unavailable external simulator code:

> STG-DDQN is compared with common-model adaptations of Iterative Migration
> (IM) [3] and Dynamic Latency-Aware Partial VNF Migration (DLAPM) [5]. All
> schemes use identical mobility traces, requests, initial states, candidate
> placements, hard feasibility checks, routing, and make-before-break migration
> mechanics. Their initial placement is shared because IM and DLAPM primarily
> define migration behavior. IM uses a latency trigger and iteratively selects
> the smallest feasible partial relocation. DLAPM selects a latency-improving
> partial relocation only when its normalized improvement exceeds the stated
> threshold. The exact common-model rules and parameter values are published
> with the source code.
