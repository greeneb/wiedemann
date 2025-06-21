**Goal:** Implement a working Block Wiedemann solver in Python within 1 month.  
**Mindset:** Implementation-first. Read theory _only as needed_ to unblock coding progress.

---

# Phase 1: Scalar Wiedemann + Scalar Berlekamp–Massey

**Main task:** Implement scalar Wiedemann and scalar Berlekamp–Massey to understand the pipeline.

**Read only:**

- Dumas Slides (Section 1–2)  
    [https://www.unilim.fr/pages_perso/jean-guillaume.dumas/Enseignement/M2/BlockWiedemann/BlockWiedemann_slides.pdf](https://www.unilim.fr/pages_perso/jean-guillaume.dumas/Enseignement/M2/BlockWiedemann/BlockWiedemann_slides.pdf)
    
- Wiedemann (1986) _skim only as background_  
    [https://ieeexplore.ieee.org/document/1057189](https://ieeexplore.ieee.org/document/1057189)
    

**Do NOT read full Coppersmith/Kaltofen yet.**

---

# Phase 2: Block Krylov Loop + Block S_t Construction

**Main task:** Implement block Krylov loop:  
Build sequence S_t = U^T A^t V where S_t are b x b matrices.

**Read only:**

- Dumas Slides (Section 3)
    

If stuck, refer to:

- Coppersmith (1994) _Algorithm description only_  
    [https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1203635-5/S0025-5718-1994-1203635-5.pdf](https://www.ams.org/journals/mcom/1994-62-205/S0025-5718-1994-1203635-5/S0025-5718-1994-1203635-5.pdf)
    

**Ignore the analysis section.**

---

# Phase 3: Block Berlekamp–Massey Implementation

**Main task:** Implement Block BM to fit matrix linear recurrence on S_t.

**Read:**

- Dumas Slides (Section 4–5)
    

If stuck:

- Coppersmith (1994) _Algorithm description_
    
- Kaltofen (1995) _Only refer if you want to understand stability / practical tips_  
    [https://www.ams.org/journals/mcom/1995-64-210/S0025-5718-1995-1273206-3/S0025-5718-1995-1273206-3.pdf](https://www.ams.org/journals/mcom/1995-64-210/S0025-5718-1995-1273206-3/S0025-5718-1995-1273206-3.pdf)
    

**DO NOT read proofs unless specifically needed.**

---

# Phase 4: Benchmark and Scaling Tests

**Main task:** Run tests, benchmark scaling, debug corner cases.

**Read:**

- Dumas Slides recap (entire deck)
    
- Kaltofen (1995) Section on practical block size choices (if needed)
    

**Optional:**

- Look at LinBox code for comparison only after your code works.  
    [https://github.com/linbox-team/linbox](https://github.com/linbox-team/linbox)
    

---

# Summary Priority Table

|Phase|Must Read|Optional Read|
|---|---|---|
|1 (Scalar Wiedemann)|Dumas slides 1–2|Wiedemann 1986 skim|
|2 (Block Krylov loop)|Dumas slides 3|Coppersmith 1994 algorithm only|
|3 (Block BM)|Dumas slides 4–5|Coppersmith 1994 + Kaltofen 1995 algorithm only|
|4 (Benchmark)|Dumas slides recap|Kaltofen 1995 block size tips|

---

# Final Guidance

- Focus on **coding first, reading second**.
    
- Use **Dumas slides as your primary reference** throughout.
    
- Only dive into Coppersmith/Kaltofen if your implementation gets stuck or if behavior is unclear.
    
- Remember: Block Wiedemann is fundamentally simple — matvec loop + matrix BM — **build it fast, then refine**.
    
