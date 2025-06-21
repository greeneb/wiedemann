
**Context:**  

 We will implement a **minimal Block Wiedemann solver** in Python, for sparse GF(2) matrices. The project will serve both as a learning exercise and as a stepping stone toward future QEC decoder work (erasure decoding, minimal polynomial extraction).

---

# Goals

🔁 **Understand and implement Block Wiedemann:**

- Build Block Krylov sequences
    
- Implement Block Berlekamp–Massey (matrix linear recurrence fitting)
    
- Extract minimal polynomials
    
- Solve Ax=b over GF(2), compute rank and nullspace
    

🔁 **Reproduce SOA results referenced in CK and LinBox:**

- Benchmark Block Wiedemann vs basic Wiedemann
    
- Study scaling behavior on sparse GF(2) matrices


---

# Deliverable

A clean, modular **Block Wiedemann solver in Python**, with:

- Full unit tests on known sparse GF(2) matrices
    
- Benchmark plots of performance vs matrix size
    
- Minimal polynomial extraction and verification
    
- Simple report summarizing the results
    

---

# Timeline (~4 weeks)

|Week|Milestone|
|---|---|
|Week 1|Scalar Wiedemann + scalar Berlekamp–Massey implementation and validation|
|Week 2|Block Krylov sequence builder, Block S_t construction|
|Week 3|Block Berlekamp–Massey implementation, validate block version|
|Week 4|Benchmark full pipeline on sparse GF(2) matrices, write simple report and plots|

---

# References

- Wiedemann, "Solving sparse linear equations over finite fields," IEEE IT (1986)
    
- Coppersmith, "Solving homogeneous linear equations over GF(2) via block Wiedemann," Math. Comp. (1994)
    
- Kaltofen, "Analysis of Coppersmith's block Wiedemann algorithm," Math. Comp. (1995)
    
- LinBox [[https://github.com/linbox-team/linbox](https://github.com/linbox-team/linbox)]
    
- CK paper [[https://arxiv.org/abs/2106.09830](https://arxiv.org/abs/2106.09830)] (for context)
    
