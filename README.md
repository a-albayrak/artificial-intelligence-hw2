# CENG 461 – Artificial Intelligence  
**Homework - Markov Chain**  
**HW2_MarkovChain**

---

Using a corpus of **58110 English words**, our aim is to generate **synthetic word samples** using a **Markov chain**, which has a total number of **27 states**: all lowercase letters in the English alphabet (**26 letters**) and an **end-of-word symbol (*)**.  

The Markov Chain has an **order of 1**, which implies the following conditional independence:  
> **P(LN | LN-1, LN-2, …, LN-k) = P(LN | LN-1)**

Using **Python** and the packages `numpy`, `matplotlib.pyplot`, and `random` if needed, complete the following:

---

### 1. Estimate P(L₀) and P(Lₙ | Lₙ₋₁) and print it.

---

### 2. Calculate the **average length of a word** using the given list of words and print it.

---

### 3. Implement a function `calcPriorProb1(words, N)`  
- Takes the given list of words and `N` as input  
- Returns **P(Lₙ)**  
- **Plot** the distributions for **N = 1, 2, 3, 4, 5** using **bar plots**

---

### 4. Implement a function `calcPriorProb2(P_L0, P_LN_given_LN1, N)`  
- Takes `P(L₀)` and `P(Lₙ | Lₙ₋₁)` (estimated in Step 1) and `N` as input  
- Returns **P(Lₙ)**  
- **Plot** the distributions for **N = 1, 2, 3, 4, 5** using **bar plots**

---

### 5. Implement a function `calcWordProb(P_L0, P_LN_given_LN1, word)`  
- Takes `P(L₀)`, `P(Lₙ | Lₙ₋₁)` (from Step 1), and a `word`  
- Returns its **probability**, assuming:  
  > **P(L₀=w, L₁=o, L₂=r, L₃=d)**  

- Calculate and print the probabilities for the following words:  
  - `sad*`  
  - `exchange*`  
  - `antidisestablishmentarianism*`  
  - `qwerty*`  
  - `zzzz*`  
  - `ae*`

---

### 6. Implement a function `generateWords(P_L0, P_LN_given_LN1, M)`  
- Takes `P(L₀)`, `P(Lₙ | Lₙ₋₁)` (from Step 1), and `M`  
- Returns randomly sampled **M English words** using the given probabilities  
- **Print 10** of the generated words

---

### 7. Generate a synthetic dataset of size **100000**  
- Estimate the **average length of a word** and print it

---

### 8. **BONUS**  
Generate words after **increasing the order** of the Markov chain:  
- Use dependencies like `P(Lₙ | Lₙ₋₁, Lₙ₋₂)` or more  
- This creates **better word samples** but increases the **conditional probability table size exponentially**

> **Tips for higher-order Markov chains:**
- You must either:
  - Use `P(L₀, L₁, ..., Lₖ)` (restricts generation to k-length prefixes from dataset)
  - Or use:
    - `P(L₀)`,  
    - `P(L₁ | L₀)`,  
    - …  
    - `P(Lₙ | Lₙ₋₁, ..., Lₙ₋ₖ₋₁)`  
  - This allows word generation of **any length**

---
