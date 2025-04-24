import numpy as np
import matplotlib.pyplot as plt
import random

def calcPriorProb1(words, states, N):
    count_dict = {s: 0 for s in states}
    total = 0
    
    for w in words:
        if len(w) >= N:
            ch = w[N-1]
        else:
            ch = '*'
        count_dict[ch] += 1
        total += 1

    P_LN = {s: count_dict[s]/total for s in states}
    return P_LN

def calcPriorProb2(P_L0, P_next_given_current, states, N):
    current_dist = P_L0.copy()

    for step in range(1, N):
        new_dist = {s: 0.0 for s in states}
        for x in states:
            px = current_dist[x]
            if px > 0:
                for s in states:
                    new_dist[s] += P_next_given_current[x][s] * px
        current_dist = new_dist

    return current_dist

def calcWordProb(P_L0, P_next_given_current, word):
    if len(word) == 0:
        return 0.0

    first_char = word[0]

    if first_char not in P_L0 or P_L0[first_char] == 0:
        return 0.0
    
    prob = P_L0[first_char]

    for i in range(1, len(word)):
        prev_char = word[i - 1]
        curr_char = word[i]
        if (prev_char not in P_next_given_current 
            or curr_char not in P_next_given_current[prev_char]):
            return 0.0
        prob *= P_next_given_current[prev_char][curr_char]
        if prob == 0.0:
            return 0.0

    return prob

def generateWords(P_L0, P_next_given_current, M, max_len=20):
    words = []
    letters = list(P_L0.keys())  

    transition_cdf = {}
    for c in letters:
        cdf_list = []
        running = 0.0
        for nx, p in P_next_given_current[c].items():
            running += p
            cdf_list.append((nx, running))
        transition_cdf[c] = cdf_list

    L0_cdf = []
    running = 0.0
    for c, p in P_L0.items():
        running += p
        L0_cdf.append((c, running))

    def sample_letter_from_cdf(cdf):
        r = random.random()
        for symbol, cp in cdf:
            if r <= cp:
                return symbol
        return cdf[-1][0]  

    for _ in range(M):
        first_char = sample_letter_from_cdf(L0_cdf)
        if first_char == '*':
            words.append("")
            continue

        word_chars = [first_char]
        current_char = first_char
        for _step in range(max_len - 1): 
            next_char = sample_letter_from_cdf(transition_cdf[current_char])
            if next_char == '*':
                break
            word_chars.append(next_char)
            current_char = next_char

        words.append("".join(word_chars))

    return words

def plot_distribution(distribution, title, states):
    filtered = {s: distribution[s] for s in states if distribution[s] > 0}
    labels = list(filtered.keys())
    probs = [filtered[s] for s in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, probs, color='skyblue')
    plt.title(title)
    plt.xlabel("State (letter or *)")
    plt.ylabel("Probability")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # -------------------------------------------------------------------------
    # 0) Read the file.
    # -------------------------------------------------------------------------
    with open("corncob_lowercase.txt", "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    states = list("abcdefghijklmnopqrstuvwxyz") + ["*"]

    first_letter_counts = {s: 0 for s in states}
    transitions = {
        s: {t: 0 for t in states} for s in states
    }

    # -------------------------------------------------------------------------
    # 1) Estimate P(L0) and P(LN | LN-1) and print it.
    # -------------------------------------------------------------------------
    for word in words:
        word = word.lower().strip()
        if not word:
            continue

        first_letter_counts[word[0]] += 1

        for i in range(len(word) - 1):
            current_letter = word[i]
            next_letter = word[i+1]
            transitions[current_letter][next_letter] += 1

        last_letter = word[-1]
        transitions[last_letter]["*"] += 1

    first_letter_total = sum(first_letter_counts.values())
    P_L0 = {s: first_letter_counts[s] / first_letter_total for s in states}

    P_next_given_current = {}
    for current_letter in states:
        row_sum = sum(transitions[current_letter].values())
        if row_sum == 0:
            P_next_given_current[current_letter] = {t: 0.0 for t in states}
        else:
            P_next_given_current[current_letter] = {
                t: transitions[current_letter][t] / row_sum
                for t in states
            }

    # -------------------------------------------------------------------------
    # 2) Calculate the average length of a word using the given list of words and print it.
    # -------------------------------------------------------------------------
    average_length = np.mean([len(w) for w in words])

    print("1. Estimate P(L0):")
    for letter in states:
        if letter != "*" and P_L0[letter] > 0:
            print(f"  {letter}: {P_L0[letter]:.4f}")

    print("\n1. Estimate P(LN | LN-1):")
    for current_letter in states:
        if sum(transitions[current_letter].values()) > 0:
            print(f"For '{current_letter}':")
            for next_letter, prob in P_next_given_current[current_letter].items():
                if prob > 0:
                    print(f"  -> {next_letter}: {prob:.4f}")
            print()

    print(f"2. Average length of a word: {average_length:.2f}\n")

    # -------------------------------------------------------------------------
    # 3) Implement calcPriorProb1. Plot the distributions for N=1,2,3,4,5 using bar plots.
    # -------------------------------------------------------------------------
    print(f"3. Implement calcPriorProb1. Plot the distributions for N=1,2,3,4,5 using bar plots:")
    for N in range(1, 6):
        P_LN_empirical = calcPriorProb1(words, states, N)
        plot_distribution(P_LN_empirical, f"calcPriorProb1 for N={N}", states)

    # -------------------------------------------------------------------------
    # 4) Implement calcPriorProb2. Plot the distributions for N=1,2,3,4,5 using bar plots.
    # -------------------------------------------------------------------------
    print(f"\n4. Implement calcPriorProb2. Plot the distributions for N=1,2,3,4,5 using bar plots:")
    for N in range(1, 6):
        P_LN_chain = calcPriorProb2(P_L0, P_next_given_current, states, N)
        plot_distribution(P_LN_chain, f"calcPriorProb2 for N={N}", states)

    # -------------------------------------------------------------------------
    # 5) Implement calcWordProb. Print the probabilities for the following words.
    # -------------------------------------------------------------------------
    print("\n5. Implement calcWordProb. Print the probabilities for the following words.:")
    sample_words = [
        "sad*", "exchange*", "antidisestablishmentarianism*",
        "qwerty*", "zzzz*", "ae*"
    ]
    for w_ in sample_words:
        prob = calcWordProb(P_L0, P_next_given_current, w_)
        print(f"  P({w_}) = {prob:.15f}")

    # -------------------------------------------------------------------------
    # 6) Implement generateWords. Randomly sample M words, print 10 of them.
    # -------------------------------------------------------------------------
    M = 20
    generated = generateWords(P_L0, P_next_given_current, M)
    print("\n6. Implement generateWords. Randomly sample M words, print 10 of them:")
    for gw in generated[:10]:
        print("  ", gw)

    # -------------------------------------------------------------------------
    # 7) Generate a synthetic dataset of size 100000, 
    #    estimate the average length of a word and print it.
    # -------------------------------------------------------------------------
    synthetic_dataset_size = 100000
    synthetic_words = generateWords(P_L0, P_next_given_current, synthetic_dataset_size)
    avg_length_synthetic = np.mean([len(w) for w in synthetic_words])
    print(f"\n7. Synthetic dataset of size {synthetic_dataset_size}: average length = {avg_length_synthetic:.2f}")

    # -------------------------------------------------------------------------
    # 8) Generate words after Increasing the order of the Markov chain.
    #    (2nd-order and 3rd-order)
    # -------------------------------------------------------------------------

    # ------------------- 8a) 2nd-order Markov chain -----------------------
    transitions2 = {}
    init_pair_counts = {}
    total_pairs = 0

    for word in words:
        if len(word) < 2:
            if len(word) == 1:
                pair = (word[0], '*')
                init_pair_counts[pair] = init_pair_counts.get(pair, 0) + 1
                total_pairs += 1
            continue

        init_pair = (word[0], word[1])
        init_pair_counts[init_pair] = init_pair_counts.get(init_pair, 0) + 1
        total_pairs += 1

        for i in range(len(word) - 2):
            pair = (word[i], word[i+1])
            nxt = word[i+2]
            if pair not in transitions2:
                transitions2[pair] = {}
            transitions2[pair][nxt] = transitions2[pair].get(nxt, 0) + 1

        last_pair = (word[-2], word[-1])
        if last_pair not in transitions2:
            transitions2[last_pair] = {}
        transitions2[last_pair]["*"] = transitions2[last_pair].get("*", 0) + 1

    P_init2 = {}
    for pair, count in init_pair_counts.items():
        P_init2[pair] = count / total_pairs

    for pair, nxt_dict in transitions2.items():
        total_count = sum(nxt_dict.values())
        for c3 in nxt_dict:
            nxt_dict[c3] = nxt_dict[c3] / total_count

    def generateWords2(P_init2, transitions2, M=10, max_len=20):
        pairs = list(P_init2.keys())
        
        init_cdf = []
        running = 0.0
        for pr, pval in P_init2.items():
            running += pval
            init_cdf.append((pr, running))

        transitions2_cdf = {}
        for pair_, nxt_dict in transitions2.items():
            cdf_list = []
            run2 = 0.0
            for symbol, pval in nxt_dict.items():
                run2 += pval
                cdf_list.append((symbol, run2))
            transitions2_cdf[pair_] = cdf_list

        def sample_from_cdf(cdf_list):
            r = random.random()
            for symbol_, cp in cdf_list:
                if r <= cp:
                    return symbol_
            return cdf_list[-1][0]

        words2 = []
        for _ in range(M):
            init_pair = sample_from_cdf(init_cdf)
            c0, c1 = init_pair

            if c0 == '*' and c1 == '*':
                words2.append("")
                continue
            elif c1 == '*':
                words2.append(c0)
                continue

            word_chars = [c0, c1]
            current_pair = (c0, c1)

            for _step in range(max_len - 2):
                if current_pair not in transitions2_cdf:
                    break
                nxt_char = sample_from_cdf(transitions2_cdf[current_pair])
                if nxt_char == '*':
                    break
                word_chars.append(nxt_char)
                current_pair = (current_pair[1], nxt_char)

            words2.append("".join(word_chars))

        return words2

    # Generate 10 words from the 2rd-order chain
    words2 = generateWords2(P_init2, transitions2, M=10)
    print("\n8a. Generate words from a 2nd-order Markov chain (10 samples):")
    for w_ in words2:
        print("  ", w_)

    # ------------------- 8b) 3rd-order Markov chain -----------------------
    transitions3 = {}
    init_triple_counts = {}
    total_triples = 0

    for word in words:
        word = word.strip()
        if len(word) < 3:
            if len(word) == 2:
                triple = (word[0], word[1], '*')
                init_triple_counts[triple] = init_triple_counts.get(triple, 0) + 1
                total_triples += 1
            elif len(word) == 1:
                triple = (word[0], '*', '*')
                init_triple_counts[triple] = init_triple_counts.get(triple, 0) + 1
                total_triples += 1
            continue

        init_triple = (word[0], word[1], word[2])
        init_triple_counts[init_triple] = init_triple_counts.get(init_triple, 0) + 1
        total_triples += 1

        for i in range(len(word) - 3):
            triple = (word[i], word[i+1], word[i+2])
            nxt = word[i+3]
            if triple not in transitions3:
                transitions3[triple] = {}
            transitions3[triple][nxt] = transitions3[triple].get(nxt, 0) + 1

        last_triple = (word[-3], word[-2], word[-1])
        if last_triple not in transitions3:
            transitions3[last_triple] = {}
        transitions3[last_triple]["*"] = transitions3[last_triple].get("*", 0) + 1

    P_init3 = {}
    for triple, count_ in init_triple_counts.items():
        P_init3[triple] = count_ / total_triples

    for triple, nxt_dict in transitions3.items():
        total_count = sum(nxt_dict.values())
        for c_ in nxt_dict:
            nxt_dict[c_] = nxt_dict[c_] / total_count

    def generateWords3(P_init3, transitions3, M=10, max_len=20):
        init_cdf = []
        running = 0.0
        for trip, pval in P_init3.items():
            running += pval
            init_cdf.append((trip, running))

        transitions3_cdf = {}
        for triple_, nxt_dict in transitions3.items():
            cdf_list = []
            run3 = 0.0
            for symbol, p_ in nxt_dict.items():
                run3 += p_
                cdf_list.append((symbol, run3))
            transitions3_cdf[triple_] = cdf_list

        def sample_from_cdf(cdf_list):
            r = random.random()
            for symbol_, cp in cdf_list:
                if r <= cp:
                    return symbol_
            return cdf_list[-1][0]

        words3 = []
        for _ in range(M):
            init_triple = sample_from_cdf(init_cdf)
            c0, c1, c2 = init_triple

            if c0 == '*' and c1 == '*' and c2 == '*':
                words3.append("")
                continue
            elif c1 == '*' and c2 == '*':
                words3.append(c0)
                continue
            elif c2 == '*':
                words3.append(c0 + c1)
                continue

            word_chars = [c0, c1, c2]
            current_triple = (c0, c1, c2)

            for _step in range(max_len - 3):
                if current_triple not in transitions3_cdf:
                    break
                nxt_char = sample_from_cdf(transitions3_cdf[current_triple])
                if nxt_char == '*':
                    break
                word_chars.append(nxt_char)
                current_triple = (current_triple[1], current_triple[2], nxt_char)

            words3.append("".join(word_chars))

        return words3

    # Generate 10 words from the 3rd-order chain
    words3 = generateWords3(P_init3, transitions3, M=10)
    print("\n8b. Generate words from a 3rd-order Markov chain (10 samples):")
    for w_ in words3:
        print("  ", w_)

if __name__ == "__main__":
    main()