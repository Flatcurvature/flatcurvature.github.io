---
title: "The Pigeonhole Principle"
published: 2020-02-13
image: "./cover.jpeg"
description: A counting idea that shows up in many parts of daily things if you have more items than containers, at least one container has to hold more than one item.
tags: [Mathematics, Combinatorics, Logic]
category: Math
draft: false
---

> Reading time: 10 minutes

> Cover image source: [Source](https://static.wikia.nocookie.net/toriko/images/d/d7/-A-Destiny_SGKK-Toriko-_07_%281280x720_H264_AAC%29_-5CB1DAD4-_20110602-14332199.jpg/revision/latest/scale-to-width-down/1000?cb=20110826175155)

The Pigeonhole Principle is a basic idea in mathematics:

> If you put more items into fewer containers, then at least one container will have more than one item.

It sounds obvious, but this principle shows up in areas like number theory, probability, and computer science.

---

### Birthday Example

Suppose you have 13 people and ask for their birth months.  
There are only 12 months, so the pigeonhole principle tells us that at least one month must have two or more people born in it.  
This doesn’t mean all months are used — only that a repeat is unavoidable.

---

### The Birthday Problem

A related and well-known case is the birthday paradox:

> In a group of just 23 people, there's over a 50% chance that at least two people share the same birthday.

This isn't exactly the pigeonhole principle, because there are 365 or 366 possible birthdays. But it shows how quickly repetition becomes likely.

To compute the probability that no two people share a birthday:

$$
P(\text{no match}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \cdots \times \frac{365 - n + 1}{365}
$$

Or in product form:

$$
P(n) = \prod_{k=0}^{n-1} \left(1 - \frac{k}{365} \right)
$$

The probability that at least two people share a birthday:

$$
P(\text{at least one match}) = 1 - P(n)
$$

For \( n = 23 \):

$$
P(\text{at least one match}) \approx 0.507
$$

More details: [Wikipedia: Birthday problem](https://en.wikipedia.org/wiki/Birthday_problem)

---

### Number Pair Example

From the [Art of Problem Solving wiki](https://artofproblemsolving.com/wiki/index.php/Pigeonhole_Principle):

Choose any 5 numbers from the set \( \{1, 2, \dots, 8\} \). Then at least two of them must add up to 9.

We group the numbers into the following 4 pairs:

- (1, 8)  
- (2, 7)  
- (3, 6)  
- (4, 5)

Each pair is a “pigeonhole” If you select 5 numbers (pigeons), the pigeonhole principle guarantees that one of the pairs is fully selected — so the two numbers in it will sum to 9.

### Cybersecurity Example: Hash Collisions

In cybersecurity, the pigeonhole principle explains why hash collisions are inevitable.

A hash function maps input data (of arbitrary size) to fixed-size outputs. For example, SHA-256 produces a 256-bit output. That gives us:

$$
2^{256} \text{ possible outputs}
$$

That number is huge but still finite.

Now suppose you're hashing files. The number of possible files is infinite (or at least far larger than \( 2^{256} \)). So if you keep hashing enough files, eventually two different files will map to the same hash — this is called a collision.

This principle is the reason why collision resistance is in a sense should be considered as design requirement for cryptographic hash functions. If an attacker can deliberately create two inputs with the same hash, it breaks the integrity of systems like digital signatures or file verification.

### Final Note

The pigeonhole principle is simple:

$$
\text{If } k > n, \text{ then placing } k \text{ items into } n \text{ containers implies at least one container has } \geq 2 \text{ items.}
$$

References:
- [AoPS: Pigeonhole Principle](https://artofproblemsolving.com/wiki/index.php/Pigeonhole_Principle)
- [Wikipedia: Birthday problem](https://en.wikipedia.org/wiki/Birthday_problem)

---