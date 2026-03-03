import numpy as np

MIN_PRIORITY = 1e-6


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, max_priority):
        """
        ### Initialize
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.max_capacity = capacity
        c = 1
        while c < capacity:
            c *= 2
        self.capacity = c
        # $\alpha$
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = np.zeros(2 * self.capacity, dtype=np.float32)
        self.priority_min = np.ones(2 * self.capacity, dtype=np.float32) * np.inf

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = max_priority

        # Arrays for buffer
        self.data = np.zeros(self.capacity, dtype=object)

        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, state, priority=None):
        """
        ### Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data[idx] = state

        # Increment next available slot
        self.next_idx = (idx + 1) % self.max_capacity
        # Calculate the size
        self.size = min(self.max_capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = (
            self.max_priority**self.alpha if priority is None else priority**self.alpha
        )
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def _find_prefix_sum_idx_batch(self, prefix_sums):
        """
        #### Vectorized batch version of prefix-sum tree lookup.
        Finds largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha \le P$ for each P in prefix_sums.
        """
        idx = np.ones(len(prefix_sums), dtype=np.int64)
        while idx[0] < self.capacity:
            left = idx * 2
            left_sums = self.priority_sum[left]
            go_left = left_sums > prefix_sums
            prefix_sums = np.where(go_left, prefix_sums, prefix_sums - left_sums)
            idx = np.where(go_left, left, left + 1)
        return (idx - self.capacity).astype(np.int32)

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """
        assert self.size >= batch_size, (
            f"Not enough samples in buffer: {self.size} < {batch_size}"
        )

        # Cache total sum and min once — avoid repeated tree reads
        total = self._sum()
        min_priority = self._min()

        # Draw uniform random values in [0, total) and run vectorized tree lookup
        rnd = np.random.random(batch_size) * total * 0.999999
        indexes = self._find_prefix_sum_idx_batch(rnd)
        indexes = np.minimum(indexes, self.size - 1)

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = min_priority / total
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-beta)

        # Vectorized weight computation
        leaf_priorities = self.priority_sum[indexes + self.capacity]
        probs = leaf_priorities / total
        weights = (probs * self.size) ** (-beta)
        weights = (weights / max_weight).astype(np.float32)

        states = self.data[indexes]

        return states, weights, indexes

    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """
        # Clip to a minimum epsilon so no sample is permanently starved
        priorities = np.maximum(priorities, MIN_PRIORITY)

        # Batch max-priority update
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indexes, priorities):
            # Calculate $p_i^\alpha$
            priority_alpha = priority**self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        """
        ### Whether the buffer is full
        """
        return self.max_capacity == self.size
