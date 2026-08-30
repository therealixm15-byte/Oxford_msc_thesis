import numpy as np

def dhondt_allocation(total_budget, parties, votes, max_seats_per_party):
    """
    Allocate an integer budget using the D'Hondt divisor method.

    Args:
        total_budget:
            Total number of seats/resources to allocate.
        parties:
            List of party identifiers.
        votes:
            List of non-negative real vote totals corresponding to parties.
        max_seats_per_party:
            Maximum number of seats any party may receive.
            If None, no upper limit is enforced.

    Returns:
        allocation:
            Dictionary mapping each party to its allocated budget.
        allocation_array:
            NumPy array containing the allocation in the same order as
            the input parties.
    """
    votes = np.asarray(votes, dtype=float)
    num_parties = len(parties)
    seats = np.zeros(num_parties, dtype=int)
    for _ in range(total_budget):
        quotients = np.where(votes > 0, votes / (seats + 1), -np.inf)
        # Parties that have reached the maximum cannot receive more seats.
        if max_seats_per_party is not None:
            quotients[seats >= max_seats_per_party] = -np.inf
        # No eligible parties remain.
        if np.all(np.isneginf(quotients)):
            break
        winner = np.argmax(quotients)
        seats[winner] += 1
    allocation = {
        party: int(seat_count)
        for party, seat_count in zip(parties, seats)
    }
    return allocation, seats