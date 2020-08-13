#!/usr/bin/env python3.7
""" kaplan_markov.py
This file contains some code relevant to Kaplan-Markov Risk-Limiting Batch Comparison Audits

For the formulas and a worked example, see
 A Kaplan-Markov auditing example using 2008 California data
 Mark Lindeman, 1/10/2010 (v. 1.2x, 3/1/2010)
 https://d56fe2f5-a-62cb3a1a-s-sites.googlegroups.com/site/electionaudits/small-batch/kaplan-example-12x.pdf

The example works with batch data from 2008 election in California’s 3rd Congressional District (CD3), in
the file ca-cd3-2018-batches.csv

The product of the final column (overall K-M P value) from the bottom of page six is 0.098616.

This is a subset of the full set of batches, from California's Statewide Database (SWDB).
To fully replicate the calculations, we'd need that whole dataset.

FIXME: calculations of taint are off when multiple discrepancies are there
"""

import csv
import logging
import types
import copy
from typing import Dict
from dataclasses import dataclass
from dataclass_csv import DataclassReader


@dataclass
class VoteCounts:
    name: str
    ballotcount: int
    tally: Dict[str, int]
    # num_winners: int


@dataclass
class ContestBatchRow:
    "Class to represent rows in ca-cd3-2018-batches.csv"

    precinct: str
    ballots_cast: int
    Lungren: int
    Durston: int
    Tuma: int
    Padilla: int
    times_drawn: int
    audit_Lungren: int
    audit_Durston: int
    audit_Tuma: int
    audit_Padilla: int
    e_Durston: float
    e_Tuma: float
    e_Padilla: float
    taint: float
    KM_factor: float
    net_KM_factor: float


def error_bound(votecounts, winners, margin):   # FIXME: needs array of margins per pair....
    """Return the error bound u_p for the given VoteCounts

    Example from top of table in page 3 of Lindeman:
    >>> error_bound(VoteCounts("060031A", 139, {"Lundgren": 48, "Durston": 83}), ["Lundgren"], 17453)
    0.005958860940812468

    >>> error_bound(VoteCounts("0606726420", 382, {"Lundgren": 158, "Durston": 172}), ["Lundgren"], 17453)
    0.021085200252105654

    >>> error_bound(VoteCounts("0606726420", 382, {"Lundgren": 158, "Durston": 172, "Tuma": 14, "Padilla": 15}), ["Lundgren"], 17453)
    0.021085200252105654 (fails because we need actual margins for each pair)

    >>> error_bound(VoteCounts("b1", 100, {"A": 60, "B": 20}), ["Elvis"], 200)
    Traceback (most recent call last):
    ValueError: Set of winners '['Elvis']' is not subset of candidates in tally: dict_keys(['A', 'B'])
    """

    candidates = set(votecounts.tally)

    if not set(winners).issubset(candidates):
        raise ValueError(
            f"Set of winners '{winners}' is not subset of candidates in tally: {votecounts.tally.keys()}"
        )

    losers = candidates - set(winners)

    bp = votecounts.ballotcount
    return max(
        (bp + votecounts.tally[winner] - votecounts.tally[loser]) / margin
        for winner in winners
        for loser in losers
    )


def relative_error(reported_votecounts, audit_votecounts, winners, margin):   # FIXME: needs array of margins per pair....
    """Return the relative error e_p for the given VoteCounts

    Example from page 4 of Lindeman:
    >>> relative_error(
    ...    VoteCounts("0606726420", 424, {"Lundgren": 158, "Durston": 172}),
    ...    VoteCounts("0606726420", 424, {"Lundgren": 158, "Durston": 182}),
    ...    ["Lundgren"],
    ...    17453)
    0.000572967398155045
    """

    # Note that we reuse the code in error_bound by inventing a new set
    # of tallies which are the discrepancies in the reported tallies, and
    # using a ballotcount of zero.

    discrepancies = {
        c: reported_votecounts.tally[c] - audit_votecounts.tally[c]
        for c in reported_votecounts.tally
    }

    discrepancy_vc = copy.deepcopy(reported_votecounts)
    discrepancy_vc.tally = discrepancies
    discrepancy_vc.ballotcount = 0

    return error_bound(discrepancy_vc, winners, margin)


def taintfactor(contest, discrepancy, u):
    """Return the taint factor for a given discrepancy and u in given contest
    TODO: add doctest.
    """

    taint = discrepancy / contest.min_margin / u
    return (1.0 - (1.0 / contest.U)) / (1.0 - taint)


def lindeman_test():
    # A quick-and-dirty Contest class instance, hardcoded from Lindeman example.

    c = types.SimpleNamespace()

    # First need to figure out the minumum margin of all pairs of winners and losers for this contest
    c.min_margin = 17453

    # To calculate U, we need all the batches, not just the selected batches.
    # Take this approximate value from the Lindeman paper
    # [May be 20.47011975018621 based on all 774 audit units]
    c.U = 20.47

    c.choices = "Lungren,Durston,Tuma,Padilla".split(",") #  TODO: after fixed to allow more than 2 candidates
    c.choices = "Lungren,Durston".split(",") # after fixed
    c.winner = "Lungren"

    losers = "Durston,Tuma,Padilla".split(",")

    reader = DataclassReader(
        open("ca-cd3-2018-batches.csv"), ContestBatchRow, delimiter=","
    )

    km_p_value = 1.0

    print(
        f"net_taint,precinct,net_discrepancy,reported_taint,calculated_taint,w_p,max_delta_in_batch"
    )

    for row in reader:
        logging.debug("%s", row)

        # reported_tally = {name: row.__dict__[name] for name in c.choices}
        reported_tally = {name: row.__dict__[name] for name in c.choices}
        reported_votecounts = VoteCounts(row.precinct, row.ballots_cast, reported_tally)
        audit_tally = {name: row.__dict__["audit_"+name] for name in c.choices}
        audit_votecounts = VoteCounts(row.precinct, row.ballots_cast, audit_tally)
        print(reported_tally, audit_tally)

        up = error_bound(reported_votecounts, [c.winner], c.min_margin)
        ep = relative_error(reported_votecounts, audit_votecounts, [c.winner], c.min_margin)
        print(f'up: {up}, ep: {ep}, taint: {ep/up}')

        w_p = getattr(row, "Lungren")
        max_delta_in_batch = min(w_p - getattr(row, loser) for loser in losers)
        u = (row.ballots_cast + row.Lungren - row.Durston) / c.min_margin
        d = row.Lungren - row.Durston - (row.audit_Lungren - row.audit_Durston)
        taint = taintfactor(c, d, u)
        net_taint = taint ** row.times_drawn

        km_p_value *= net_taint

        print(
            f"{net_taint}, {row.precinct}, {d}, {row.taint}, {taint}, {w_p}, {max_delta_in_batch}"
        )

        # print(f'{row.precinct},{row.Lungren},{row.KM_factor*2}')

    print(f"Overall km_p_value: {km_p_value}")


if __name__ == "__main__":
    lindeman_test()


# The code below is taken from /srv/s/electionaudits/master/electionaudits/models.py
# and provides some more code snippets which might be helpful in fleshing this code out.

if False:
    # from class Contest(models.Model):
    def km_select_units(self, factor=2.0, prng=None):
        """Return a list of selected contest_batches for the contest, based on error bounds and seed
        Return "factor" times as many as the current confidence level requires to show what may be needed if there are discrepancies.
        prng (Pseudo-Random Number Generator) is a function.  It defaults to Rivest's Sum of Square Roots, but
        can be specified as a function that returns numbers in the range [0, 1)
        """

        contest_batches = self.contestbatch_set.all()
        weights = [cb.u for cb in contest_batches]

        confidence = self.confidence
        if confidence == 90:  # FIXME: make this more general - use log ratio like this?
            confidence = 50

        alpha = (100 - confidence) / 100.0
        n = int(
            math.ceil(math.log(alpha) / math.log(1.0 - (1.0 / self.U))) * factor
        )  #  FIXME: deal with U = None

        if not prng:
            # The default pseudo-random number generator is to call ssr (Rivest's Sum of Square Roots algorithm)
            # with an incrementing first argument, and the current election seed: ssr(1, seed); ssr(2, seed), etc.
            prng = itertools.imap(
                erandom.ssr,
                itertools.count(1),
                itertools.repeat(self.election.random_seed),
            ).next

        # FIXME: avoid tricks to retain random values here and make this and weightedsample() into
        # some sort of generator that returns items that are nicely bundled with associated random values
        random_values = [prng() for i in range(n)]
        prng_replay = iter(random_values).next

        return zip(
            erandom.weightedsample(
                contest_batches, weights, n, replace=True, prng=prng_replay
            ),
            random_values,
        )

    # from class ContestBatch(models.Model)

    u = models.FloatField(
        blank=True, null=True, help_text="Maximum miscount / total apparent margin."
    )

    def error_bounds(self):
        """Calculate winners, losers, overall Margin between each pair of them in this contest,
        and error bound 'u' for each audit unit.
        """

        choices = self.choice_set.all()
        ranked = sorted(
            [choice for choice in choices if choice.name not in ["Under", "Over"]],
            key=lambda o: o.votes,
            reverse=True,
        )
        winners = ranked[: self.numWinners]
        losers = ranked[self.numWinners :]

        if len(winners) == 0 or winners[0].votes == 0:
            logging.warning("Contest %s has no votes" % self)
            return

        # margins between winners and losers

        margins = {}
        # FIXME: delete existing Margin database entries for this contest
        for winner in winners:
            margins[winner] = {}
            for loser in losers:
                margins[winner][loser] = max(
                    0, winner.votes - loser.votes - self.margin_offset
                )

                # FIXME: Look for, deal with ties....

                margin, created = Margin.objects.get_or_create(
                    votes=margins[winner][loser], choice1=winner, choice2=loser
                )
                margin.save()

        self.U = 0.0

        for au in self.contestbatch_set.all():
            au.u = 0.0
            vc = {}
            for voteCount in VoteCount.objects.filter(contest_batch__id__exact=au.id):
                vc[voteCount.choice] = voteCount.votes

            for winner in winners:
                for loser in losers:
                    if margins[winner][loser] <= 0:
                        logging.warning(
                            "Margin is %d for %s vs %s"
                            % (margins[winner][loser], winner, loser)
                        )
                        continue
                    au.u = max(
                        au.u,
                        float(au.contest_ballots() + vc[winner] - vc[loser])
                        / margins[winner][loser],
                    )

            au.save()
            self.U = self.U + au.u

        self.save()

        return {
            "U": self.U,
            "winners": winners,
            "losers": losers,
            "margins": margins,
        }
