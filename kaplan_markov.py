#!/usr/bin/env python3
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
"""

import csv
import csv_to_objects
import logging
import types

# A quick-and-dirty Contest class instance, hardcoded from Lindeman example.

c = types.SimpleNamespace()

# First need to figure out the minumum margin of all pairs of winners and losers for this contest
c.min_margin = 17453

# To calculate U, we need all the batches, not just the selected batches.
# Take this approximate value from the Lindeman paper
c.U = 20.47

c.choices = 'Lungren,Durston,Tuma,Padilla'.split(',')
c.winner = 'Lungren'


def taintfactor(contest, discrepancy, u):
    """Return the taint factor for a given discrepancy and u in given contest
    TODO: add doctest.
    """

    taint = discrepancy / c.min_margin / u
    return (1.0 - (1.0 / c.U)) / (1.0 - taint)


def main():
    reader = csv.DictReader(open('ca-cd3-2018-batches.csv'), delimiter=',')

    selected = csv_to_objects.read_rows(reader)

    losers = 'Durston,Tuma,Padilla'.split(',')

    km_p_value = 1.0

    for row in selected:
        for attr in 'ballots_cast,Lungren,Durston,Tuma,Padilla,times_drawn,audit_Lungren,audit_Durston,audit_Tuma,audit_Padilla'.split(','):
            setattr(row, attr, int(getattr(row, attr)))
        for attr in 'e_Durston,e_Tuma,e_Padilla,taint,KM_factor,net_KM_factor'.split(','):
            setattr(row, attr, float(getattr(row, attr)))

        logging.debug("%s", row)
        #print(row)

        w_p = getattr(row, 'Lungren')
        max_delta_in_batch = min(w_p - getattr(row, loser) for loser in losers)
        u = row.ballots_cast + row.Lungren - row.Durston / c.min_margin
        d = row.Lungren - row.Durston - (row.audit_Lungren - row.audit_Durston)
        taint = taintfactor(c, d, u)
        net_taint = taint ** row.times_drawn

        km_p_value *= net_taint

        print(f'{net_taint}, {row.precinct}, {d}, {row.taint}, {taint}, {w_p}, {max_delta_in_batch}')

        # print(f'{row.precinct},{row.Lungren},{row.KM_factor*2}')

    print(f'Overall km_p_value: {km_p_value}')


if __name__ == "__main__":
    main()


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
        if confidence == 90:	# FIXME: make this more general - use log ratio like this?
            confidence = 50

        alpha = ((100-confidence)/100.0)
        n = int(math.ceil(math.log(alpha) / math.log(1.0 - (1.0 / self.U))) * factor)	#  FIXME: deal with U = None

        if not prng:
            # The default pseudo-random number generator is to call ssr (Rivest's Sum of Square Roots algorithm)
            # with an incrementing first argument, and the current election seed: ssr(1, seed); ssr(2, seed), etc.
            prng = itertools.imap(erandom.ssr, itertools.count(1), itertools.repeat(self.election.random_seed)).next

        # FIXME: avoid tricks to retain random values here and make this and weightedsample() into
        # some sort of generator that returns items that are nicely bundled with associated random values
        random_values = [prng() for i in range(n)]
        prng_replay = iter(random_values).next

        return zip(erandom.weightedsample(contest_batches, weights, n, replace=True, prng=prng_replay), random_values)


# from class ContestBatch(models.Model)

    u = models.FloatField(blank=True, null=True,
                    help_text="Maximum miscount / total apparent margin." )


    def error_bounds(self):
        """Calculate winners, losers, overall Margin between each pair of them in this contest,
        and error bound 'u' for each audit unit.
        """

        choices = self.choice_set.all()
        ranked = sorted([choice for choice in choices if choice.name not in ["Under", "Over"]], key=lambda o: o.votes, reverse=True)
        winners = ranked[:self.numWinners]
        losers = ranked[self.numWinners:]

        if len(winners) == 0 or winners[0].votes == 0:
            logging.warning("Contest %s has no votes" % self)
            return

        # margins between winners and losers

        margins={}
        # FIXME: delete existing Margin database entries for this contest
        for winner in winners:
            margins[winner] = {}
            for loser in losers:
                margins[winner][loser] = max(0, winner.votes - loser.votes - self.margin_offset)

                # FIXME: Look for, deal with ties....

                margin, created = Margin.objects.get_or_create(votes = margins[winner][loser], choice1 = winner, choice2 = loser)
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
                         logging.warning("Margin is %d for %s vs %s" % (margins[winner][loser], winner, loser))
                         continue
                     au.u = max(au.u, float(au.contest_ballots() + vc[winner] - vc[loser]) / margins[winner][loser])

            au.save()
            self.U = self.U + au.u

        self.save()

        return {'U': self.U,
                'winners': winners,
                'losers': losers,
                'margins': margins,
                }
