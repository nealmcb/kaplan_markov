#!/usr/bin/env python3
""" kaplan_markov.py
This file contains some code relevant to kaplan-markov risk-limiting batch comparison audits
See A Kaplan-Markov auditing example using 2008 California data
Mark Lindeman, 1/10/2010 (v. 1.2x, 3/1/2010)
 https://d56fe2f5-a-62cb3a1a-s-sites.googlegroups.com/site/electionaudits/small-batch/kaplan-example-12x.pdf
for formulas and a worked example.
The code below taken from /srv/s/electionaudits/master/electionaudits/models.py
provides some code snippets which might be helpful in coding that up in Python
"""

import csv
import csv_to_objects
import logging
import types

"""
contest
    u
    U
    choices
    min_margin
"""

# First figure out the minumum margin of all pairs of winners and losers for this contest

c = types.SimpleNamespace()

# Hardcode from Lindeman example
c.min_margin = 17453
c.U = 20.47
c.choices = 'Lungren,Durston,Tuma,Padilla'.split(',')
c.winner = 'Lungren'

def taintfactor(contest, discrepancy, u):
    "Taint for a given discrepancy and u in given contest"

    taint = discrepancy / c.min_margin / u
    return (1.0 - (1.0 / c.U)) / (1.0 - taint)

print(f'taintfactor = {taintfactor(c, 231-230, 0.1)}')

# data.csv:
# Data from https://d56fe2f5-a-62cb3a1a-s-sites.googlegroups.com/site/electionaudits/small-batch/kaplan-example-12x.pdf
#  crude extraction in emacs: check for errors....
#  Subset of full set of batches, from california data
#  Product of final column = overall K-M P value: 0.098616 

reader = csv.DictReader(open('data.csv'), delimiter=',')

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

#...

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


#...

