# kaplan_markov: Kaplan-Markov Risk-Limiting Batch Comparison Audits

kaplan_markov provides a module and two examples of the use of
the Kaplan-Markov approach to batch comparison risk-limiting tabulation audits.

For the underlying formulas and a worked example, see
 A Kaplan-Markov auditing example using 2008 California data
 Mark Lindeman, 1/10/2010 (v. 1.2x, 3/1/2010)
 https://d56fe2f5-a-62cb3a1a-s-sites.googlegroups.com/site/electionaudits/small-batch/kaplan-example-12x.pdf

See [Kaplan-Markov Example](kaplan_markov_example.ipynb) for a Jupyter notebook
demonstrating how it can be used, both with the example from Lindeman, and
with the 2010 Boulder County Coroner contest as documented at:

  http://bcn.boulder.co.us/~neal/elections/boulder-audit-10-11/

Requirements: Python 3.7+.

Use pytest to run tests on the code.

For full testing, requires dataclass_csv module.

The code is released under the MIT license.
