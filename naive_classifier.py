from operator import itemgetter
from sets import Set
from utils import *

# Build tablet collection
tablets = collect_pns_gns()
collect_tablet_years(tablets)

# Modify this value to determine the window size
n_window = 6

# Build up two pn lists
#  - unique names in each year
#  - frequency of names in each year
years = {}
years_freq = {}
for tablet in tablets:
	tablet = tablets[tablet]
	if tablet.year not in years:
		years[tablet.year] = Set()
		years_freq[tablet.year] = {}
	
	# Unique set -- union of year with current tablet
	years[tablet.year] = years[tablet.year] | Set(tablet.pns)

	# Frequency set
	for pn in tablet.pns:
		if pn not in years_freq[tablet.year]:
			years_freq[tablet.year][pn] = 1
		else:
			years_freq[tablet.year][pn] += 1

# Unique Overlap
for tablet in tablets:
	tablet = tablets[tablet]
	guesses = []

	for year in years:
		pn_overlap = Set(tablet.pns) & years[year]

		# Possible match found? Add it to the 'highscore' list
		if len(pn_overlap) != 0:
			guesses.append((year, len(pn_overlap)))
			guesses = sorted(guesses, key=itemgetter(1), reverse=True)[:n_window]

	tablet.predictions.append(i[0] for i in guesses)

# Weighted Overlap
for tablet in tablets:
	tablet = tablets[tablet]
	guesses = []

	pn_sum = float(sum(tablet.pns.values()))
	for year in years_freq:
		weighted_sum = 0.0
		year_pns = years_freq[year]
		for pn in tablet.pns:
			if pn in year_pns:
				weighted_sum += float(year_pns[pn] - 1.0) * (float(tablet.pns[pn]) / float(pn_sum))

		# Possible match found? Add it to the 'highscore' list
		if weighted_sum != 0:
			guesses.append((year, weighted_sum))
			guesses = sorted(guesses, key=itemgetter(1), reverse=True)[:n_window]

	tablet.predictions.append(i[0] for i in guesses)

# Determine accuracy of guesses -- if year is within set of prediction, we have a hit.
correct = [0 for i in range(0, 2)]
for tablet in tablets:
	tablet = tablets[tablet]

	i = 0
	for prediction in tablet.predictions:
		if tablet.year in prediction:
			correct[i] += 1
		i += 1


print '~~Naive approaches~~'
print 'Unique Overlap Accuracy: %.2f%% (%i/%i)' % (float(correct[0]) / len(tablets) * 100.0, correct[0], len(tablets))
print 'Weighted Overlap Accuracy: %.2f%% (%i/%i)' % (float(correct[1]) / len(tablets) * 100.0, correct[1], len(tablets))
