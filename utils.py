from collections import OrderedDict
from models import *
import csv

attestations_filename = "Attestations_PNs_GNs.csv"
texts_filename = "Texts.csv"

# Utility functions
def normalize_year(year):
	period = 0
	if "XXX" in year or len(year) == 0 or year.startswith("Che"):
		year = -999
		period = -1
	else:
		if "XX" in year[:4]:
			if "AS" in year:
				period = 1
			elif "SS" in year:
				period = 2
			elif "IS" in year:
				period = 3
			else:
				period = -1
			year = -999
		elif "AS" in year:
			year = 100 + int(year[2:4])
			period = 1
		elif "SS" in year:
			year = 200 + int(year[2:4])
			period = 2
		elif "IS" in year:
			year = 300 + int(year[2:4])
			period = 3
		else:
			year = -999
			period = -1
	return year, period


def collect_tablet_years(tablets):
	#####   Collect Tablet Years  ####
	texts_fields = ["Id BDTNS", "Provenance", "Date", "Seal", "Id Line", "Part text", "Line", "Text"]

	texts_file = open('data/' + texts_filename, "rb")
	texts = csv.DictReader(texts_file, texts_fields)

	years = OrderedDict()
	invalid_tablets = []

	for row in texts:
		if "Date" in row[texts_fields[2]]: continue

		tablet_id = row[texts_fields[0]]
		year = row[texts_fields[2]]
		period = 0

		# Fix for inconsistent years
		if year.startswith("]"):
			year = row[texts_fields[2]][1:]

		year, period = normalize_year(year)

		if tablet_id not in tablets:
			if tablet_id not in invalid_tablets:
				invalid_tablets.append(tablet_id)
		else:
			tablets[tablet_id].year = year
			tablets[tablet_id].period = period

	# Print invalid tablet ids (tablets w/o pns/gns info)
	# for tablet in invalid_tablets:
	# 	print tablet


def collect_pns_gns():
	#####   Compile all pns for each tablet  ####
	PN_ID = 0
	GN_ID = 1
	att_fields = ["Id BDTNS", "Id Line", "Part text", "Line", 
		"Text", "PN/GN as attested", "Id PN/GN as attested", 
		"PN/GN Normalized", "Id PN/GN normalized", "PN or GN", 
		"Seal", "Date"]

	att_file = open('data/' + attestations_filename)
	attestations = csv.DictReader(att_file, att_fields)

	tablets = OrderedDict()

	# Build up tablet list with their respective pns and their frequencies
	for row in attestations:
		try:
			tablet_id = row[att_fields[0]]
			if "Id" in tablet_id: continue

			if tablet_id not in tablets:
				tablets[tablet_id] = Tablet(tablet_id)

			# Add pn/gn to the identified tablet
			if row[att_fields[9]] == "PN":
				pn = row[att_fields[6]]
				tablets[tablet_id].add_pn(pn)

			elif row[att_fields[9]] == "GN":
				gn = row[att_fields[6]]
				tablets[tablet_id].add_gn(gn)
		except:
			continue

	att_file.close()
	return tablets














