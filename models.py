
class Tablet(object):

	def __init__(self, tablet_id):
		self.tablet_id = tablet_id
		self.pns = {}
		self.gns = {}
		self.year = 0
		self.period = 0

	def add_pn(self, pn):
		if pn not in self.pns:
			self.pns[pn] = 0
		self.pns[pn] += 1

	def add_gn(self, gn):
		if gn not in self.gns:
			self.gns[gn] = 0
		self.gns[gn] += 1

	def toStr(self):
		return tablet_id