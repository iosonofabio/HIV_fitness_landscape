from itertools import izip
from collections import defaultdict

def parse_hinkley():
	header_file = open('data/Hinkley/header.txt', 'ru')
	fitness_file = open('data/Hinkley/ME-cv-NODRUG.csv', 'ru')

	fitness_landscape = defaultdict(dict)
	for mut, val in izip(header_file,fitness_file):
		prot = mut[:2]
		if prot=='PT': prot='PR'
		pos=int(mut.split(':')[0][2:])
		aa = mut.strip().split(':')[1]
		fitness = float(val.strip())
		fitness_landscape[(prot, pos) ][aa] = fitness

	header_file.close()
	fitness_file.close()
	return fitness_landscape
