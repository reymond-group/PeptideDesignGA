import sys
from PDGA_class import PDGA, set_seed

# var: pop_size, mut_rate, gen_gap, query, sim treshold, porpouse
ga = PDGA(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

# debug:
# ga.write_param()
# ga.set_verbose_true()

# riproducibility:
# set_seed(1) 

# building bloks:
# ga.exclude_buildingblocks(['Ac'])

# time limit:
# ga.set_time_limit('24:00:00')

ga.run()



