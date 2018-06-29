
import pstats
stats = pstats.Stats('out.prof')
stats.strip_dirs().sort_stats('cumtime').print_stats(20)
