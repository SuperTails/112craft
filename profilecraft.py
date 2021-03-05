import cProfile
import craft
import pstats
from pstats import SortKey

cProfile.run('craft.main()', 'stats')
p = pstats.Stats('stats')
p.sort_stats(SortKey.CUMULATIVE).print_stats(50)