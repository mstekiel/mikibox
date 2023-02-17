from timeit import default_timer as timer
import mikibox as ms

test_Bij= [10, 10, 3]

CEFpars = ms.crysfipy.CEFpars('C4', test_Bij, 'meV')
cefion_CePdAl3 = ms.crysfipy.CEFion(ms.crysfipy.Ion('Ce'),[0,0,0], CEFpars)

print(cefion_CePdAl3)