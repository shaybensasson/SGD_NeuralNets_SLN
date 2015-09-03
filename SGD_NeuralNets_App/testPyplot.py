import numpy as np
import matplotlib.pyplot as plt

#create big-expensive-figure
#plt.ioff()      # turn updates off
#plt.title('now how much would you pay?')
#plt.xticklabels(fontsize=20, color='green')
#plt.draw()      # force a draw
#plt.savefig('alldone', dpi=300)
#plt.close()

#plt.ioff()      # turn updating back on
plt.plot(np.random.rand(20), '-g')
plt.draw()      # force a draw
plt.pause(0.1)
plt.plot(np.random.rand(20), '-r')
plt.draw()      # force a draw
plt.pause(0.1)
plt.plot(np.random.rand(20), '-y')
plt.draw()      # force a draw
plt.waitforbuttonpress(timeout=-1)