import numpy as np
import random
#def mc_pi(n=100):
#    """Use Monte Calo Method to estimate pi.
#    """
#    m = 0
#    i = 0
#    while i < n:
#        x, y = np.random.rand(2)
#        if x**2 + y**2 < 1:
#            m += 1
#        i += 1
#    pi = 4. * m / n
#    res = {'total_point': n, 'point_in_circle': m, 'estimated_pi': pi}
#    
#    return res

### iterate number
total = int(100 * 10000)
#local_collection = xrange(1, total)
#### parallelize a data set into the cluster
#rdd = sc.parallelize(local_collection)       \
#        .setName("parallelized_data")        \
#        .cache()
#        
#def map_func(element):
#    x = random.random()       ## [0, 1)
#    y = random.random()       ## [0, 1)
#    
#    return (x, y)             ## random point
#def map_func_2(element):
#    x, y = element
#    return 1 if x**2 + y**2 < 1 else 0
#rdd2 = rdd.map(map_func)            \
#          .setName("random_point")  \
#          .cache()
#### calculate the number of points in and out the circle
#rdd3 = rdd2.map(map_func_2)                 \
#           .setName("points_in_out_circle") \
#           .cache()
#### how many points are in the circle
#in_circle = rdd3.reduce(operator.add)
#pi = 4. * in_circle / total
#print 'iterate {} times'.format(total)
#print 'estimated pi : {}'.format(pi)
#
#sc.parallelize(xrange(total))\
#    .map(lambda x: (random.random(), random.random()))\
#    .map(lambda x: 1 if x[0]**2 + x[1]**2 < 1 else 0)\
#    .reduce(lambda x, y: x + y)\
#    / float(total) * 4
           
pi2=sc.parallelize(xrange(total))                                  \
    .map(lambda x: 1 if sum(np.random.random(2) ** 2)<1 else 0)  \
    .reduce(lambda x, y: x + y)                                \
    / float(total) * 4
print 'pi2=',pi2
