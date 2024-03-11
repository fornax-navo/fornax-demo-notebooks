### If given more time:
I can work on the following to improve the code:

**Optimization**: The code can be optimized further for better performance, such as tuning Dask cluster settings based on system specifications and workload characteristics.

**Scalability**: Testing the code with a range of num_sample values to assess scalability and identify potential bottlenecks.

**Dask Diagnostic tools**: Using Dask diagnostic tools to identify performance bottlenecks and optimize the code.
<hr>

> Test your code to determine how it scales as num_sample increases. Write down the results, your interpretation of the results, and/or what you would try next to see if it improves your code.

![graph](./graph.png)

(Graph can be regenerated from the `dask_script`.)

From the graph, we can see that the dask implementation is significantly faster than the serial implementation. For the dask implementation as the num sample increase the time taken increases linearly. Whereas for the serial implementation the time taken increases exponentially. 

