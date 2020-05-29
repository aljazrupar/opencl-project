// kernel

__kernel void mELLxVec(__global const int *col,
					   __global const double *data,
					   __global const int *mJDS_row_permute,
					   __global double *vin,
					   __global double *vout,
					   int low_limit,
					   int count_row_permute,
					   int rows, // number of rows in jag - 2
					   int elemsinrow) // len of 1 row - 4
{		
    int gid = get_global_id(0); 
	
	if(gid < rows) // 0 - 1
	{
		double sum = 0.0;
		int idx;
		for (int j = 0; j < elemsinrow; j++) // 0 - 3
		{
			idx = j * rows + gid + low_limit; // 0-3 * 2 + 0-1
            sum += data[idx] * vin[col[idx]];
			// printf("sum, gid: %d --> sum: %lf, data: %lf, idx: %d\n", gid, sum, data[idx], idx);
		}
		vout[mJDS_row_permute[gid+count_row_permute]] = sum;
		// printf("CL--> gid: %d : %lf\n",gid,  vout[mJDS_row_permute[gid]]);
	}
}

__kernel void dotProduct(__global const double *a,
						 __global const double *b,		
						 __global double *p,		
						 int size,		
						 __local double *partial)						
{
 
    // Get the index of the current element to be processed
	int lid = get_local_id(0);
    int gid = get_global_id(0); 
 
    // Do the operation
	double sum = 0.0;
	while( gid < size )
	{
		sum += a[gid] * b[gid];
		gid += get_global_size(0);
	}
	partial[lid] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	int floorPow2 = exp2(log2((float)get_local_size(0)));
    if (get_local_size(0) != floorPow2)										
	{
		if ( lid >= floorPow2 )
            partial[lid - floorPow2] += partial[lid];
		barrier(CLK_LOCAL_MEM_FENCE);
    }

	for(int i = (floorPow2>>1); i>0; i >>= 1) 
	{
		if(lid < i){ 
			partial[lid] += partial[lid + i];
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		p[get_group_id(0)] = partial[0];
	}
}
