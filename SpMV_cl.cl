// kernel
// one work-item makes partial summation

__kernel void mCSRxVec(__global const int *rowptr,
					   __global const int *col,
					   __global const float *data,
					   __global float *vin,
					   __global float *vout,
					   int rows)
{		
    int gid = get_global_id(0); 

	if(gid < rows)
	{
		float sum = 0.0f;
        for (int j = rowptr[gid]; j < rowptr[gid + 1]; j++)
            sum += data[j] * vin[col[j]];
		vout[gid] = sum;
	}
}														

__kernel void mELLxVec(__global const int *rowptr,
					   __global const int *col,
					   __global const float *data,
					   __global float *vin,
					   __global float *vout,
					   int rows,
					   int elemsinrow)
{		
    int gid = get_global_id(0); 

	if(gid < rows)
	{
		float sum = 0.0f;
		int idx;
		for (int j = 0; j < elemsinrow; j++)
		{
			idx = j * rows + gid;
            sum += data[idx] * vin[col[idx]];
		}
		vout[gid] = sum;
	}
}
