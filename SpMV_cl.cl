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
		double sum = 0.0f;
		int idx;
		for (int j = 0; j < elemsinrow; j++) // 0 - 3
		{
			idx = j * rows + gid + low_limit; // 0-3 * 2 + 0-1
            sum += data[idx] * vin[col[idx]];
			// printf("sum, gid: %d --> sum: %lf, data: %lf, idx: %d\n", gid, sum, data[idx], idx);
		}
		vout[mJDS_row_permute[gid+count_row_permute]] = sum;
		printf("CL--> gid: %d : %lf\n",gid,  vout[mJDS_row_permute[gid]]);
	}
}
