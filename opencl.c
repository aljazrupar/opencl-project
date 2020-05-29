#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include "mtx_sparse.h"
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 1024
#define REPEAT 1000

#define MAX_S 10
#define MIN_S -10
#define PRECISION 1e-10

#define JAGPADD 1
// finite number of iterations, which is not larger than the size of the matrix
// gcc SpMV_cl.c mtx_sparse.c -fopenmp -O2 -I/usr/include/cuda -L/usr/lib64 -l:"libOpenCL.so.1" -o ou

int generate_vector_s(double *vector, int vec_len){
    
    for(int i = 0; i < vec_len; i++){
        double new_num = rand() % (MAX_S + 1 - MIN_S) + MIN_S;
        vector[i] = new_num;
    }
    return 0;
}
int matrix_vector_product(struct mtx_ELL mELL, double *vector_s, double *vector_b){
    int el_in_row = mELL.num_elementsinrow;
    for(int i = 0; i < mELL.num_rows; i++){ // Gre se enkrat cez za "brezveze". Mogoce bols z calloc nardit vsakic
        vector_b[i] = 0;
    }
    for(int i = 0; i < mELL.num_rows; i++){
        for(int j = 0; j < el_in_row; j++){
            int col_idx = mELL.col[j*mELL.num_rows + i];
            // printf("data-> %lf   s-> %lf\n",mELL.data[i*el_in_row + j], vector_s[col_idx]);
            vector_b[i] += mELL.data[j*mELL.num_rows + i] * vector_s[col_idx];
        }
    }
    return 0;
}

int vector_vector_minus(double *vector1, double *vector2, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = vector1[i] - vector2[i];
    }
    return 0;
}

double vectorT_vector_product(double *vector_1, double *vector_2, int len){
    double product = 0;
    for(int i = 0; i < len; i++){
        product += vector_1[i] *vector_2[i];
    }
    return product;
}

int copy_vector(double *vector1, double *vector2, int len){
    for(int i = 0; i < len; i++){
        vector2[i] = vector1[i];
    }
    return 0;
}

int scalar_vector_product(double coef, double *vector, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = coef * vector[i];
    }
    return 0;
}

int vector_vector_plus(double *vector1, double *vector2, int len, double *vec_out){
    for(int i = 0; i < len; i++){
        vec_out[i] = vector1[i] + vector2[i];
    }
    return 0;
}

int print_vector(double *vector, int len){
    for(int i = 0; i < len; i++){
        printf("%ld\n", vector[i]);
    }
    return 0;
}

int main(int argc, char *argv[]) // argv -> 0: matrix, 1: kernel, 3: precision
{
    FILE *f;
    struct mtx_COO mCOO;
    struct mtx_CSR mCSR;
    struct mtx_ELL mELL;
    struct mtx_JDS mJDS;
	cl_int clStatus;
    int repeat;
    printf("Start\n");
    if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else    
    { 
        
        if ((f = fopen(argv[1], "r")) == NULL){ 
            printf("Unable to open file\n");
            exit(1);
        }
    }
    // create sparse matrices
    if (mtx_COO_create_from_file(&mCOO, f) != 0){
        printf("Error reading file.");
        exit(1);
    }
    
    mtx_CSR_create_from_mtx_COO(&mCSR, &mCOO);
    mtx_ELL_create_from_mtx_CSR(&mELL, &mCSR);
    mtx_JDS_create_from_mtx_CSR(&mJDS, &mCSR, JAGPADD);
    /* for(int i = 0; i < mELL.num_elements; i++){
        printf("%lf - ", mELL.data[i]);
    }
    printf("\n");
    for(int i = 0; i < mELL.num_elements; i++){
        printf("%d - ", mELL.col[i]);
    } */



    

    // START OPENCL:
    printf("Start OPENCL\n");

    FILE *fp;
    char fileName[100];
    char *source_str;
    size_t source_size;
    sprintf(fileName,"%s.cl", argv[2]);
    fp = fopen(fileName, "r");
    if (!fp) 
	{
		fprintf(stderr, ":-(#\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
    fclose( fp );
    
    // Get platforms
    cl_uint num_platforms;
    clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    
    //Get platform devices
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    num_devices = 1; // limit to one device
    cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    // Context
    cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &clStatus);
 
    // Command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], 0, &clStatus);

    // Create and build a program
    cl_program program = clCreateProgramWithSource(context,	1, (const char **)&source_str, NULL, &clStatus);
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char *build_log;
	clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
    if (build_log_len > 2)
    {
        build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
        clStatus = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 
                                        build_log_len, build_log, NULL);
        printf("%s", build_log);
        free(build_log);
        return 1;
    }
    

    // number of iterations must be size of matrix
    int iter = mJDS.num_cols * mJDS.num_rows;
    double *vector_s = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_b = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_temp= (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_r = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_Ap = (double *)calloc(mELL.num_cols, sizeof(double)); // A*p(k)
    double *vector_beta_num = (double *)calloc(mELL.num_cols, sizeof(double));

    double *vector_alpha_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_beta_p = (double *)calloc(mELL.num_cols, sizeof(double));
    double *vector_alpha_A_p = (double *)calloc(mELL.num_cols, sizeof(double));

    double *vecOutJDS = (double *)calloc(mJDS.num_rows, sizeof(double));
    double *vector_zeros = (double *)calloc(mELL.num_rows, sizeof(double));

    generate_vector_s(vector_s, mELL.num_cols);
    matrix_vector_product(mELL, vector_s, vector_b); //A*s, generate vector b.
    
    // 2. Inicialize starting vector of ones X0
    double *vector_x = (double *)calloc(mELL.num_cols, sizeof(double));
    for(int i = 0; i < mELL.num_cols; i++){
        vector_x[i] = 1;
    }

    double precision_curr = 0;
    double coef_alpha = 0;
    double coef_alpha_denom = 0;
    double coef_beta = 0;
    double coef_beta_num = 0;

    
    matrix_vector_product(mELL, vector_x, vector_temp); // A*x
    vector_vector_minus(vector_b, vector_temp, mELL.num_cols, vector_r); // r = b- A*x
    
    copy_vector(vector_r, vector_p, mELL.num_cols); // r -> p
    
    // allocate memory on device and transfer data from host JDS - Each jag seperate
    // Matrix-vector product

    cl_mem mJDS_col = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                      mJDS.data_arr_length * sizeof(cl_int), NULL, &clStatus);
    cl_mem mJDS_data = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                       mJDS.data_arr_length * sizeof(cl_double), NULL, &clStatus);

    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_col, CL_TRUE, 0,						
                                    mJDS.data_arr_length * sizeof(cl_int), mJDS.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_data, CL_TRUE, 0,						
                                    mJDS.data_arr_length * sizeof(cl_double), mJDS.data, 0, NULL, NULL);				
    cl_mem mJDS_row_permute = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mJDS.jds_rows * sizeof(cl_int), NULL, &clStatus); 
    clStatus = clEnqueueWriteBuffer(command_queue, mJDS_row_permute, CL_TRUE, 0,						
                                    mJDS.jds_rows * sizeof(cl_int), mJDS.row_permute, 0, NULL, NULL);

    // vectors
    cl_mem vecIn = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mJDS.num_cols * sizeof(cl_double), vector_x, &clStatus); // set vecIn the one to be A * vector_x
    clStatus = clEnqueueWriteBuffer(command_queue, vecIn, CL_TRUE, 0,						
                                        mJDS.num_cols*sizeof(cl_double), vector_x, 0, NULL, NULL);	

    cl_mem vecOut = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                     mJDS.num_cols * sizeof(cl_double), NULL, &clStatus);
    /* printf("cl stat1-> %d\n",clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, vecOut, CL_TRUE, 0,						
                                        mJDS.num_cols*sizeof(cl_double), vector_zeros, 0, NULL, NULL); */
                                        
    printf("cl stat2-> %d\n", clStatus);
    // create kernel ELL and set arguments
    cl_kernel kernelELL = clCreateKernel(program, "mELLxVec", &clStatus);
    clStatus |= clSetKernelArg(kernelELL, 0, sizeof(cl_mem), (void *)&mJDS_col);
    printf("cl stat4-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void *)&mJDS_data);
    printf("cl stat5-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void *)&mJDS_row_permute);
    printf("cl stat6-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void *)&vecIn);
    printf("cl stat7-> %d\n", clStatus);
    clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_mem), (void *)&vecOut);
    printf("cl stat8-> %d\n", clStatus);
    

    // vecOut bo skos isti, notr je ++, pazi ce bos spremenu na paralelno enqueue kernel
    int curr_jag_size = 0;
    int low_limit = 0;
    int upper_limit = 0;
    int curr_jag_rows = 0;
    int count_row_permute = 0;
    int count_len = mJDS.max_elementsinrow; // count for every iteration, max len = 4, count_len = 4,3,2,1
    for(int i = 0; i < mJDS.size_of_jaggged_ptr; i++){
        printf("count_len--: %d\n", count_len);
        low_limit = mJDS.jagged_ptr[i];
        upper_limit = mJDS.jagged_ptr[i+1];
        curr_jag_size = upper_limit - low_limit; // 8, 4
        if(curr_jag_size == 0){
            count_len--;
            continue;
        }
        // printf("curr_jag_size: %d\n",curr_jag_size);
        curr_jag_rows = curr_jag_size / count_len; // 2 vsakic v mojem primeru.
        // printf("curr_jag_rows: %d\n", curr_jag_rows);
        
        clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void *)&low_limit);
        printf("cl stat9-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 6, sizeof(cl_int), (void *)&count_row_permute);
        printf("cl stat10-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 7, sizeof(cl_int), (void *)&curr_jag_rows); // mELL.num_rows
        printf("cl stat11-> %d\n", clStatus);
        clStatus |= clSetKernelArg(kernelELL, 8, sizeof(cl_int), (void *)&count_len); // mELL.num_elementsinrow
        printf("cl stat12-> %d\n", clStatus);
        size_t local_item_size = WORKGROUP_SIZE; // 1024
        int num_groups = ((curr_jag_rows - 1) / local_item_size + 1);
        size_t global_item_size_ELL = num_groups * local_item_size; // 1024

        clStatus = clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL,						
                                          &global_item_size_ELL, &local_item_size, 0, NULL, NULL);	
        printf("cl stat12-> %d\n", clStatus);
        count_row_permute += curr_jag_rows;
        count_len--;
    }

    clStatus = clEnqueueReadBuffer(command_queue, vecOut, CL_TRUE, 0,						
                                    mELL.num_rows*sizeof(cl_double), vecOutJDS, 0, NULL, NULL);
    printf("cl stat6-> %d\n", clStatus);
    for(int i = 0; i < mELL.num_rows; i++){
        printf("result: %lf\n", vecOutJDS[i]);
    }

/*
    // allocate memory on device and transfer data from host CSR
    cl_mem mCSRrowptr_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                         (mCSR.num_rows + 1) * sizeof(cl_int), NULL, &clStatus);
    cl_mem mCSRcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                      mCSR.num_nonzeros * sizeof(cl_int),NULL, &clStatus);
    cl_mem mCSRdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                       mCSR.num_nonzeros * sizeof(cl_double), NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRrowptr_d, CL_TRUE, 0,						
                                    (mCSR.num_rows + 1) * sizeof(cl_int), mCSR.rowptr, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRcol_d, CL_TRUE, 0,						
                                    mCSR.num_nonzeros * sizeof(cl_int), mCSR.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mCSRdata_d, CL_TRUE, 0,						
                                    mCSR.num_nonzeros * sizeof(cl_double), mCSR.data, 0, NULL, NULL);				

    // allocate memory on device and transfer data from host ELL
    cl_mem mELLcol_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                      mELL.num_elements * sizeof(cl_int), NULL, &clStatus);
    cl_mem mELLdata_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                       mELL.num_elements * sizeof(cl_double), NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, mELLcol_d, CL_TRUE, 0,						
                                    mELL.num_elements * sizeof(cl_int), mELL.col, 0, NULL, NULL);				
    clStatus = clEnqueueWriteBuffer(command_queue, mELLdata_d, CL_TRUE, 0,						
                                    mELL.num_elements * sizeof(cl_double), mELL.data, 0, NULL, NULL);				

    // vectors
    cl_mem vecIn_d = clCreateBuffer(context, CL_MEM_READ_ONLY, 
								    mCOO.num_cols * sizeof(cl_double), vecIn, &clStatus);
    cl_mem vecOut_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                     mCOO.num_rows * sizeof(cl_double), NULL, &clStatus);

    //allocate memory on device and transfer data from host JDS


  
    // create kernel CSR and set arguments
    cl_kernel kernelCSR = clCreateKernel(program, "mCSRxVec", &clStatus);
    clStatus  = clSetKernelArg(kernelCSR, 0, sizeof(cl_mem), (void *)&mCSRrowptr_d);
    clStatus |= clSetKernelArg(kernelCSR, 1, sizeof(cl_mem), (void *)&mCSRcol_d);
    clStatus |= clSetKernelArg(kernelCSR, 2, sizeof(cl_mem), (void *)&mCSRdata_d);
    clStatus |= clSetKernelArg(kernelCSR, 3, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelCSR, 4, sizeof(cl_mem), (void *)&vecOut_d);
	clStatus |= clSetKernelArg(kernelCSR, 5, sizeof(cl_int), (void *)&(mCSR.num_rows));

    // create kernel ELL and set arguments
    cl_kernel kernelELL = clCreateKernel(program, "mELLxVec", &clStatus);
    clStatus  = clSetKernelArg(kernelELL, 0, sizeof(cl_mem), NULL);
    clStatus |= clSetKernelArg(kernelELL, 1, sizeof(cl_mem), (void *)&mELLcol_d);
    clStatus |= clSetKernelArg(kernelELL, 2, sizeof(cl_mem), (void *)&mELLdata_d);
    clStatus |= clSetKernelArg(kernelELL, 3, sizeof(cl_mem), (void *)&vecIn_d);
    clStatus |= clSetKernelArg(kernelELL, 4, sizeof(cl_mem), (void *)&vecOut_d);
	clStatus |= clSetKernelArg(kernelELL, 5, sizeof(cl_int), (void *)&(mELL.num_rows));
	clStatus |= clSetKernelArg(kernelELL, 6, sizeof(cl_int), (void *)&(mELL.num_elementsinrow));

	// Divide work CSR
    size_t local_item_size = WORKGROUP_SIZE;
	int num_groups = (mCSR.num_rows - 1) / local_item_size + 1;
    size_t global_item_size_CSR = num_groups * local_item_size;

	// Divide work ELL
	num_groups = ((mELL.num_rows - 1) / local_item_size + 1);
    size_t global_item_size_ELL = num_groups * local_item_size;

	// CSR write, execute, read
    double dtimeCSR_cl = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++)
    {
        clStatus = clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mCSR.num_cols*sizeof(cl_double), vecIn, 0, NULL, NULL);				
        clStatus = clEnqueueNDRangeKernel(command_queue, kernelCSR, 1, NULL,						
                                        &global_item_size_CSR, &local_item_size, 0, NULL, NULL);	
        clStatus = clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                        mCSR.num_rows*sizeof(cl_double), vecOutCSR_cl, 0, NULL, NULL);				
    }
    dtimeCSR_cl = omp_get_wtime()-dtimeCSR_cl;
																						
	// ELL write, execute, read
    double dtimeELL_cl = omp_get_wtime();
    for (repeat = 0; repeat < REPEAT; repeat++)
    {
        clStatus = clEnqueueWriteBuffer(command_queue, vecIn_d, CL_TRUE, 0,						
                                        mELL.num_cols*sizeof(cl_double), vecIn, 0, NULL, NULL);				
        clStatus = clEnqueueNDRangeKernel(command_queue, kernelELL, 1, NULL,						
                                          &global_item_size_ELL, &local_item_size, 0, NULL, NULL);	
        clStatus = clEnqueueReadBuffer(command_queue, vecOut_d, CL_TRUE, 0,						
                                    mELL.num_rows*sizeof(cl_double), vecOutELL_cl, 0, NULL, NULL);				
    }
    dtimeELL_cl = omp_get_wtime()-dtimeELL_cl;
*/
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    /* clStatus = clReleaseKernel(kernelCSR);
    clStatus = clReleaseKernel(kernelELL); */
    clStatus = clReleaseProgram(program);
    /* clStatus = clReleaseMemObject(mCSRrowptr_d);
    clStatus = clReleaseMemObject(mCSRcol_d);
    clStatus = clReleaseMemObject(mCSRdata_d);
    clStatus = clReleaseMemObject(mELLcol_d);
    clStatus = clReleaseMemObject(mELLdata_d);
    clStatus = clReleaseMemObject(vecIn_d);
    clStatus = clReleaseMemObject(vecOut_d); */
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
	free(devices);
    free(platforms);
/* 
    ///////////////////////
    // OpenCL code - end //
    ///////////////////////

    // output
    printf("size: %ld x %ld, nonzero: %ld, max elems in row: %d\n", mCOO.num_rows, mCOO.num_cols, mCOO.num_nonzeros, mELL.num_elementsinrow);
    int errorsCSR_seq = 0;
    int errorsELL_seq = 0;
    int errorsCSR_cl = 0;
    int errorsELL_cl = 0;
    for(int i = 0; i < mCOO.num_rows; i++)
    {
        if (fabs(vecOutCOO_seq[i]-vecOutCSR_seq[i]) > 1e-4 )
            errorsCSR_seq++;
        if (fabs(vecOutCOO_seq[i]-vecOutELL_seq[i]) > 1e-4 )
            errorsELL_seq++;
        if (fabs(vecOutCOO_seq[i]-vecOutCSR_cl[i]) > 1e-4 )
            errorsCSR_cl++;
        if (fabs(vecOutCOO_seq[i]-vecOutELL_cl[i]) > 1e-4 )
            errorsELL_cl++;
    }
    printf("Errors: %d(CSR_seq), %d(ELL_seq), %d(CSR_cl), %d(ELL_cl)\n", 
           errorsCSR_seq, errorsELL_seq, errorsCSR_cl, errorsELL_cl);
    printf("Times: %lf(COO_seq), %lf(CSR_seq), %lf(ELL_seq)\n", dtimeCOO_seq, dtimeCSR_seq, dtimeELL_seq);
    printf("Times: %lf(CSR_cl), %lf(ELL_cl)\n\n", dtimeCSR_cl, dtimeELL_cl);

    // deallocate
    free(vecIn);
    free(vecOutCOO_seq);
    free(vecOutCSR_seq);
    free(vecOutELL_seq);
    free(vecOutCSR_cl);
    free(vecOutELL_cl);
 */

    mtx_COO_free(&mCOO);
    mtx_CSR_free(&mCSR);
    mtx_ELL_free(&mELL);

	return 0;
}
