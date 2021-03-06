#include <stdio.h>
#include <stdlib.h>
#include "mtx_sparse.h"

struct mtx_MM
{
    int row;
    int col;
    double data;
};

int mtx_COO_compare(const void * a, const void * b)
{   
    struct mtx_MM aa = *(struct mtx_MM *)a;
    struct mtx_MM bb = *(struct mtx_MM *)b;

    if (aa.row < bb.row)
        return -1;
    else if (aa.row > bb.row)
        return +1;
    else if (aa.col < bb.col)
        return -1;
    else if (aa.col > bb.col)
        return +1;
    else 
        return 0;
}

int mtx_COO_create_from_file(struct mtx_COO *mCOO, FILE *f)
{
    char line[1024];

    // skip comments
    do 
    {
        if (fgets(line, 1024, f) == NULL) 
            return 1;
    } 
    while (line[0] == '%');
    // get matrix size
    if (sscanf(line, "%d %d %d", &(mCOO->num_rows), &(mCOO->num_cols), &(mCOO->num_nonzeros)) != 3)
        return 1;
    // allocate matrix
    struct mtx_MM *mMM = (struct mtx_MM *)malloc(mCOO->num_nonzeros * sizeof(struct mtx_MM));
    mCOO->data = (double *) malloc(mCOO->num_nonzeros * sizeof(double));
    mCOO->col = (int *) malloc(mCOO->num_nonzeros * sizeof(int));
    mCOO->row = (int *) malloc(mCOO->num_nonzeros * sizeof(int));
    // read data
    for (int i = 0; i < mCOO->num_nonzeros; i++)
    {
        fscanf(f, "%d %d %lf\n", &mMM[i].row, &mMM[i].col, &mMM[i].data);
        mMM[i].row--;  /* adjust from 1-based to 0-based row/column */
        mMM[i].col--;
    }    
    fclose(f);

    // sort elements
    qsort(mMM, mCOO->num_nonzeros, sizeof(struct mtx_MM), mtx_COO_compare);

    // copy to mtx_COO structures (GPU friendly)
    for (int i = 0; i < mCOO->num_nonzeros; i++)
    {
        mCOO->data[i] = mMM[i].data;
        mCOO->row[i] = mMM[i].row;
        mCOO->col[i] = mMM[i].col;
    }
    

    free(mMM);

    return 0;
}

int mtx_COO_free(struct mtx_COO *mCOO)
{
    free(mCOO->data);
    free(mCOO->col);
    free(mCOO->row);

    return 0;
}

int mtx_CSR_create_from_mtx_COO(struct mtx_CSR *mCSR, struct mtx_COO *mCOO)
{
    mCSR->num_nonzeros = mCOO->num_nonzeros;
    mCSR->num_rows = mCOO->num_rows;
    mCSR->num_cols = mCOO->num_cols;

    mCSR->data =  (double *)malloc(mCSR->num_nonzeros * sizeof(double));
    mCSR->col = (int *)malloc(mCSR->num_nonzeros * sizeof(int));
    mCSR->rowptr = (int *)calloc(mCSR->num_rows + 1, sizeof(int));
    mCSR->data[0] = mCOO->data[0];
    mCSR->col[0] = mCOO->col[0];
    mCSR->rowptr[0] = 0;
    mCSR->rowptr[mCSR->num_rows] = mCSR->num_nonzeros;
    for (int i = 1; i < mCSR->num_nonzeros; i++)
    {
        mCSR->data[i] = mCOO->data[i];
        mCSR->col[i] = mCOO->col[i];
        if (mCOO->row[i] > mCOO->row[i-1])
        {
            int r = mCOO->row[i];
            while (r > 0 && mCSR->rowptr[r] == 0)
                mCSR->rowptr[r--] = i;
        }
    }
    

    return 0;
}

int mtx_CSR_free(struct mtx_CSR *mCSR)
{
    free(mCSR->data);
    free(mCSR->col);
    free(mCSR->rowptr);

    return 0;
}

int mtx_ELL_create_from_mtx_CSR(struct mtx_ELL *mELL, struct mtx_CSR *mCSR)
{
    mELL->num_nonzeros = mCSR->num_nonzeros;
    mELL->num_rows = mCSR->num_rows;
    mELL->num_cols = mCSR->num_cols;
    mELL->num_elementsinrow = 0;

    for (int i = 0; i < mELL->num_rows; i++)
        if (mELL->num_elementsinrow < mCSR->rowptr[i+1]-mCSR->rowptr[i]) 
            mELL->num_elementsinrow = mCSR->rowptr[i+1]-mCSR->rowptr[i];
    mELL->num_elements = mELL->num_rows * mELL->num_elementsinrow;
    mELL->data = (double *)calloc(mELL->num_elements, sizeof(double));
    mELL->col = (int *) calloc(mELL->num_elements, sizeof(int));    
    for (int i = 0; i < mELL->num_rows; i++)
    {
        for (int j = mCSR->rowptr[i]; j < mCSR->rowptr[i+1]; j++)
        {            
            int ELL_j = (j - mCSR->rowptr[i]) * mELL->num_rows + i;
            mELL->data[ELL_j] = mCSR->data[j];
            mELL->col[ELL_j] = mCSR->col[j];
        }
    }

    return 0;
}

int mtx_ELL_free(struct mtx_ELL *mELL)
{
    free(mELL->col);
    free(mELL->data);

    return 0;
}

int findAllNextMax(int *same_jag, int *el_per_row, int len, int JAGPADD, int *arr_found_max){
    int curr_max_element = -1;
    for(int i = 0; i < len; i++){
        if(el_per_row[i] > curr_max_element){
            curr_max_element = el_per_row[i];
        }
    }
    if(curr_max_element == -1){
        return 0;
    }
    else{ // find all same
        int count = 0;
        arr_found_max[1] = curr_max_element;
        for(int i = 0; i < len; i++){
            if(curr_max_element <= el_per_row[i] + JAGPADD && el_per_row[i] > 0){
                same_jag[count] = i;
                el_per_row[i] = -2;
                count++;
            }
        }
        arr_found_max[0] = count;
        return count;
    }
}

int mtx_JDS_create_from_mtx_CSR(struct mtx_JDS *mJDS, struct mtx_CSR *mCSR, int JAGPADD)
{
    mJDS->num_nonzeros = mCSR->num_nonzeros;
    mJDS->num_rows = mCSR->num_rows;
    mJDS->num_cols = mCSR->num_cols;
    mJDS->jag_padd = JAGPADD;

    //tok je max, loh tut mn
    mJDS->data =  (double *)malloc((mCSR->num_nonzeros) *10* sizeof(double));
    mJDS->col = (int *)malloc((mCSR->num_nonzeros) *10* sizeof(int));

    //max sta tok kr je vrstic, lahko tudi manj!
    mJDS->row_permute = (int *)malloc(mCSR->num_rows * sizeof(int));
    mJDS->jagged_ptr = (int *)malloc(mCSR->num_rows * sizeof(int));

    int *el_per_row = (int *)malloc(mCSR->num_rows * sizeof(int));
    
    for(int i = 0; i < mCSR->num_rows; i++){
        el_per_row[i] = mCSR->rowptr[i+1] - mCSR->rowptr[i];
        // printf("el per row-> %d\n", el_per_row[i]);
    } 

    int *same_jag = (int *)malloc(mCSR->num_nonzeros * sizeof(int));
    int *arr_found_max = (int *)malloc(2 * sizeof(int));
    int found = findAllNextMax(same_jag, el_per_row, mJDS->num_rows, JAGPADD, arr_found_max);

    mJDS->max_elementsinrow = arr_found_max[1];

    int data_count = 0;
    int jag_pointer_count = 0;
    int row_permute_count = 0;
    int jag_count = 0;
    int col_major_count = 0;
    int prev_jag_len = arr_found_max[1] - 1;
    int count = 0;
    while(found){
        
        // printf("found -> %d, max-> %d\n", arr_found_max[0], arr_found_max[1]);
        int curr_jag_len = arr_found_max[1]; // curr max len, padd all to this one. 
        if(mCSR->rowptr[same_jag[0]+1] - mCSR->rowptr[same_jag[0]] == 0){
            break;
        }
        
        for(int i = 0; i < prev_jag_len-curr_jag_len-1; i++){ // add zero lines that had 1 element less.
            mJDS->jagged_ptr[jag_pointer_count] = data_count;
            jag_pointer_count++;
        }
        

        mJDS->jagged_ptr[jag_pointer_count] = data_count;
        
        
        for(int i = 0; i < found; i++){
            // printf("same_jag[i]: %d, found : %d\n", same_jag[i], found);
            mJDS->row_permute[row_permute_count] = same_jag[i];

            int curr_found_len = mCSR->rowptr[same_jag[i]+1] - mCSR->rowptr[same_jag[i]];
            
            int diff = arr_found_max[1] - curr_found_len; // difference between max and curr in found.
            
            // printf("curr: %d, curr_found: %d, diff--> %d\n",arr_found_max[1],curr_found_len, diff);
            col_major_count = 0;
            
            for(int j = mCSR->rowptr[same_jag[i]]; j < mCSR->rowptr[same_jag[i]+1] + diff; j++){
            
                
                if(j >= mCSR->rowptr[same_jag[i]+1]){ // add zero
                    
                    mJDS->data[data_count + col_major_count*found+i] = 0;
                    mJDS->col[data_count + col_major_count*found+i] = 0;
                    
                }
                else{
                    
                    mJDS->data[data_count + col_major_count*found+i] = mCSR->data[j];
                    mJDS->col[data_count + col_major_count*found+i] = mCSR->col[j];
                    
                }
                
                col_major_count++;
                
            }
            
            


            row_permute_count++;
        }
        

        prev_jag_len = curr_jag_len;
        data_count += col_major_count*found;
        

        found = findAllNextMax(same_jag, el_per_row, mJDS->num_rows,JAGPADD, arr_found_max);
        
        // printf("---\n");
        jag_pointer_count++;
        jag_count++;
    }
    mJDS->jagged_ptr[jag_pointer_count] = data_count;
    mJDS->num_of_jags_nonzero = jag_count;
    mJDS->size_of_jaggged_ptr = jag_pointer_count;
    mJDS->jds_rows = row_permute_count;
    mJDS->data_arr_length = data_count;

    
    free(same_jag);
    return 0;
}

int mtx_JDS_free(struct mtx_JDS *mJDS)
{
    /* free(mJDS->col);
    free(mJDS->data);
    free(mJDS->jagged_ptr);
    free(mJDS->row_permute); */
    return 0;
}
