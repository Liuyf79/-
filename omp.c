// ######################################################
// ##  姓名: 刘羽丰
// ##  文件说明: openmp版本
// ######################################################
#include <stdio.h>
#include <windows.h>
#include <omp.h>
#include <time.h>
#include <math.h>

// ######################################################
// ##  INIT函数,将主机端的两数组进行初始化
// ##  A_host: 输入的矩阵,初始化时随机0-15
// ##  B_host: 输出的矩阵,初始化时初始为0
// ##  ROWS,COLS分别为矩阵的高和宽
// ######################################################
void INIT(int* A,float* B,int ROWS,int COLS){
    srand(time(NULL));
    for(int i = 0; i < ROWS; i++){
        for(int j = 0; j < COLS; j++){
            A[i*COLS+j] = rand() % 16;
            B[i*COLS+j] = 0;
        }
    }
}
// log数组
const float mylog[25] = {
            0.000000,
            0.693147,
            1.098612,
            1.386294,
            1.609438,
            1.791759,
            1.945910,
            2.079442,
            2.197225,
            2.302585,
            2.397895,
            2.564949,
            2.639057,
            2.708050,
            2.772589,
            2.833213,
            2.890372,
            2.944439,
            2.995732,
            3.044522,
            3.091042,
            3.135494,
            3.178054};
// ######################################################
// ##  计算函数,用于计算二维数组中以每个元素为中心的熵
// ##  A: 输入的矩阵
// ##  B: 输出的矩阵
// ##  rows,cols分别为矩阵的高和宽
// ######################################################
void cal_entropy(int *A, float *B, int rows, int cols){
    #pragma omp parallel for num_threads(16)
    for(int idx = 0; idx < rows*cols; ++idx){
        // 变换得到坐标
        int row = idx / cols;
        int col = idx - row*cols;
        // 避免大量重复if，计算窗口四边界
        int up = max(row-2,0);
        int down = min(row+2,rows-1);
        int left = max(col-2,0);
        int right = min(col+2,cols-1);
        // 利用char存储降低寄存器压力
        char digit[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // printf("(%d,%d) l:%d,r:%d,u:%d,d:%d\n",row,col,left,right,up,down);
        // 根据窗口四边界可直接得到窗口大小
        int count = (right-left+1)*(down-up+1);
        // 记录熵
        float result = 0;
        // 中间值
        float temp = 0;
        // 遍历周围元素并统计
        for(int i = up; i <= down; i++){
            for(int j = left; j <= right; j++){
                ++digit[A[i*cols+j]];
            }
        }
        // 非查表
        // for(int i = 0; i < 16; i++)
        //     if(digit[i] != 0){
        //         temp = (float)digit[i]/count;
        //         result += temp*logf(temp);
        //     }
        // 查表,计算熵
        for(int i = 0; i < 16; i++)
            if(digit[i] != 0){
                result += (((float)digit[i])/count)*(mylog[digit[i]-1]-mylog[count-1]);
            }
        B[idx] = -result;
        // printf("c:%d idx:%d r:%f\n",count,idx,B[idx]);
    }
}
int main(int argc,char *argv[])
{   
    int i;
    // 矩阵高度
    int ROWS = 5;
    // 矩阵宽度
    int COLS = 5;

    for(i = 1; i < argc; i++)
    {
        if(i == 1){
            ROWS = atoi(argv[i]);
        }
        else if(i == 2){
            COLS = atoi(argv[i]);
        }
    }
    //输入二维矩阵，4096*4096，单精度浮点型。
    int Bytes = ROWS*COLS*sizeof(int);
    int FBytes = ROWS*COLS*sizeof(float);
    //开辟主机内存
    int* A = (int*)malloc(Bytes);
    float* B = (float*)malloc(FBytes);

    INIT(A,B,ROWS,COLS);

    double start = omp_get_wtime();
    cal_entropy(A,B,ROWS,COLS);
    double end = omp_get_wtime();
    printf("omp_time:%fms\n",(end-start)*1000);
    
    free(A);
    free(B);
    return 0;
}
// gcc -fopenmp omp.c -o main