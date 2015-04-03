#include "common.h"

namespace RNNLM
{
double random(double min, double max)
{
    return rand()/(double)RAND_MAX*(max-min)+min;
}

void goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(fi);
        if (feof(fi)) {
            printf("Unexpected end of file\n");
            exit(1);
        }
    }
}
}

