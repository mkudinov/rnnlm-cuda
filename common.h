#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>

#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

#define MAX_STRING 100

namespace RNNLM
{
    enum FileTypeEnum {TEXT, BINARY};
    void goToDelimiter(int delim, FILE *fi);
    double random(double min, double max);
}

#endif // COMMON_H
