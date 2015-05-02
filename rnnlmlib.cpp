///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"
#include <sys/time.h>
#include <iostream>

namespace RNNLM
{
void Snapshot::readFromFile(FILE *fi)
{
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    char buff[100];
    goToDelimiter(':', fi);
    fscanf(fi, "%s", buff);
    train_file = buff;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", buff);
    valid_file = buff;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    int train_words;
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &starting_alpha);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &alpha);

    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
}

void Snapshot::writeToFile(FILE *fo)
{
    fprintf(fo, "RNNLM model file\n");
    fprintf(fo, "file format: %d\n\n", filetype);
    fprintf(fo, "training data file: %s\n", train_file.c_str());
    fprintf(fo, "validation data file: %s\n\n", valid_file.c_str());
    fprintf(fo, "number of finished iterations: %d\n", iter);
    fprintf(fo, "# of training words: %d\n", train_words);
    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");
}


}
