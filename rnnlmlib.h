#ifndef RNNLMLIB_H_
#define RNNLMLIB_H_

#include <assert.h>
#include "common.h"
#include "vocabulary.h"
#include "classic-rnnlm.h"

namespace RNNLM
{

struct Snapshot
{
    FileTypeEnum filetype;
    double gradient_cutoff;
    double alpha;
    double starting_alpha;
    bool alpha_divide;
    double beta;
    float min_improvement;
    int train_words;
    int iter;
    std::string valid_file;
    std::string train_file;

    void readFromFile(FILE *fi);
    void writeToFile(FILE *fo);
};

class CRnnLM
{
public:

    CRnnLM();

    int alpha_set, train_file_set;    
    double random(double min, double max);
    
    void setFileType(FileTypeEnum newt) {filetype=newt;}
    void setGradientCutoff(double newGradient) {gradient_cutoff=newGradient;}
    
    void setLearningRate(double newAlpha) {alpha=newAlpha;}
    void setRegularization(double newBeta) {beta=newBeta;}
    void setMinImprovement(double newMinImprovement) {min_improvement=newMinImprovement;}
    void setRandSeed(int newSeed) {srand(newSeed);}
    void setDebugMode(int newDebug) {debug_mode=newDebug;}
    void trainNet(char *train_file, char *valid_file, char *snapshot_file, const ModelOptions& i_options);

private:
    void restoreFromSnapshot_(char *i_snapshot_file, Vocabulary& o_vocab, ClassicRnnlm& o_model);

    void initTraining_(char *train_file, char *valid_file, char *snapshot_file, const ModelOptions& i_options);
    std::tuple<double, clock_t, int>  learningPhase_();
    std::tuple<double, int> validationPhase_();
    void saveSnapshot_(const std::string& i_trainFileName, const std::string& i_validFileName, const std::string& i_snapshotFileName);

    int debug_mode;
    FileTypeEnum filetype;

    double gradient_cutoff;
    double alpha;
    double starting_alpha;
    bool alpha_divide;
    double beta;
    float min_improvement;
    int iter;

    ClassicRnnlm m_model;
    Vocabulary m_vocab;
    int m_vocabSize;
    InputSequence m_trainSource;
    InputSequence m_validSource;
    int m_trainWords;
};

}

#endif
