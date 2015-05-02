#ifndef RNNLMRUSSIANMORPHOLOGY_H
#define RNNLMRUSSIANMORPHOLOGY_H

#include <string.h>
#include "common.h"
#include "cuda-frontend.h"

namespace RNNLM
{

struct ModelOptions
{
    ModelOptions(int i_layer1_size, int i_bptt, int i_bptt_block, bool i_independent = false) :
        independent(i_independent)
      , bptt(i_bptt)
      , bptt_block(i_bptt_block)
      , layer1_size(i_layer1_size)
    {}

    bool independent;
    int bptt;
    int bptt_block;
    int layer1_size;
};

class RnnlmRussianMorphology
{
public:
    RnnlmRussianMorphology() : m_logProb(double(0)) {}
    void saveNet();
    void saveWeights();			//saves current weights and unit activations
    void restoreWeights();
    void readFromFile(FILE *fi, FileTypeEnum filetype);
    void netFlush();
    void clearMemory();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)

    void computeNet(int last_word, int last_morph, int word, int morph);
    void learnNet(int last_word, int last_morph, int word, int morph, double alpha, double beta, int counter);
    void copyHiddenLayerToInput();
    bool independent() const {return m_independent;}
    void initNet(int i_vocabSize, int i_morphSize, const ModelOptions& i_options);
    double logProb() {return double(m_logProb);}
    void resetLogProb() { m_logProb = 0;}
    void writeToFile(FILE *fo, FileTypeEnum filetype);

private:
    CudaValue<double> m_logProb;

    int m_vocabSize;
    int m_morphologySize;
    int layer1_size;

    Layer neu0; // copy of previous recurrent layer
    Layer neu1; //recurrent layer
    Layer neu2v; // output layer for lemmas
    Layer neu2m; // output layer for morphology

    Matrix syn0v; //lemma lookup table
    Matrix syn0m; //morphology lookup table
    Matrix syn0h; //recurrent weight matrix
    Matrix syn1v;  //weights between hidden and lemmas output layer
    Matrix syn1m;  //weights between hidden and morphology output layer

    //backup used in training:
    Layer neu0b;
    Layer neu1b;
    Layer neu2vb;
    Layer neu2mb;

    Matrix syn0vb;
    Matrix syn0hb;
    Matrix syn0mb;
    Matrix syn1vb;
    Matrix syn1mb;

    bool m_independent;
    int bptt;
    int bptt_block;
    std::vector<int> bptt_history_m;
    std::vector<int> bptt_history_v;
    std::vector<Layer> bptt_hidden;
    Matrix bptt_syn0h;
    Matrix bptt_syn0v;
    Matrix bptt_syn0m;

    void computeRecurrentLayer_(int i_wordIndex, int i_morphIndex);
    void computeOutputLayer_();
    void computeErrorOnOutput_(int i_trueWord, int i_trueMorph);
    void applyGradient_(double i_lr, const Vector& i_next, const Vector& i_prev, Matrix& i_mat, double beta);
    void applyGradient_(double i_lr, const Vector& i_next, int i_column, Matrix& i_mat, double beta);
    void computeErrorOnPrevious_(const Layer& i_nextLayer, Matrix& i_synMat, Layer& i_prevLayer);
    void computeErrorOnHidden_(const Layer& i_outVLayer, const Layer& i_outMLayer, Matrix& i_synVMat, Matrix& i_synMMat, Layer& i_hiddenLayer);
    void addGradient_(const Matrix& i_update, Matrix& o_mat , double beta2);
    void applyGradient_(const Matrix& i_update, int i_updateColumn, Matrix& io_target, int i_targetColumn, double i_beta);
    void updateBptt_(int last_word, int last_morph);
    void makeBptt_(int word, int morph, double alpha, double beta, int counter);
    void initNet_();
    void incremetLogProbByWordLP_(int word, int morph);

            //restores current weights and unit activations from backup copy
    void saveContext();
    void restoreContext();
};

}

#endif // RNNLMRUSSIANMORPHOLOGY_H
